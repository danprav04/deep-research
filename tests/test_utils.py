
# tests/test_utils.py (Placeholder)
import pytest
import json
from deep_research_app import utils
from deep_research_app.config import MIN_SANITIZED_CONTENT_LENGTH

# --- Test Data ---

PLAN_LLM_RESPONSE_JSON_BLOCK = """
Some introductory text.
```json
[
  {"step": "Step 1", "keywords": ["kw1", "kw2"]},
  {"step": "Step 2", "keywords": ["kw3"]}
]
```
Some trailing text.
"""

PLAN_LLM_RESPONSE_DIRECT_LIST = """
[
  {"step": "Step 1 Direct", "keywords": ["kw1d", "kw2d"]},
  {"step": "Step 2 Direct", "keywords": ["kw3d", 123]}
]
"""

PLAN_LLM_RESPONSE_NESTED_LIST = """
{
  "comment": "Here is the plan",
  "plan": [
    {"step": "Step 1 Nested", "keywords": ["kw1n"]},
    {"step": "Step 2 Nested", "keywords": ["kw2n", "kw3n"]}
  ]
}
"""

PLAN_LLM_RESPONSE_INVALID_JSON = """
```json
[
  {"step": "Step 1", "keywords": ["kw1", "kw2"]},
  {"step": "Step 2", "keywords": ["kw3"]},สังเคราะห์
]
```
"""

PLAN_LLM_RESPONSE_INVALID_STRUCTURE = """
```json
{
  "step": "Not a list", "keywords": []
}
```
"""

PLAN_LLM_RESPONSE_MISSING_KEYS = """
```json
[
  {"step": "Step 1", "keywords": ["k1"]},
  {"description": "Step 2", "terms": ["k2"]}
]
```
"""

PLAN_LLM_RESPONSE_STRING_KEYWORDS = """
```json
[
  {"step": "Step 1", "keywords": "kw1, kw2"},
  {"step": "Step 2", "keywords": ["kw3"]}
]
```
"""

PLAN_LLM_RESPONSE_NO_JSON = "This is just text, no JSON block."

# --- Tests for parse_research_plan ---

@pytest.mark.parametrize("response, expected_steps, expected_keywords", [
    (PLAN_LLM_RESPONSE_JSON_BLOCK, ["Step 1", "Step 2"], [["kw1", "kw2"], ["kw3"]]),
    (PLAN_LLM_RESPONSE_DIRECT_LIST, ["Step 1 Direct", "Step 2 Direct"], [["kw1d", "kw2d"], ["kw3d", "123"]]), # Expects string conversion
    (PLAN_LLM_RESPONSE_NESTED_LIST, ["Step 1 Nested", "Step 2 Nested"], [["kw1n"], ["kw2n", "kw3n"]]),
    (PLAN_LLM_RESPONSE_STRING_KEYWORDS, ["Step 1", "Step 2"], [["kw1", "kw2"], ["kw3"]]), # Expects keyword fix
])
def test_parse_research_plan_success(response, expected_steps, expected_keywords):
    """Test successful parsing from various valid formats."""
    plan = utils.parse_research_plan(response)
    assert isinstance(plan, list)
    assert len(plan) == len(expected_steps)
    for i, step in enumerate(plan):
        assert step.get("step") == expected_steps[i]
        assert step.get("keywords") == expected_keywords[i]

@pytest.mark.parametrize("response, expected_error_msg_part", [
    ("", "LLM response was empty"),
    ("   ", "LLM response was empty"),
    (PLAN_LLM_RESPONSE_INVALID_JSON, "Failed to parse/validate JSON"),
    (PLAN_LLM_RESPONSE_INVALID_STRUCTURE, "Parsed JSON is not a list"),
    (PLAN_LLM_RESPONSE_MISSING_KEYS, "missing required keys"),
    (PLAN_LLM_RESPONSE_NO_JSON, "Could not find or parse JSON structure"),
])
def test_parse_research_plan_failure(response, expected_error_msg_part):
    """Test parsing failures for various invalid inputs."""
    plan = utils.parse_research_plan(response)
    assert isinstance(plan, list)
    assert len(plan) == 1 # Failure indicated by a single-item list
    assert plan[0]["step"].startswith("Failed")
    assert expected_error_msg_part in plan[0]["step"]
    assert "keywords" in plan[0] and plan[0]["keywords"] == []

# --- Test Data for Bibliography ---
SCRAPED_SOURCES_VALID = [
    {'url': 'http://example.com/page1', 'temp_filepath': '/tmp/1.txt'},
    {'url': 'https://example.org/article', 'temp_filepath': '/tmp/2.txt'},
    {'url': 'http://example.com/page3?query=1', 'temp_filepath': '/tmp/3.txt'},
]
SCRAPED_SOURCES_DUPLICATE = [
    {'url': 'http://example.com/page1', 'temp_filepath': '/tmp/1.txt'},
    {'url': 'http://example.com/page1', 'temp_filepath': '/tmp/1_dup.txt'}, # Duplicate URL
    {'url': 'https://example.org/article', 'temp_filepath': '/tmp/2.txt'},
]
SCRAPED_SOURCES_INVALID = [
    {'url': 'http://example.com/page1', 'temp_filepath': '/tmp/1.txt'},
    {'url': 'ftp://example.com/file', 'temp_filepath': '/tmp/ftp.txt'}, # Invalid scheme
    {'no_url': 'value', 'temp_filepath': '/tmp/no_url.txt'}, # Missing URL key
    {'url': None, 'temp_filepath': '/tmp/none.txt'}, # URL is None
]

# --- Tests for generate_bibliography_map ---

def test_generate_bibliography_map_success():
    """Test generating a map from valid scraped sources."""
    url_map, prompt_list = utils.generate_bibliography_map(SCRAPED_SOURCES_VALID)

    assert isinstance(url_map, dict)
    assert len(url_map) == 3
    assert url_map['http://example.com/page1'] == 1
    assert url_map['https://example.org/article'] == 2
    assert url_map['http://example.com/page3?query=1'] == 3

    assert isinstance(prompt_list, str)
    expected_prompt = "[1]: http://example.com/page1\n[2]: https://example.org/article\n[3]: http://example.com/page3?query=1"
    assert prompt_list == expected_prompt

def test_generate_bibliography_map_duplicates():
    """Test handling of duplicate URLs."""
    url_map, prompt_list = utils.generate_bibliography_map(SCRAPED_SOURCES_DUPLICATE)

    assert len(url_map) == 2 # Only unique URLs included
    assert 'http://example.com/page1' in url_map
    assert 'https://example.org/article' in url_map
    assert url_map['http://example.com/page1'] == 1 # First occurrence gets index 1
    assert url_map['https://example.org/article'] == 2
    assert "[1]: http://example.com/page1\n[2]: https://example.org/article" in prompt_list

def test_generate_bibliography_map_invalid():
    """Test handling of invalid source entries."""
    url_map, prompt_list = utils.generate_bibliography_map(SCRAPED_SOURCES_INVALID)

    assert len(url_map) == 1 # Only the valid URL included
    assert 'http://example.com/page1' in url_map
    assert url_map['http://example.com/page1'] == 1
    assert prompt_list == "[1]: http://example.com/page1"

def test_generate_bibliography_map_empty():
    """Test with an empty list of sources."""
    url_map, prompt_list = utils.generate_bibliography_map([])
    assert url_map == {}
    assert prompt_list == ""

# --- Test Data for Sanitization ---
HTML_WITH_SCRIPTS = "<p>Good content<script>alert('XSS')</script></p><style>body{color:red}</style><iframe>"
HTML_WITH_ALLOWED_TAGS = "<h1>Title</h1><p>Paragraph with <b>bold</b> and <i>italic</i>.</p><ul><li>Item 1</li></ul><a href='http://safe.com'>Link</a>"
HTML_WITH_DISALLOWED_TAGS = "<p>Content <object>bad</object><form><input type='text'></form></p>"
HTML_WITH_ATTRIBUTES = "<p style='color:blue' class='foo' id='bar'>Styled para</p><a href='javascript:alert(1)' onclick='bad()'>Bad Link</a><img src='x' onerror='alert(1)'>"
PLAIN_TEXT = "Just simple text content."
EMPTY_HTML = "  "

# --- Tests for sanitize_scraped_content ---

def test_sanitize_removes_scripts_styles_iframes():
    sanitized = utils.sanitize_scraped_content(HTML_WITH_SCRIPTS)
    assert "<script" not in sanitized.lower()
    assert "<style" not in sanitized.lower()
    assert "<iframe" not in sanitized.lower()
    assert "Good content" in sanitized

def test_sanitize_keeps_allowed_tags():
    sanitized = utils.sanitize_scraped_content(HTML_WITH_ALLOWED_TAGS)
    assert "<h1>Title</h1>" in sanitized
    assert "<p>Paragraph with <b>bold</b> and <i>italic</i>.</p>" in sanitized
    assert "<ul><li>Item 1</li></ul>" in sanitized
    assert "<a href='http://safe.com'>Link</a>" in sanitized # Note: Bleach might reformat href quotes

def test_sanitize_removes_disallowed_tags():
    sanitized = utils.sanitize_scraped_content(HTML_WITH_DISALLOWED_TAGS)
    assert "<object" not in sanitized.lower()
    assert "<form" not in sanitized.lower()
    assert "<input" not in sanitized.lower()
    assert "Content" in sanitized

def test_sanitize_removes_disallowed_attributes():
    sanitized = utils.sanitize_scraped_content(HTML_WITH_ATTRIBUTES)
    assert "style=" not in sanitized.lower()
    assert "class=" not in sanitized.lower() # Class not in ALLOWED_ATTRS
    assert "id=" not in sanitized.lower()   # ID not in ALLOWED_ATTRS
    assert "onclick=" not in sanitized.lower()
    assert "onerror=" not in sanitized.lower()
    assert "javascript:" not in sanitized.lower() # Bleach removes unsafe protocols
    assert "<a href=\"\">Bad Link</a>" in sanitized or "<a>Bad Link</a>" in sanitized # href might be removed or empty if unsafe
    assert "<img src" not in sanitized.lower() # Img tag itself is removed

def test_sanitize_handles_plain_text():
    sanitized = utils.sanitize_scraped_content(PLAIN_TEXT)
    assert sanitized == PLAIN_TEXT # Plain text should pass through relatively unchanged

def test_sanitize_handles_empty_input():
    sanitized = utils.sanitize_scraped_content(EMPTY_HTML)
    assert sanitized == ""
    sanitized = utils.sanitize_scraped_content("")
    assert sanitized == ""

# --- Test Data for Markdown Conversion ---
MARKDOWN_BASIC = "# Title\n\nParagraph **bold** *italic*.\n\n* Item 1\n* Item 2"
MARKDOWN_WITH_FOOTNOTES = """
Here is text with a footnote[^1].

And another[^longnote].

[^1]: This is the first footnote.
[^longnote]: This is the second, longer footnote definition. It can contain paragraphs.

    Or code blocks.
"""
MARKDOWN_WITH_HTML = "## Subtitle\n\n<script>alert('bad')</script><p style='color:red'>Raw HTML para</p>"
MARKDOWN_TABLE = """
| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |
"""
MARKDOWN_TASKLIST = """
* [x] Done task
* [ ] Todo task
"""

# --- Tests for convert_markdown_to_html ---

def test_convert_markdown_basic():
    html = utils.convert_markdown_to_html(MARKDOWN_BASIC)
    assert html.startswith("<article>") and html.endswith("</article>")
    assert "<h1>Title</h1>" in html
    assert "<p>Paragraph <strong>bold</strong> <em>italic</em>.</p>" in html
    assert "<ul>" in html and "<li>Item 1</li>" in html

def test_convert_markdown_footnotes():
    html = utils.convert_markdown_to_html(MARKDOWN_WITH_FOOTNOTES)
    assert html.startswith("<article>") and html.endswith("</article>")
    # Check for footnote reference link
    assert '<sup class="footnote-ref"><a href="#fn:1" id="fnref:1">[1]</a></sup>' in html or '<sup class="footnote-ref"><a id="fnref:1" href="#fn:1">[1]</a></sup>' in html # Order might vary
    # Check for footnote definition list
    assert '<section class="footnotes">' in html
    assert '<li id="fn:1">' in html
    assert 'This is the first footnote.' in html
    assert '<a href="#fnref:1" class="footnote-backref">↩</a>' in html or '<a class="footnote-backref" href="#fnref:1">↩</a>' in html

def test_convert_markdown_disables_html():
    """Ensure raw HTML tags are escaped or removed."""
    html = utils.convert_markdown_to_html(MARKDOWN_WITH_HTML)
    assert html.startswith("<article>") and html.endswith("</article>")
    assert "<h2>Subtitle</h2>" in html
    # Check that script tag is NOT present as a tag
    assert "<script>" not in html.lower()
    # Check that inline style is NOT present
    assert "style=" not in html.lower()
    # Check that the paragraph tag itself might be present, but its content should be escaped
    # Depending on markdown-it config, it might escape the <p> or remove it.
    # Let's assert the dangerous parts are gone.
    assert "Raw HTML para" in html # Content should remain, but styling/scripting gone

def test_convert_markdown_table():
    html = utils.convert_markdown_to_html(MARKDOWN_TABLE)
    assert html.startswith("<article>") and html.endswith("</article>")
    assert "<table>" in html and "<thead>" in html and "<tbody>" in html
    assert "<th>Header 1</th>" in html
    assert "<td>Cell 3</td>" in html

def test_convert_markdown_tasklist():
     html = utils.convert_markdown_to_html(MARKDOWN_TASKLIST)
     assert html.startswith("<article>") and html.endswith("</article>")
     assert '<ul class="contains-task-list">' in html or '<ul class="task-list">' in html # Class name might vary slightly
     assert '<li class="task-list-item">' in html
     assert '<input type="checkbox" class="task-list-item-checkbox" disabled="" checked="">' in html or '<input checked="" class="task-list-item-checkbox" disabled="" type="checkbox">' in html
     assert '<input type="checkbox" class="task-list-item-checkbox" disabled="">' in html or '<input class="task-list-item-checkbox" disabled="" type="checkbox">' in html

def test_convert_markdown_empty():
    html = utils.convert_markdown_to_html("")
    assert "<p><em>Report content is empty or contains only whitespace.</em></p>" in html
    html = utils.convert_markdown_to_html("   ")
    assert "<p><em>Report content is empty or contains only whitespace.</em></p>" in html
