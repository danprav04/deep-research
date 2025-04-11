# utils.py
import re
import json
import html  # For escaping fallback text
from markdown_it import MarkdownIt
# REMOVED: from markdown_it.exceptions import MarkdownItError
from mdit_py_plugins.footnote import footnote_plugin

# --- REMOVED Check for Optional Dependencies (DOCX) ---

# --- Constants ---
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.MULTILINE)
CITATION_URL_PATTERN = re.compile(r"\[Source URL:\s*(https?://[^\s\]]+)\s*\]")

# Configure Markdown-it instance
# Enable HTML tags, links, typographer, tables, and footnotes
# Added 'commonmark' preset for robust base behavior
# Ensure extensions like 'table' are enabled.
md = (
    MarkdownIt("commonmark", {"html": True, "linkify": True, "typographer": True})
    .enable("table")
    .enable("strikethrough") # Optional: Enable strikethrough
    .use(footnote_plugin)
)


# --- Functions ---

def parse_research_plan(llm_response: str) -> list:
    """
    Parses the LLM response to extract the JSON research plan.
    """
    match = JSON_BLOCK_PATTERN.search(llm_response)
    if not match:
        try:
            potential_json = llm_response.strip()
            if potential_json.startswith('[') and potential_json.endswith(']'):
                 plan = json.loads(potential_json)
                 if isinstance(plan, list) and all(isinstance(item, dict) and 'step' in item and 'keywords' in item for item in plan):
                      print("INFO: Parsed research plan using fallback (no ```json block).")
                      return plan
                 else:
                      raise ValueError("Fallback JSON does not match expected structure.")
            else:
                 raise ValueError("No JSON block found and response doesn't look like a JSON list.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"ERROR: Failed to parse research plan. No valid JSON block found or fallback failed. Error: {e}")
            print(f"LLM Response Snippet: {llm_response[:200]}...")
            return [{"step": f"Failed to parse research plan: {e}", "keywords": []}]

    json_content = match.group(1).strip()
    try:
        plan = json.loads(json_content)
        if not isinstance(plan, list):
            raise TypeError("Parsed JSON is not a list.")
        for item in plan:
            if not isinstance(item, dict) or 'step' not in item or 'keywords' not in item:
                raise TypeError("Parsed list items do not have 'step' and 'keywords' keys.")
            if not isinstance(item['keywords'], list):
                 if isinstance(item['keywords'], str):
                      print(f"Warning: Fixing keywords field for step '{item['step']}' - was string, expected list.")
                      item['keywords'] = [kw.strip() for kw in item['keywords'].split(',') if kw.strip()]
                 else:
                      raise TypeError(f"Keywords for step '{item['step']}' is not a list.")
        return plan
    except (json.JSONDecodeError, TypeError) as e:
        print(f"ERROR: Failed to parse JSON from extracted block. Error: {e}")
        print(f"Extracted Content: {json_content[:200]}...")
        return [{"step": f"Failed to parse JSON in plan: {e}", "keywords": []}]


def generate_bibliography_map(scraped_sources: list) -> tuple[dict, str]:
    """
    Creates a mapping from URL to citation number and a formatted list for the LLM prompt.
    """
    url_to_index_map = {}
    bibliography_entries = []
    for i, source in enumerate(scraped_sources, 1):
        url = source.get('url')
        if url:
            url_to_index_map[url] = i
            bibliography_entries.append(f"[{i}]: {url}")

    bibliography_prompt_list = "\n".join(bibliography_entries)
    return url_to_index_map, bibliography_prompt_list


def convert_markdown_to_html(markdown_text: str) -> str:
    """Converts Markdown text to HTML using markdown-it-py."""
    if not markdown_text:
        print("Warning: Attempted to convert empty Markdown to HTML.")
        return "<p><em>Report content is empty.</em></p>"
    try:
        # Render the markdown to HTML
        html_output = md.render(markdown_text)
        if not html_output.strip():
             print("Warning: Markdown rendering resulted in empty HTML.")
             escaped_md = html.escape(markdown_text)
             return f"<p><em>Markdown conversion resulted in empty HTML. Raw content:</em></p><pre><code>{escaped_md}</code></pre>"
        return html_output
    # REMOVED specific except MarkdownItError block
    except Exception as e:
        # Catch any unexpected errors during rendering
        print(f"ERROR: Unexpected error converting Markdown to HTML: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
        escaped_md = html.escape(markdown_text)
        return f"<pre><strong>Unexpected error during Markdown conversion: {html.escape(str(e))}</strong>\n\nRaw Markdown:\n{escaped_md}</pre>"

# --- REMOVED convert_html_to_docx ---