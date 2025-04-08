# utils.py
import re
import json
from io import BytesIO
from markdown_it import MarkdownIt
from mdit_py_plugins.footnote import footnote_plugin # Ensure footnote plugin is used

# --- Check for Optional Dependencies ---
DOCX_CONVERSION_AVAILABLE = False
try:
    import html2docx
    DOCX_CONVERSION_AVAILABLE = True
except ImportError:
    pass # Handled in app.py, just avoid error here

# --- Constants ---
# Matches ```json ... ``` blocks, handling potential variations
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.MULTILINE)
# Matches [Source URL: <url>] for replacement later
CITATION_URL_PATTERN = re.compile(r"\[Source URL:\s*(https?://[^\s\]]+)\s*\]")

# Configure Markdown-it instance
# Enable HTML tags, links, typographer, and footnotes
md = MarkdownIt("commonmark", {"html": True, "linkify": True, "typographer": True}).enable("table").use(footnote_plugin)

# --- Functions ---

def parse_research_plan(llm_response: str) -> list:
    """
    Parses the LLM response to extract the JSON research plan.

    Args:
        llm_response: The raw string response from the LLM.

    Returns:
        A list of dictionaries representing the research plan steps,
        or a list with a single error dictionary if parsing fails.
        Format: [{"step": "...", "keywords": ["...", "..."]}, ...]
    """
    match = JSON_BLOCK_PATTERN.search(llm_response)
    if not match:
        # Attempt to find JSON even without backticks as a fallback
        try:
            # Be cautious with loading arbitrary JSON directly from LLM
            # This assumes the LLM output *is* the JSON list if no block is found
            potential_json = llm_response.strip()
            if potential_json.startswith('[') and potential_json.endswith(']'):
                 plan = json.loads(potential_json)
                 # Basic validation
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
        # Validate structure
        if not isinstance(plan, list):
            raise TypeError("Parsed JSON is not a list.")
        for item in plan:
            if not isinstance(item, dict) or 'step' not in item or 'keywords' not in item:
                raise TypeError("Parsed list items do not have 'step' and 'keywords' keys.")
            if not isinstance(item['keywords'], list):
                 # Attempt to fix if keywords is a string instead of list (common LLM error)
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

    Args:
        scraped_sources: A list of dictionaries, where each dict has at least a 'url' key.
                         Example: [{'url': 'http://...', 'content': '...'}, ...]

    Returns:
        A tuple containing:
        - url_to_index_map (dict): {'http://...': 1, 'http://...': 2, ...}
        - bibliography_prompt_list (str): A formatted string like:
          "[1]: http://...\n[2]: http://..."
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
    """Converts Markdown text to HTML using markdown-it-py with footnotes."""
    if not markdown_text:
        return ""
    try:
        # The 'md' instance is already configured with footnote_plugin
        html = md.render(markdown_text)
        return html
    except Exception as e:
        print(f"Error converting Markdown to HTML: {e}")
        # Return raw text wrapped in pre tags as fallback
        return f"<pre>Error during Markdown conversion: {e}\n\n{markdown_text}</pre>"


def convert_html_to_docx(html_content: str) -> BytesIO | None:
    """
    Converts HTML content string to a DOCX file in memory.

    Args:
        html_content: The HTML string to convert.

    Returns:
        A BytesIO buffer containing the DOCX file data, or None if conversion fails.
        Returns None if html2docx is not installed.
    """
    if not DOCX_CONVERSION_AVAILABLE:
        print("Error: html2docx library not available for DOCX conversion.")
        return None

    if not html_content:
        print("Warning: Attempted to convert empty HTML to DOCX.")
        # Return an empty buffer or handle as needed, maybe create a minimal docx?
        # For now, returning None as it's likely an upstream issue if content is empty.
        return None

    buffer = BytesIO()
    try:
        # html2docx expects a file-like object or path for parsing, we use BytesIO
        # Need to encode the HTML string to bytes first
        html_bytes = html_content.encode('utf-8')
        html_buffer = BytesIO(html_bytes)

        # Convert using html2docx
        parser = html2docx.HTML2Docx(buffer, "BUFFER") # Use "BUFFER" to work with BytesIO
        parser.parse_html_stream(html_buffer) # Use parse_html_stream for buffers

        buffer.seek(0) # Reset buffer position to the beginning for reading
        return buffer
    except Exception as e:
        print(f"Error converting HTML to DOCX: {e}")
        import traceback
        traceback.print_exc()
        return None # Indicate failure