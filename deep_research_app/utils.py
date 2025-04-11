# utils.py
import re
import json
import html
import logging
from typing import List, Dict, Tuple, Any

from markdown_it import MarkdownIt
from mdit_py_plugins.footnote import footnote_plugin
# from mdit_py_plugins.front_matter import front_matter_plugin # Example: if you need front matter
# from mdit_py_plugins.tasklists import tasklists_plugin # Example: if you need task lists

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Constants ---
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.MULTILINE)
# Pattern to find the raw URL citations inserted by the synthesis step
CITATION_URL_PATTERN = re.compile(r"\[Source URL:\s*(https?://[^\s\]]+)\s*\]")

# --- Configure Markdown-it Instance ---
# Initialize with 'commonmark' preset for base compliance.
# Enable useful features:
# - html=True: Allows raw HTML in Markdown (use with caution if input is untrusted)
# - linkify=True: Auto-detects URLs and makes them links
# - typographer=True: Enables smart quotes, dashes, etc.
# Enable specific GFM (GitHub Flavored Markdown) features and plugins:
md = (
    MarkdownIt("commonmark", {
        "html": True,        # Allow HTML tags
        "linkify": True,     # Autoconvert URL-like text to links
        "typographer": True, # Enable smart quotes, dashes, etc.
        "breaks": False,     # Keep True if you want single newlines converted to <br> (GFM style)
    })
    .enable("table")          # Enable GFM tables
    .enable("strikethrough")  # Enable GFM strikethrough (~~text~~)
    # .enable("tasklist") # If using tasklists_plugin: Enable task lists [-] [x]
    .use(footnote_plugin)     # Enable footnotes ([^1], [^1]: text)
    # .use(front_matter_plugin) # Example: Enable YAML front matter parsing
    # .use(tasklists_plugin, enabled=True) # Example: Enable task list rendering
)
logger.info("Markdown-it parser configured with commonmark base, table, strikethrough, and footnote plugins.")


# --- Functions ---

def parse_research_plan(llm_response: str) -> List[Dict[str, Any]]:
    """
    Parses the LLM response to extract the JSON research plan.
    Handles cases with or without the ```json block and validates structure.
    """
    if not llm_response:
        logger.error("Attempted to parse an empty LLM response for research plan.")
        return [{"step": "Failed to parse research plan: LLM response was empty.", "keywords": []}]

    match = JSON_BLOCK_PATTERN.search(llm_response)
    json_content = ""
    parse_method = ""

    if match:
        json_content = match.group(1).strip()
        parse_method = "json block"
    else:
        # Fallback: Try parsing the whole response if it looks like a JSON list/object
        stripped_response = llm_response.strip()
        if stripped_response.startswith('[') and stripped_response.endswith(']'):
            json_content = stripped_response
            parse_method = "direct list"
        elif stripped_response.startswith('{') and stripped_response.endswith('}'):
             # Handle case where LLM might wrap the list in an object (e.g., {"plan": [...]})
             try:
                 potential_obj = json.loads(stripped_response)
                 # Look for a key that holds a list (common variations)
                 for key in potential_obj:
                     if isinstance(potential_obj[key], list):
                         json_content = json.dumps(potential_obj[key]) # Re-serialize the list part
                         parse_method = f"nested list in key '{key}'"
                         break
                 if not json_content:
                     raise ValueError("JSON object found, but no list value detected within it.")
             except (json.JSONDecodeError, ValueError) as e:
                 logger.warning(f"Failed fallback attempt to parse response as JSON object containing list: {e}")
                 # Continue to error reporting below

        if not json_content:
            logger.error(f"Failed to parse research plan. No ```json block found and response doesn't appear to be a JSON list or recognized object. Response snippet: {llm_response[:200]}...")
            return [{"step": "Failed to parse research plan: Could not find or parse JSON structure.", "keywords": []}]

    # --- Try parsing the extracted/identified JSON content ---
    try:
        logger.info(f"Attempting to parse research plan using method: {parse_method}")
        plan = json.loads(json_content)

        # --- Validation ---
        if not isinstance(plan, list):
            raise TypeError("Parsed JSON is not a list.")

        if not plan:
             raise ValueError("Parsed plan is an empty list.")

        validated_plan = []
        for i, item in enumerate(plan):
            if not isinstance(item, dict):
                raise TypeError(f"Item at index {i} is not a dictionary: {item}")
            if 'step' not in item or not isinstance(item['step'], str) or not item['step'].strip():
                raise ValueError(f"Item at index {i} missing 'step' key or step is empty: {item}")
            if 'keywords' not in item:
                 logger.warning(f"Item at index {i} missing 'keywords' key. Defaulting to empty list. Item: {item}")
                 item['keywords'] = [] # Add default empty list
            elif not isinstance(item['keywords'], list):
                 # Attempt to fix if keywords are a single string instead of a list
                 if isinstance(item['keywords'], str):
                      logger.warning(f"Fixing 'keywords' field for step '{item['step']}' - was string, expected list. Splitting by comma.")
                      item['keywords'] = [kw.strip() for kw in item['keywords'].split(',') if kw.strip()]
                 else:
                      raise TypeError(f"Keywords for step '{item['step']}' is not a list or string: {type(item['keywords'])}")

            # Ensure keywords are strings
            item['keywords'] = [str(kw) for kw in item['keywords'] if isinstance(kw, (str, int, float)) and str(kw).strip()]
            validated_plan.append(item)

        logger.info(f"Successfully parsed and validated research plan with {len(validated_plan)} steps.")
        return validated_plan

    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.error(f"Failed to parse or validate JSON for research plan. Method: {parse_method}. Error: {e}", exc_info=True)
        logger.error(f"Problematic JSON Content Snippet: {json_content[:500]}...")
        return [{"step": f"Failed to parse/validate JSON in plan: {e}", "keywords": []}]
    except Exception as e:
         logger.error(f"Unexpected error parsing research plan: {e}", exc_info=True)
         return [{"step": f"Unexpected error parsing plan: {e}", "keywords": []}]


def generate_bibliography_map(scraped_sources: List[Dict[str, Any]]) -> Tuple[Dict[str, int], str]:
    """
    Creates a mapping from URL to citation number [1...N] and a formatted
    string list suitable for inclusion in an LLM prompt.

    Args:
        scraped_sources: A list of dictionaries, where each dict represents a
                         successfully scraped source and contains at least an 'url' key.

    Returns:
        A tuple containing:
        - url_to_index_map: Dictionary mapping {url: citation_index}.
        - bibliography_prompt_list: A newline-separated string of "[index]: url".
    """
    url_to_index_map: Dict[str, int] = {}
    bibliography_entries: List[str] = []
    citation_index = 1
    processed_urls = set() # Handle potential duplicate URLs in input list

    for source in scraped_sources:
        url = source.get('url')
        if url and isinstance(url, str) and url.strip() and url not in processed_urls:
            url_to_index_map[url] = citation_index
            bibliography_entries.append(f"[{citation_index}]: {url}")
            processed_urls.add(url)
            citation_index += 1
        elif url in processed_urls:
            logger.warning(f"Duplicate URL found in scraped sources list, skipping for bibliography: {url}")
        # else: logger.debug(f"Skipping invalid or missing URL in source for bibliography: {source}")


    bibliography_prompt_list = "\n".join(bibliography_entries)
    logger.info(f"Generated bibliography map with {len(url_to_index_map)} unique entries.")
    return url_to_index_map, bibliography_prompt_list


def convert_markdown_to_html(markdown_text: str) -> str:
    """
    Converts Markdown text to HTML using the configured markdown-it-py instance.
    Includes basic error handling and fallbacks.
    """
    if not markdown_text or not markdown_text.strip():
        logger.warning("Attempted to convert empty or whitespace-only Markdown to HTML.")
        return "<p><em>Report content is empty or contains only whitespace.</em></p>"
    try:
        # Render the markdown to HTML using the pre-configured 'md' instance
        html_output = md.render(markdown_text)

        if not html_output or not html_output.strip():
             logger.warning("Markdown rendering resulted in empty or whitespace-only HTML.")
             # Provide context in the fallback message
             escaped_md_snippet = html.escape(markdown_text[:200]) + ('...' if len(markdown_text) > 200 else '')
             return f"<p><em>Markdown conversion resulted in empty HTML. Raw content snippet:</em></p><pre><code>{escaped_md_snippet}</code></pre>"

        # logger.debug(f"Successfully converted Markdown to HTML (length: {len(html_output)} chars).")
        return html_output

    except Exception as e:
        # Catch any unexpected errors during rendering
        logger.error(f"Unexpected error converting Markdown to HTML: {e}", exc_info=True)
        # Provide a more informative error message in the HTML output
        escaped_md_snippet = html.escape(markdown_text[:500]) + ('...' if len(markdown_text) > 500 else '')
        error_details = html.escape(str(e))
        return (
            f"<pre><strong>Unexpected error during Markdown conversion:</strong>\n"
            f"{error_details}\n\n"
            f"<strong>Raw Markdown Snippet:</strong>\n"
            f"{escaped_md_snippet}</pre>"
        )

# --- Placeholder for future citation replacement if needed outside LLM ---
# def replace_citations_with_footnotes(text: str, url_map: Dict[str, int]) -> str:
#     """Replaces [Source URL: <url>] with footnote markers [^N]."""
#     def replace_match(match):
#         url = match.group(1)
#         index = url_map.get(url)
#         if index is not None:
#             return f"[^{index}]"
#         else:
#             logger.warning(f"URL found in text but not in bibliography map: {url}")
#             return "" # Remove the tag if URL not found
#     return CITATION_URL_PATTERN.sub(replace_match, text)