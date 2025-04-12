# utils.py
import re
import json
import html
import logging
from typing import List, Dict, Tuple, Any, Optional

from markdown_it import MarkdownIt
from markdown_it.utils import OptionsDict
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.attrs import attrs_plugin # Allows adding attributes like classes/IDs
from mdit_py_plugins.deflist import deflist_plugin # Definition lists
from mdit_py_plugins.tasklists import tasklists_plugin # Task lists
import bleach # For sanitizing scraped content *before* LLM

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Constants ---
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.MULTILINE)
# Pattern to find the raw URL citations inserted by the synthesis step (kept for reference, but replacement is done by LLM now)
# CITATION_URL_PATTERN = re.compile(r"\[Source URL:\s*(https?://[^\s\]]+)\s*\]")

# --- Bleach Configuration (for Sanitizing Scraped Content BEFORE LLM) ---
# Define allowed tags - focus on text structure and semantics, remove interactive/scripting/styling
ALLOWED_TAGS_FOR_LLM = [
    'p', 'br', 'b', 'strong', 'i', 'em', 'u', 'strike', 'del', 's',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li',
    'blockquote', 'code', 'pre',
    'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td',
    'a', # Keep links, href will be checked/cleaned separately if needed
    'dl', 'dt', 'dd', # Definition lists
    # Maybe allow 'figure', 'figcaption'? For now, keep it minimal.
]
# Allow only basic attributes, mainly 'href' for links
ALLOWED_ATTRS_FOR_LLM = {
    'a': ['href'],
    'th': ['colspan', 'rowspan'],
    'td': ['colspan', 'rowspan'],
}

# --- Configure Markdown-it Instance (for Final Report Rendering) ---
# Initialize with 'gfm-like' preset for common features.
md_options: OptionsDict = {
    "html": False,       # Disable raw HTML tags in final report Markdown for security
    "linkify": True,     # Autoconvert URL-like text to links
    "typographer": True, # Enable smart quotes, dashes, etc.
    "breaks": True,      # Convert single newlines in paragraphs to <br> (GFM style)
}
md = (
    MarkdownIt("gfm-like", md_options)
    .enable("table")          # Ensure GFM tables are enabled
    .enable("strikethrough")  # Ensure GFM strikethrough is enabled
    .use(footnote_plugin)     # Enable footnotes ([^1], [^1]: text)
    .use(attrs_plugin)        # Enable adding attributes like {.class #id}
    .use(deflist_plugin)      # Enable definition lists
    .use(tasklists_plugin, enabled=True) # Enable task lists [-] [x]
    # Disable rules that could be risky if bad markdown is generated
    .disable('html_inline')
    .disable('html_block')
)
logger.info("Markdown-it parser configured for final report rendering (gfm-like, html disabled).")


# --- Functions ---

def sanitize_scraped_content(html_content: str, url: str = "Unknown URL") -> str:
    """
    Sanitizes scraped HTML content to remove potentially harmful elements
    before sending it to an LLM. Uses Bleach library.

    Args:
        html_content: The raw HTML string scraped from a webpage.
        url: The URL from which the content was scraped (for logging).

    Returns:
        A sanitized string containing only allowed HTML tags and attributes,
        or plain text if HTML parsing/cleaning fails significantly.
    """
    if not html_content or not html_content.strip():
        return ""

    try:
        # Use Bleach to clean the HTML according to allowed lists
        cleaned_html = bleach.clean(
            html_content,
            tags=ALLOWED_TAGS_FOR_LLM,
            attributes=ALLOWED_ATTRS_FOR_LLM,
            strip=True,  # Remove disallowed tags entirely
            strip_comments=True
        )

        # Optional: Further simplify? e.g., convert headings to bold text?
        # For now, keep basic structure.

        # Get text content as a fallback or primary method if HTML structure isn't crucial
        # soup = BeautifulSoup(cleaned_html, 'html.parser')
        # plain_text = soup.get_text(separator='\n', strip=True)
        # return plain_text # If plain text is preferred

        logger.debug(f"Sanitized content for {url[:70]}... Original length: {len(html_content)}, Cleaned length: {len(cleaned_html)}")
        return cleaned_html.strip()

    except Exception as e:
        logger.error(f"Error sanitizing HTML content for {url[:70]}... with Bleach: {e}", exc_info=False)
        # Fallback: Try to extract plain text aggressively if Bleach fails
        try:
            from bs4 import BeautifulSoup # Local import for fallback only
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove script/style first
            for element in soup(["script", "style", "nav", "footer", "aside", "header", "form"]):
                element.decompose()
            plain_text = soup.get_text(separator='\n', strip=True)
            logger.warning(f"Sanitization failed for {url[:70]}, falling back to basic text extraction.")
            # Very basic cleaning of excessive newlines
            return re.sub(r'\n{3,}', '\n\n', plain_text).strip()
        except Exception as fallback_e:
            logger.error(f"Fallback text extraction also failed for {url[:70]}: {fallback_e}", exc_info=False)
            # Last resort: return an empty string or a placeholder error
            return "[Content Sanitization Failed]"


def parse_research_plan(llm_response: str) -> List[Dict[str, Any]]:
    """
    Parses the LLM response to extract the JSON research plan.
    Handles cases with or without the ```json block and validates structure.
    Returns an empty list or list indicating failure on error.
    """
    if not llm_response or not llm_response.strip():
        logger.error("Attempted to parse an empty LLM response for research plan.")
        return [{"step": "Failed to parse research plan: LLM response was empty.", "keywords": []}] # Indicate failure

    match = JSON_BLOCK_PATTERN.search(llm_response)
    json_content = ""
    parse_method = ""

    if match:
        json_content = match.group(1).strip()
        # Handle potential escaped characters if LLM escapes them within the block
        try:
            # Basic unescaping - might need more robust handling if complex escapes are used
            json_content = bytes(json_content, "utf-8").decode("unicode_escape")
        except Exception as decode_err:
             logger.warning(f"Could not decode unicode escapes in JSON block, proceeding anyway: {decode_err}")
        parse_method = "json block"
    else:
        # Fallback: Try parsing the whole response if it looks like a JSON list/object
        stripped_response = llm_response.strip()
        if stripped_response.startswith('[') and stripped_response.endswith(']'):
            json_content = stripped_response
            parse_method = "direct list"
        elif stripped_response.startswith('{') and stripped_response.endswith('}'):
             try:
                 potential_obj = json.loads(stripped_response)
                 # Look for common keys that might hold the plan list
                 for key in ['plan', 'research_plan', 'steps', 'keywords', 'result']:
                     if isinstance(potential_obj.get(key), list):
                         json_content = json.dumps(potential_obj[key]) # Re-serialize just the list
                         parse_method = f"nested list in key '{key}'"
                         break
                 if not json_content:
                     # If no common key found, check if *any* value is a list
                    for key, value in potential_obj.items():
                        if isinstance(value, list):
                            json_content = json.dumps(value)
                            parse_method = f"nested list in key '{key}' (generic)"
                            break
                 if not json_content:
                    raise ValueError("JSON object found, but no list value detected within it using common keys.")
             except (json.JSONDecodeError, ValueError) as e:
                 logger.warning(f"Failed fallback attempt to parse response as JSON object containing list: {e}")
                 # Continue to error reporting below
        if not json_content:
            # Log the failure but return specific failure indicator list
            logger.error(f"Failed to parse research plan. No ```json block found and response doesn't appear to be a valid JSON list or recognized object. Response snippet: {llm_response[:200]}...")
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
        required_keys = {"step", "keywords"}
        for i, item in enumerate(plan):
            if not isinstance(item, dict):
                raise TypeError(f"Plan item at index {i} is not a dictionary: {item}")

            if not required_keys.issubset(item.keys()):
                 missing = required_keys - item.keys()
                 raise ValueError(f"Plan item at index {i} is missing required keys: {missing}. Item: {item}")

            step = item.get('step')
            keywords = item.get('keywords')

            if not step or not isinstance(step, str) or not step.strip():
                raise ValueError(f"Item at index {i} has invalid or empty 'step': {item}")
            if not isinstance(keywords, list):
                 # Attempt to fix if keywords are a single string instead of a list
                 if isinstance(keywords, str):
                      logger.warning(f"Fixing 'keywords' field for step '{step}' - was string, expected list. Splitting by comma.")
                      keywords = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                 else:
                      raise TypeError(f"Keywords for step '{step}' is not a list or string: {type(keywords)}")

            # Ensure keywords are non-empty strings
            valid_keywords = [str(kw).strip() for kw in keywords if kw and isinstance(kw, (str, int, float)) and str(kw).strip()]
            item['keywords'] = valid_keywords # Update item with cleaned keywords
            validated_plan.append(item)

        logger.info(f"Successfully parsed and validated research plan with {len(validated_plan)} steps.")
        return validated_plan

    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.error(f"Failed to parse or validate JSON for research plan. Method: {parse_method}. Error: {e}", exc_info=False) # Keep log less verbose
        logger.debug(f"Problematic JSON Content Snippet for plan parsing: {json_content[:500]}...")
        return [{"step": f"Failed to parse/validate JSON in plan: {escape(str(e))}", "keywords": []}] # Return failure indicator
    except Exception as e:
         logger.error(f"Unexpected error parsing research plan: {e}", exc_info=True)
         return [{"step": f"Unexpected error parsing plan: {escape(str(e))}", "keywords": []}] # Return failure indicator


def generate_bibliography_map(scraped_sources: List[Dict[str, Any]]) -> Tuple[Dict[str, int], str]:
    """
    Creates a mapping from URL to citation number [1...N] and a formatted
    string list suitable for inclusion in an LLM prompt.

    Args:
        scraped_sources: A list of dictionaries from successfully scraped sources,
                         each containing at least a valid 'url' key.

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
        # Validate source structure and URL presence/type
        if isinstance(source, dict) and 'url' in source:
            url = source.get('url')
            if url and isinstance(url, str) and url.strip().startswith(('http://', 'https://')):
                # Normalize URL slightly? (e.g., remove fragment, trailing slash?) - Be careful not to break matching.
                # url_normalized = url.split('#')[0].rstrip('/')
                url_normalized = url # Keep original URL for now to ensure exact match with synthesis output

                if url_normalized not in processed_urls:
                    url_to_index_map[url_normalized] = citation_index
                    bibliography_entries.append(f"[{citation_index}]: {url_normalized}")
                    processed_urls.add(url_normalized)
                    citation_index += 1
                # else: logger.debug(f"Duplicate URL '{url_normalized}' skipped for bibliography.")
            else:
                logger.warning(f"Invalid or non-HTTP(S) URL found in source dict, skipping for bibliography: {url}")
        else:
            logger.warning(f"Invalid source item format encountered, skipping for bibliography: {source}")


    bibliography_prompt_list = "\n".join(bibliography_entries)
    if not bibliography_entries:
        logger.warning("Generated an empty bibliography map. No valid scraped sources provided.")
    else:
        logger.info(f"Generated bibliography map with {len(url_to_index_map)} unique entries.")

    return url_to_index_map, bibliography_prompt_list


def convert_markdown_to_html(markdown_text: str) -> str:
    """
    Converts Markdown text to HTML using the configured markdown-it-py instance.
    Includes basic error handling and fallbacks. Ensures HTML is disabled.
    """
    if not markdown_text or not markdown_text.strip():
        logger.warning("Attempted to convert empty or whitespace-only Markdown to HTML.")
        # Return semantic HTML indicating emptiness
        return "<article><p><em>Report content is empty or contains only whitespace.</em></p></article>"
    try:
        # Render the markdown to HTML using the pre-configured 'md' instance
        html_output = md.render(markdown_text)

        if not html_output or not html_output.strip():
             logger.warning("Markdown rendering resulted in empty or whitespace-only HTML.")
             escaped_md_snippet = html.escape(markdown_text[:200]) + ('...' if len(markdown_text) > 200 else '')
             return f"<article><p><em>Markdown conversion resulted in empty HTML. Raw content snippet:</em></p><pre><code>{escaped_md_snippet}</code></pre></article>"

        # Wrap the output in <article> for semantic structure
        return f"<article>{html_output}</article>"

    except Exception as e:
        # Catch any unexpected errors during rendering
        logger.error(f"Unexpected error converting Markdown to HTML: {e}", exc_info=True)
        escaped_md_snippet = html.escape(markdown_text[:500]) + ('...' if len(markdown_text) > 500 else '')
        error_details = html.escape(str(e))
        # Return semantic error HTML
        return (
            f"<article>"
            f"<h2>Markdown Conversion Error</h2>"
            f"<p><strong>An unexpected error occurred while rendering the report content:</strong></p>"
            f"<pre><code>{error_details}</code></pre>"
            f"<p><strong>Raw Markdown Snippet:</strong></p>"
            f"<pre><code>{escaped_md_snippet}</code></pre>"
            f"</article>"
        )

# --- Citation replacement function (kept for reference, but LLM does it now) ---
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