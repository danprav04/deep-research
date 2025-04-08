# utils.py
import re
import json
from io import BytesIO
import markdown as md_lib
# html2docx is imported conditionally in app.py to set a flag

def parse_research_plan(llm_response):
    """
    Parses the LLM response to extract a research plan.
    Attempts JSON parsing first, then falls back to regex/line-based parsing.
    Returns a list of dictionaries [{'step_description': str, 'keywords': [str]}] or a failure indicator.
    """
    plan = []
    if not llm_response:
        print("Error: Received empty response from LLM for plan generation.")
        return [{"step_description": "Failed - Empty LLM response", "keywords": []}]

    raw_response = llm_response.strip()

    # 1. Try parsing JSON from Markdown code block or raw string
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_response, re.MULTILINE)
    json_str = None
    if match:
        json_str = match.group(1).strip()
    elif (raw_response.startswith('[') and raw_response.endswith(']')) or \
         (raw_response.startswith('{') and raw_response.endswith('}')):
        json_str = raw_response # Assume raw response is JSON

    if json_str:
        try:
            data = json.loads(json_str)
            # Validate structure: must be a non-empty list of dicts with 'step' and 'keywords'
            if isinstance(data, list) and data and \
               all(isinstance(item, dict) and 'step' in item and 'keywords' in item for item in data):
                temp_plan = []
                for item in data:
                    keywords_list = item.get('keywords', [])
                    # Handle keywords being string or list
                    if isinstance(keywords_list, str):
                        keywords_list = [k.strip() for k in re.split(r'[,\n]', keywords_list) if k.strip()]
                    elif not isinstance(keywords_list, list):
                        keywords_list = [] # Default to empty list if invalid type
                    # Ensure keywords are strings
                    valid_keywords = [str(k).strip() for k in keywords_list if str(k).strip()]
                    step_desc = str(item.get('step', '')).strip()
                    if step_desc: # Ensure step description is not empty
                        temp_plan.append({"step_description": step_desc, "keywords": valid_keywords})
                if temp_plan: # If we successfully extracted at least one valid step
                    return temp_plan
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing failed: {json_err}. Falling back.")
        except Exception as parse_err: # Catch other potential errors during validation
            print(f"Error validating parsed JSON structure: {parse_err}. Falling back.")


    # 2. Fallback: Try Regex parsing for Markdown lists or similar structures
    # Pattern: Optional number/bullet, step description, optional keywords in brackets/parens
    pattern_regex = re.compile(
        r"^\s*(?:\d+\.?\s*[:-]?|-|\*)\s*(.*?)"  # Line start, optional list marker, capture step description
        r"(?:\s*[\(\[]\s*Keywords?\s*[:\-]?\s*(.*?)\s*[\)\]]\s*)?$", # Optional keywords section
        re.MULTILINE | re.IGNORECASE
    )
    matches = pattern_regex.findall(raw_response)
    if matches:
        plan = []
        for desc, keys_str in matches:
            desc = desc.strip()
            # Clean description from potential keyword remnants if regex captured too much
            desc = re.sub(r'\s*\(?Keywords?[:\-]?.*?\)?$', '', desc, flags=re.IGNORECASE).strip()
            keys = []
            if keys_str:
                 keys = [k.strip() for k in re.split(r'[,\n]', keys_str) if k.strip()] # Split by comma or newline
            if desc: # Only add if description is non-empty
                 plan.append({"step_description": desc, "keywords": keys})
        if plan:
            return plan

    # 3. Fallback: Simple Line-based Parsing (Less Robust)
    lines = raw_response.strip().split('\n')
    plan = []
    current_step = None
    keyword_markers = ["Keywords:", "keywords:", "Search Terms:", "search terms:", "Keywords -"]
    for line in lines:
        line = line.strip()
        if not line: continue

        # Match lines starting with "Step X:", numbers, or bullets
        step_match = re.match(r"^\s*(?:step\s+)?(\d+)\s*[:.\-]?\s*(.*)|^\s*[-*+]\s+(.*)", line, re.IGNORECASE)
        if step_match:
            step_desc = (step_match.group(2) or step_match.group(3) or "").strip()
            keys = []
            # Check if keywords are on the same line
            for marker in keyword_markers:
                if marker.lower() in step_desc.lower():
                    parts = re.split(marker, step_desc, maxsplit=1, flags=re.IGNORECASE)
                    if len(parts) == 2:
                        step_desc = parts[0].strip() # Take text before marker as description
                        keys_str = parts[1].strip()
                        keys = [k.strip() for k in re.split(r'[,\n]', keys_str) if k.strip()]
                        break # Found keywords on this line
            if step_desc:
                current_step = {"step_description": step_desc, "keywords": keys}
                plan.append(current_step)
            else:
                current_step = None # Reset if line didn't yield a valid description
        # If line doesn't look like a step start, check if it's keywords for the previous step
        elif current_step and not current_step["keywords"]: # Only if previous step needs keywords
             is_keyword_line = False
             for marker in keyword_markers:
                 if line.lower().startswith(marker.lower()):
                     keys_str = line[len(marker):].strip()
                     current_step["keywords"] = [k.strip() for k in re.split(r'[,\n]', keys_str) if k.strip()]
                     is_keyword_line = True
                     break
             # Avoid adding random lines as step descriptions if they don't match step/keyword formats
             # if not is_keyword_line and current_step:
             #    current_step["step_description"] += "\n" + line # Append to previous description (optional, can be noisy)

    if plan:
        plan = [p for p in plan if p.get("step_description")] # Final filter for valid steps
        if plan:
             return plan

    # 4. Failure Case
    fail_msg = "Failed to parse plan structure from LLM response."
    raw_snippet = raw_response[:200] + '...' if raw_response else '(empty)'
    print(f"{fail_msg} Raw Snippet: '{raw_snippet}'")
    return [{"step_description": fail_msg, "keywords": []}]


def generate_bibliography_map(scraped_sources_list):
    """Creates a map of {url: index} and a numbered list string for prompts."""
    if not scraped_sources_list:
        return {}, ""
    url_to_index = {data['url']: i + 1 for i, data in enumerate(scraped_sources_list)}
    numbered_list_str = "\n".join([f"{i+1}. {data['url']}" for i, data in enumerate(scraped_sources_list)])
    return url_to_index, numbered_list_str

def convert_markdown_to_html(markdown_text):
    """Converts Markdown string to HTML string."""
    try:
        # Common extensions for footnotes, code blocks, tables etc.
        html = md_lib.markdown(
            markdown_text,
            extensions=[
                'footnotes',    # For [^N] style citations
                'fenced_code',  # For ```code``` blocks
                'tables',       # For Markdown tables
                'nl2br',        # Convert newlines to <br>
                'attr_list',    # Add attributes to elements like headers
                'md_in_html'    # Allow Markdown inside HTML blocks
            ]
        )
        return html
    except Exception as e:
        print(f"Error converting Markdown to HTML: {e}")
        # Fallback: return escaped Markdown in a <pre> tag
        from html import escape
        return f"<p><strong>Markdown Conversion Error:</strong> {escape(str(e))}</p><pre>{escape(markdown_text)}</pre>"


def convert_html_to_docx(html_content, base_url=None):
    """
    Converts an HTML string to a DOCX file in memory (BytesIO buffer).
    Requires the 'html2docx' library.
    """
    # Import here to avoid making it a hard dependency if DOCX isn't used/available
    try:
        from html2docx import Html2Docx
    except ImportError:
        print("Error: html2docx library is required for DOCX conversion but not found.")
        raise # Re-raise the import error to be caught by the caller

    parser = Html2Docx(base_url=base_url) # base_url might help resolve relative paths if needed
    buffer = BytesIO()
    parser.parse_html_string(html_content)
    parser.docx.save(buffer) # Saves the internal python-docx object to buffer
    buffer.seek(0) # Reset buffer position to the beginning for reading
    return buffer