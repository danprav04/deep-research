# utils.py
import json
import re  # Import the regular expression module
from html import escape # For escaping error messages or fallback text
from io import BytesIO
import markdown # Added for Markdown conversion


# --- Check for Optional Dependencies ---
DOCX_CONVERSION_AVAILABLE = False
try:
    import html2docx
    DOCX_CONVERSION_AVAILABLE = True
    print("INFO: 'html2docx' library found. DOCX download will be available.")
except ImportError:
    print("WARN: 'html2docx' library not found. DOCX download will be disabled.")
    print("      Install it using: pip install html2docx")


def parse_research_plan(raw_response):
    """
    Parses the JSON research plan potentially embedded in a Markdown code block.
    Handles errors and returns a list of steps.
    """
    json_text = None
    try:
        if not raw_response:
            raise ValueError("Empty response received from LLM.")

        # Use regex to find the JSON block, ignoring potential surrounding text
        # re.DOTALL makes '.' match newlines
        match = re.search(r"```json\s*(.*?)\s*```", raw_response, re.DOTALL)

        if match:
            json_text = match.group(1).strip() # Extract the content within the block and strip whitespace
        else:
            # Fallback: If no code block found, maybe the LLM outputted raw JSON?
            # Try parsing the whole response, but warn if it works.
            # This is less reliable.
            print("WARN: LLM response did not contain the expected ```json ... ``` block. Attempting to parse entire response as JSON.")
            json_text = raw_response.strip()
            # Basic check: Does it look like JSON? (Starts with [ or {)
            if not (json_text.startswith('[') or json_text.startswith('{')):
                 raise ValueError("Could not find JSON block and response doesn't look like JSON.")

        if not json_text:
            raise ValueError("Extracted JSON text is empty.")

        # Attempt to parse the extracted (or fallback) JSON text
        plan_list = json.loads(json_text)

        if not isinstance(plan_list, list):
            raise ValueError("Parsed JSON is not a list as expected for the plan.")

        # Validate each item in the list
        for item in plan_list:
            if not isinstance(item, dict) or 'step' not in item or 'keywords' not in item:
                raise ValueError("Each plan item must be a dict with 'step' and 'keywords'.")
            if not isinstance(item['keywords'], list):
                raise ValueError("'keywords' must be a list in each plan item.")

        return plan_list

    except json.JSONDecodeError as e:
        # Error during the actual JSON parsing
        error_message = f"Error parsing extracted JSON: {e}."
        print(error_message)
        print("--- Extracted Text Attempted (JSON Parsing Error) ---")
        print(json_text if json_text else "N/A (No text extracted)")
        print("--- End Extracted Text ---")
        print("--- Full Raw LLM Response (JSON Parsing Error) ---")
        print(raw_response)
        print("--- End Full Response ---")
        return [{"step": f"Failed to parse extracted JSON: {e}", "keywords": []}]

    except ValueError as e:
        # Error during extraction or validation
        error_message = f"Error processing research plan: {e}."
        print(error_message)
        print("--- Full Raw LLM Response (Processing Error) ---")
        print(raw_response)
        print("--- End Full Response ---")
        return [{"step": f"Failed to process plan response: {e}", "keywords": []}]

    except Exception as e: # Catch broader errors
        print(f"Unexpected error parsing research plan: {e}.")
        print("--- Full Raw LLM Response (Unexpected Error) ---")
        print(raw_response)
        print("--- End Full Response ---")
        return [{"step": f"Unexpected error parsing plan: {e}", "keywords": []}]


def generate_bibliography_map(scraped_sources_list):
    """Creates a mapping from URL to index and a string list for the LLM prompt."""
    url_to_index_map = {}
    bibliography_prompt_list = []
    for index, source in enumerate(scraped_sources_list):
        url = source.get('url')
        if url:
            ref_number = index + 1
            url_to_index_map[url] = ref_number
            bibliography_prompt_list.append(f"[{ref_number}]: {url}")
    return url_to_index_map, "\n".join(bibliography_prompt_list) if bibliography_prompt_list else ""

def convert_markdown_to_html(markdown_text):
    """Converts Markdown text to HTML using the 'footnotes' extension."""
    if not markdown_text or not isinstance(markdown_text, str):
        return "<p><i>No Markdown content provided.</i></p>"
    try:
        # Enable footnotes extension for [^N] and [^N]: syntax
        # Enable other useful extensions like tables, fenced_code
        html = markdown.markdown(
            markdown_text,
            extensions=['footnotes', 'tables', 'fenced_code', 'md_in_html'] # md_in_html can help sometimes
        )
        return html
    except Exception as e:
        print(f"Error converting Markdown to HTML: {e}")
        # Return escaped error message and original text as fallback
        return f"<h2>Markdown Conversion Error</h2><p>{escape(str(e))}</p><h3>Original Markdown:</h3><pre>{escape(markdown_text)}</pre>"

def convert_html_to_docx(html_content):
    """Converts HTML content to a DOCX file in memory."""
    docx_buffer = BytesIO()
    try:
        html2docx.convert(html_content, docx_buffer)
        docx_buffer.seek(0) # Important: Reset buffer position to the beginning
        return docx_buffer
    except Exception as e:
        print(f"Error during HTML to DOCX conversion: {e}")
        raise