#!python
import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, stream_with_context, send_file
# --- NEW: Import Google Generative AI ---
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # For safety settings if needed later

from dotenv import load_dotenv
from duckduckgo_search import DDGS
# Use the Markdown library for better footnote support
import markdown as md_lib
from urllib.parse import quote, unquote
import concurrent.futures
from io import BytesIO

# --- CHANGED: Import pypandoc and add a more robust check ---
PANDOC_AVAILABLE = False
try:
    # Recommend installing pypandoc-binary for bundled executable
    import pypandoc
    # Try getting the path at startup. This is a more reliable check.
    pandoc_path = pypandoc.get_pandoc_path()
    print(f"INFO: Found pandoc executable at: {pandoc_path}")
    PANDOC_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"WARNING: pypandoc or the pandoc executable not found. DOCX download will be disabled. Error: {e}")
    print("RECOMMENDATION: Install 'pypandoc-binary' (pip install pypandoc-binary) which includes the executable.")
    PANDOC_AVAILABLE = False


# --- Configuration ---
load_dotenv()
# --- CHANGED: Use Google API Key and Model Name ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Use a valid Google model name (e.g., gemini-1.5-pro-latest, gemini-1.5-flash-latest)
# Corrected default model name
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash-latest") # Changed default

MAX_SEARCH_RESULTS_PER_STEP = 10
MAX_TOTAL_URLS_TO_SCRAPE = 100
MAX_WORKERS = 10
REQUEST_TIMEOUT = 12
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
PICO_CSS_CDN = "https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css"
# --- NEW: Delay between DDGS searches to avoid rate limiting ---
DDGS_SEARCH_DELAY_SECONDS = 1.5

# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Initialize Google Generative AI Client ---
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Optionally check available models
    # models = [m.name for m in genai.list_models()]
    # print("Available Google Models:", models)
except Exception as e:
    raise RuntimeError(f"Failed to configure Google Generative AI: {e}")

# --- Helper Functions ---

# --- REFACTORED: call_gemini to use Google API ---
def call_gemini(prompt, system_prompt=None, max_retries=3, delay=5):
    """Calls the specified Google Gemini model with retry logic."""
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    generation_config = genai.types.GenerationConfig(
        temperature=0.6,
        # max_output_tokens=8192 # Set if needed, depends on model
    )
    # Basic safety settings (adjust as needed)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(
                GOOGLE_MODEL_NAME,
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=system_prompt if system_prompt else None
            )
            effective_prompt = prompt if system_prompt else full_prompt

            # print(f"--- Calling Google Model ({GOOGLE_MODEL_NAME}) ---") # Debug
            response = model.generate_content(effective_prompt)
            # print(f"--- Raw Google Response --- \n{response}\n--- End Raw Response ---") # Debug

            # More robust check for blocking/empty response
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = str(response.prompt_feedback.block_reason)
                 safety_ratings_str = str(getattr(response.prompt_feedback, 'safety_ratings', 'N/A'))
                 raise ValueError(f"API response blocked by safety settings. Block Reason: {block_reason}. Safety Ratings: {safety_ratings_str}")

            if not response.candidates:
                 # This case might occur if severely blocked or other issues
                 finish_details = getattr(response, 'finish_details', 'N/A') # Check if other finish details exist
                 raise ValueError(f"API response contained no candidates. Finish Details: {finish_details}")

            # Check the first candidate specifically
            candidate = response.candidates[0]
            if candidate.finish_reason not in [None, 1, "STOP", "MAX_TOKENS"]: # 1 == Stop for some versions
                # Other reasons: SAFETY, RECITATION, OTHER
                 raise ValueError(f"API response candidate finished unexpectedly. Finish Reason: {candidate.finish_reason}. Safety Ratings: {getattr(candidate, 'safety_ratings', 'N/A')}")


            # Access the text content safely
            if not candidate.content or not candidate.content.parts:
                 # This can happen even if not explicitly blocked, e.g., empty generation
                 raise ValueError(f"API response candidate has no content parts. Finish Reason: {candidate.finish_reason}")

            response_content = response.text.strip() # response.text conveniently joins parts
            if not response_content:
                 raise ValueError("API response generated empty text content after successful call.")

            return response_content

        except Exception as e:
            print(f"Error calling Google Gemini API (Attempt {attempt + 1}/{max_retries}): {e}")
            if "API key not valid" in str(e):
                 print("Critical Error: Invalid Google API Key.")
                 raise # Don't retry on invalid key
            if "quota" in str(e).lower():
                 print("Warning: Quota possibly exceeded.")
                 # Optionally increase delay or stop retrying for quota
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Failing LLM call.")
                raise

# --- REFACTORED: stream_gemini to use Google API ---
def stream_gemini(prompt, system_prompt=None):
    """
    Calls the Google Gemini model with streaming enabled and yields content chunks.
    """
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    generation_config = genai.types.GenerationConfig(
        temperature=0.6,
        # max_output_tokens=8192 # Set if needed
    )
    safety_settings = { # Same safety settings as non-streaming
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    try:
        model = genai.GenerativeModel(
            GOOGLE_MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=system_prompt if system_prompt else None
        )
        effective_prompt = prompt if system_prompt else full_prompt

        stream = model.generate_content(effective_prompt, stream=True)
        # print(f"--- Streaming Google Model ({GOOGLE_MODEL_NAME}) ---") # Debug

        complete_response_text = ""
        stream_blocked = False

        for chunk in stream:
             # Check for blocking reasons within the chunk's prompt feedback first
             if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                  block_reason_str = str(chunk.prompt_feedback.block_reason)
                  safety_ratings_str = str(getattr(chunk.prompt_feedback, 'safety_ratings', 'N/A'))
                  print(f"ERROR: LLM stream blocked by prompt safety filters. Reason: {block_reason_str}. Ratings: {safety_ratings_str}")
                  yield {'type': 'stream_error', 'message': f'LLM prompt blocked by safety filters (Reason: {block_reason_str}). Cannot generate response.'}
                  stream_blocked = True
                  break # Stop processing if the prompt itself was blocked

             # Check for blocking during generation (in candidate)
             if not stream_blocked and chunk.candidates:
                 candidate = chunk.candidates[0]
                 if candidate.finish_reason and candidate.finish_reason not in [None, 1, "STOP", "MAX_TOKENS"]:
                     finish_reason_str = str(candidate.finish_reason)
                     safety_ratings_str = str(getattr(candidate, 'safety_ratings', 'N/A'))
                     print(f"WARNING: LLM stream stopped during generation. Reason: {finish_reason_str}. Safety: {safety_ratings_str}")
                     yield {'type': 'stream_warning', 'message': f'LLM stream may have been interrupted or blocked during generation (Reason: {finish_reason_str}). Output may be incomplete.'}
                     # Decide whether to break based on reason (e.g., break for SAFETY)
                     if finish_reason_str == "SAFETY":
                         stream_blocked = True # Treat as blocked
                         break

             # Yield text content if available and not blocked
             if not stream_blocked and hasattr(chunk, 'text') and chunk.text:
                  chunk_text = chunk.text
                  complete_response_text += chunk_text
                  yield {'type': 'chunk', 'content': chunk_text}
             # else: # Debugging for empty chunks
                  # if not stream_blocked: print(f"Debug: Received chunk with no text: {chunk}")

        # Send end event only if the stream wasn't explicitly blocked early
        if not stream_blocked:
             yield {'type': 'stream_end', 'finish_reason': 'IterationComplete'} # Simple end signal

    except Exception as e:
        print(f"Error during Google Gemini stream: {e}")
        error_message = f"LLM stream error: {e}"
        if "API key not valid" in str(e):
            error_message = f"LLM stream error: Invalid Google API Key. ({e})"
        elif "quota" in str(e).lower():
            error_message = f"LLM stream error: Quota likely exceeded. ({e})"
        elif "resource has been exhausted" in str(e).lower(): # Another quota message
             error_message = f"LLM stream error: Quota likely exceeded (Resource Exhausted). ({e})"
        elif "prompt" in str(e).lower() and ("too long" in str(e).lower() or "size" in str(e).lower()):
             error_message = f"LLM stream error: Prompt likely too long for the model's context window. ({e})"

        yield {'type': 'stream_error', 'message': error_message}


# parse_research_plan - No changes needed from previous correct version
def parse_research_plan(llm_response):
    # ... (keep the robust parser from previous step) ...
    # (Code is identical to the previous version, omitted for brevity)
    plan = []
    if not llm_response:
        print("Error: Received empty response from LLM for plan generation.")
        return [{"step_description": "Failed - Empty LLM response", "keywords": []}]

    raw_response = llm_response.strip()

    # 1. Try parsing JSON within ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_response, re.MULTILINE)
    json_str = None
    if match:
        json_str = match.group(1).strip()
        print("Attempting to parse JSON found within markdown code block.")
    else:
        # 2. Try parsing the entire response as JSON if it looks like it
        if (raw_response.startswith('[') and raw_response.endswith(']')) or \
           (raw_response.startswith('{') and raw_response.endswith('}')):
            json_str = raw_response
            print("No markdown code block found. Attempting to parse entire response as JSON.")
        else:
           print("Response doesn't appear to be JSON or within a code block. Proceeding to text parsing.")

    if json_str:
        try:
            data = json.loads(json_str)
            if isinstance(data, list) and data:
                # Check structure more carefully
                if all(isinstance(item, dict) and 'step' in item and 'keywords' in item for item in data):
                    temp_plan = []
                    for item in data:
                        keywords_list = item.get('keywords', [])
                        # Handle keywords being a single string or list
                        if isinstance(keywords_list, str):
                            keywords_list = [k.strip() for k in re.split(r'[,\n]', keywords_list) if k.strip()]
                        elif not isinstance(keywords_list, list):
                            print(f"Warning: Keywords for step '{item.get('step')}' was not a list or string, setting to empty.")
                            keywords_list = []
                        # Filter out empty strings again
                        valid_keywords = [str(k).strip() for k in keywords_list if str(k).strip()]
                        step_desc = str(item.get('step', 'N/A')).strip()
                        if step_desc and step_desc != 'N/A':
                            temp_plan.append({"step_description": step_desc, "keywords": valid_keywords})
                        else:
                             print(f"Warning: Skipping step with invalid description: {item.get('step')}")
                    if temp_plan:
                        print("Successfully parsed research plan as JSON.")
                        return temp_plan
                    else:
                         print("Parsed JSON structure, but failed to extract valid step descriptions.")
                else:
                     print("Parsed JSON but structure/keys ('step', 'keywords') are incorrect or missing. Falling back.")
            else:
                print("Parsed JSON but it's not a list or is empty. Falling back.")
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing failed: {json_err}. Falling back to text parsing.")

    # 3. Fallback: Regex/Text Parsing
    print("Attempting Markdown/text parsing as fallback...")
    # Regex to capture numbered/bulleted lines and optional keywords in parentheses/brackets
    pattern_regex = re.compile(
        r"^\s*(?:\d+\.?\s*[:-]?|-|\*)\s*(.*?)" # Step number/bullet and description
        r"(?:\s*[\(\[]\s*Keywords?\s*[:\-]?\s*(.*?)\s*[\)\]]\s*)?$", # Optional keywords
        re.MULTILINE | re.IGNORECASE
    )
    matches = pattern_regex.findall(raw_response)
    if matches:
        print(f"Regex parsing found {len(matches)} potential steps.")
        plan = []
        for desc, keys_str in matches:
            desc = desc.strip()
            # Remove keyword part from description if captured separately or missed
            desc = re.sub(r'\s*\(?Keywords?[:\-]?.*?\)?$', '', desc, flags=re.IGNORECASE).strip()
            keys = []
            if keys_str:
                 # Handle comma or newline separated keywords
                 keys = [k.strip() for k in re.split(r'[,\n]', keys_str) if k.strip()]
            if desc: # Only add if description is non-empty
                 plan.append({"step_description": desc, "keywords": keys})
        if plan:
            print("Successfully parsed research plan using regex.")
            return plan
        else:
            print("Regex matched structure but failed to extract valid steps/keywords.")


    # 4. Final Fallback: Simple Line Parsing
    print("Regex parsing failed or yielded no results. Trying simple line parsing.")
    lines = raw_response.strip().split('\n')
    plan = []
    current_step = None
    keyword_markers = ["Keywords:", "keywords:", "Search Terms:", "search terms:", "Keywords -"]
    for line in lines:
        line = line.strip()
        if not line: continue
        # Try matching lines starting with a number/dash/bullet followed by text
        step_match = re.match(r"^\s*(?:step\s+)?(\d+)\s*[:.\-]?\s*(.*)|^\s*[-*+]\s+(.*)", line, re.IGNORECASE)
        if step_match:
            # Extract description based on which group matched
            step_desc = (step_match.group(2) or step_match.group(3) or "").strip()
            keys = []
            # Check if keywords are on the same line
            for marker in keyword_markers:
                marker_lower = marker.lower()
                if marker_lower in step_desc.lower():
                    parts = re.split(marker, step_desc, maxsplit=1, flags=re.IGNORECASE)
                    if len(parts) == 2:
                        step_desc = parts[0].strip()
                        # Handle comma/newline separators for keys
                        keys = [k.strip() for k in re.split(r'[,\n]', parts[1]) if k.strip()]
                        break # Stop after first keyword marker found
            if step_desc:
                current_step = {"step_description": step_desc, "keywords": keys}
                plan.append(current_step)
            else:
                current_step = None # Reset if description becomes empty after keyword removal
        elif current_step and not current_step["keywords"]:
             # Check if the line *only* contains keywords for the previous step
             is_keyword_line = False
             for marker in keyword_markers:
                 marker_lower = marker.lower()
                 if line.lower().startswith(marker_lower):
                     keys_str = line[len(marker):].strip()
                     # Handle comma/newline separators for keys
                     current_step["keywords"] = [k.strip() for k in re.split(r'[,\n]', keys_str) if k.strip()]
                     is_keyword_line = True
                     break
             # Avoid appending random lines as description continuation unless logic is very robust

    if plan:
        # Final cleanup: Remove steps with no description
        plan = [p for p in plan if p.get("step_description")]
        if plan:
             print("Parsed research plan using simple line-based approach.")
             return plan

    # 5. Absolute Failure Case
    print("All parsing methods failed to extract a valid research plan.")
    # Return a clear failure message, maybe include raw response snippet
    fail_msg = "Failed to parse plan structure from LLM response."
    if raw_response:
        fail_msg += f" Raw Response Snippet: '{raw_response[:200]}...'"
    return [{"step_description": fail_msg, "keywords": []}]


# search_duckduckgo - Added retry logic for rate limiting
def search_duckduckgo(keywords, max_results=MAX_SEARCH_RESULTS_PER_STEP, max_retries=2, initial_delay=DDGS_SEARCH_DELAY_SECONDS):
    """Performs a search on DuckDuckGo using the library with retry for rate limits."""
    query = " ".join(keywords)
    urls = []
    if not query: return []
    print(f"  -> DDGS Searching for: '{query}' (max_results={max_results})")

    delay = initial_delay # Use the base delay between *different* searches initially

    for attempt in range(max_retries + 1):
        try:
            with DDGS(timeout=15) as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                urls = [r['href'] for r in results if r and 'href' in r]
                print(f"  -> DDGS Found {len(urls)} URLs for '{query}'")
                return urls # Success
        except Exception as e:
            # Check if the error looks like a rate limit (heuristic)
            # DDGS library might raise specific exceptions in future, but for now check string
            is_rate_limit = "Ratelimit" in str(e) or "202" in str(e) or "429" in str(e)

            if is_rate_limit and attempt < max_retries:
                retry_delay = delay * (2 ** attempt) # Exponential backoff for retries
                print(f"  -> DDGS Rate limit detected for '{query}'. Retrying attempt {attempt + 1}/{max_retries} in {retry_delay:.1f} seconds...")
                time.sleep(retry_delay)
                continue # Go to next attempt
            else:
                # Log non-rate-limit errors or final rate limit failure
                print(f"Error searching DuckDuckGo for '{query}' (Attempt {attempt+1}): {e}")
                if is_rate_limit:
                    print(f"  -> Max retries reached for rate limit on '{query}'.")
                return [] # Return empty list on persistent failure

    return [] # Should not be reached if loop logic is correct

# scrape_url - No changes needed, already handles errors well
def scrape_url(url):
    """
    Scrapes text content from a given URL. Handles common errors.
    Returns a dictionary {'url': url, 'content': text} on success, None on failure.
    """
    log_url = url[:75] + '...' if len(url) > 75 else url
    # print(f"Attempting scrape: {log_url}") # Debug
    try:
        headers = {'User-Agent': USER_AGENT}
        # Consider adding 'Accept' header
        # headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True, stream=True)
        response.raise_for_status() # Check for 4xx/5xx errors immediately

        content_type = response.headers.get('content-type', '').lower()
        # Slightly broader check for text content, but prioritize html
        if 'html' not in content_type and 'text' not in content_type:
            print(f"Skipping non-HTML/text content: {log_url} (Type: {content_type})")
            response.close()
            return None

        content_length = response.headers.get('content-length')
        # Check content length *before* reading the whole thing if possible
        MAX_CONTENT_LENGTH_MB = 10
        if content_length and int(content_length) > MAX_CONTENT_LENGTH_MB * 1024 * 1024:
            print(f"Skipping large file (>{MAX_CONTENT_LENGTH_MB}MB): {log_url}")
            response.close()
            return None

        # Read content now, add another size check during read if needed
        html_content = response.content # Reads the entire content
        response.close() # Close the connection after reading

        # Check size again after reading if header was missing
        if not content_length and len(html_content) > MAX_CONTENT_LENGTH_MB * 1024 * 1024:
             print(f"Skipping large file discovered after download (>{MAX_CONTENT_LENGTH_MB}MB): {log_url}")
             return None

        # Try lxml first, fallback to html.parser
        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except Exception: # Catch potential lxml parsing errors too
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
            except Exception as parse_err:
                print(f"Failed to parse HTML for {log_url} with both lxml and html.parser: {parse_err}")
                return None

        # Remove unwanted tags aggressively
        for element in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "input", "textarea", "select", "img", "figure", "iframe", "video", "audio", "picture", "source", "noscript", "meta", "link", "svg", "canvas", "map", "area"]):
            element.decompose()

        # Try finding common main content containers
        main_content = soup.find('main') or \
                       soup.find('article') or \
                       soup.find('div', attrs={'role': 'main'}) or \
                       soup.find('section', id='content') or \
                       soup.find('div', id='content') or \
                       soup.find('div', class_=re.compile(r'\b(content|main|post|entry|article[-_]body|body[-_]content)\b', re.I)) or \
                       soup.find('div', id=re.compile(r'\b(content|main)\b', re.I))

        if main_content:
             text = main_content.get_text(separator='\n', strip=True)
        else:
            # Fallback to body if no specific main content found
             body = soup.find('body')
             if body:
                 text = body.get_text(separator='\n', strip=True)
             else:
                 # Absolute fallback if even body tag is missing (unlikely for valid HTML)
                 text = soup.get_text(separator='\n', strip=True)

        # Cleaning the extracted text
        lines = (line.strip() for line in text.splitlines())
        # Break multi-sentence lines into potential paragraphs, then strip whitespace aggressively
        chunks = (' '.join(phrase.split()) for line in lines for phrase in line.split("  ") if phrase.strip()) # Replace multiple spaces
        # Rejoin, keeping meaningful lines (more than just a couple of short words)
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk and len(chunk.split()) > 2) # Keep lines with > 2 words

        # Final check for meaningful content length
        MIN_MEANINGFUL_WORDS = 50 # Adjusted threshold
        meaningful_word_count = len([word for word in cleaned_text.split() if len(word) > 1]) # Count words > 1 char
        if meaningful_word_count < MIN_MEANINGFUL_WORDS:
             # print(f"Skipping due to low meaningful content ({meaningful_word_count} words < {MIN_MEANINGFUL_WORDS}): {log_url}") # Less verbose
             return None

        # print(f"Successfully scraped: {log_url} ({len(cleaned_text)} chars)") # Less verbose
        return {'url': url, 'content': cleaned_text}

    except requests.exceptions.Timeout:
        print(f"Timeout error fetching URL {log_url}")
    except requests.exceptions.HTTPError as e:
         # Specifically log HTTP errors (like 403, 404, 500)
        print(f"HTTP Error {e.response.status_code} fetching URL {log_url}: {e}")
    except requests.exceptions.RequestException as e:
        # Catch other request errors (DNS, connection, etc.)
        print(f"Request Error fetching URL {log_url}: {e}")
    except Exception as e:
        # Catch-all for other unexpected errors (parsing, processing)
        print(f"Error parsing or processing URL {log_url}: {type(e).__name__} - {e}")
        # import traceback # Uncomment for deep debugging if needed
        # traceback.print_exc()
    finally:
        # Ensure response is closed in case of exceptions before reading content fully
        if 'response' in locals() and response and hasattr(response, 'close') and callable(response.close):
             try:
                 response.close()
             except Exception:
                 pass # Ignore errors during close

    return None


# generate_bibliography_map - No changes needed
def generate_bibliography_map(scraped_sources_list):
    """Creates a map of {url: index} and a numbered list for prompts."""
    url_to_index = {data['url']: i + 1 for i, data in enumerate(scraped_sources_list)}
    numbered_list_str = "\n".join([f"{i+1}. {data['url']}" for i, data in enumerate(scraped_sources_list)])
    return url_to_index, numbered_list_str

# --- Flask Routes ---

@app.route('/')
def index():
    """Displays the main input form."""
    return render_template('index.html', pico_css=PICO_CSS_CDN)

@app.route('/research', methods=['POST'])
def research_start():
    """Redirects to the results page which connects to the SSE stream."""
    topic = request.form.get('topic')
    if not topic:
        return redirect(url_for('index'))
    encoded_topic = quote(topic)
    return render_template('results.html', topic=topic, encoded_topic=encoded_topic, pico_css=PICO_CSS_CDN)

@app.route('/stream')
def stream():
    """The main SSE route that performs research and streams progress."""
    encoded_topic = request.args.get('topic', '')
    topic = unquote(encoded_topic)
    if not topic:
        topic = "Default Topic - No Topic Provided" # More informative default

    def generate_updates():
        # ... (Variable initializations remain the same) ...
        scraped_sources_list = []
        all_found_urls_set = set()
        urls_to_scrape_list = []
        research_plan = []
        accumulated_synthesis = ""
        final_report_markdown = ""
        url_to_index_map = {}
        start_time_total = time.time()


        def send_event(data):
            """Helper to format and yield SSE events."""
            try:
                payload = json.dumps(data)
                yield f"data: {payload}\n\n"
            except TypeError as e:
                print(f"Error serializing data for SSE: {e}. Data: {data}")
                # Try serializing a safe version
                try:
                    safe_data = {'type': data.get('type', 'error'), 'message': f"Serialization Error: {e}"}
                    payload = json.dumps(safe_data)
                    yield f"data: {payload}\n\n"
                except Exception: # Fallback if even that fails
                    yield "data: {\"type\": \"error\", \"message\": \"Internal server error during SSE serialization\"}\n\n"


        def send_progress(message):
            yield from send_event({'type': 'progress', 'message': message})

        def send_error_event(message):
            """Sends an error event via SSE."""
            print(f"ERROR: {message}")
            yield from send_event({'type': 'error', 'message': message})

        try:
            # === Step 1: Generate Research Plan ===
            yield from send_progress(f"Generating research plan for: '{topic}' using {GOOGLE_MODEL_NAME}...")
            # Prompt remains the same
            plan_prompt = f"""
            Create a detailed, step-by-step research plan with 10-15 distinct steps for the topic: "{topic}"
            Each step should represent a specific question or area of inquiry related to the topic.
            Format the output STRICTLY as a JSON list of objects within a ```json ... ``` block. Each object must have two keys:
            1. "step": A string describing the research step/question.
            2. "keywords": A list of 2-4 relevant keyword strings for searching about this step.

            Example format:
            ```json
            [
              {{"step": "Define the core concept", "keywords": ["term definition", "term explanation"]}},
              {{"step": "Explore historical origins", "keywords": ["history of term", "term origins"]}},
              ... (10-15 steps total)
            ]
            ```

            Ensure the output is ONLY the valid JSON list inside the markdown block. No introductory text, no explanations outside the JSON block.
            Generate 10 to 15 relevant steps for the specific topic: "{topic}".
            """
            try:
                plan_response = call_gemini(plan_prompt)
            except Exception as e:
                 yield from send_error_event(f"Failed to generate research plan from LLM ({GOOGLE_MODEL_NAME}): {e}")
                 return

            research_plan = parse_research_plan(plan_response)

            # Improved check for failed parsing
            if not research_plan or not isinstance(research_plan, list) or not all(isinstance(step, dict) and 'step_description' in step and 'keywords' in step for step in research_plan) or \
               (len(research_plan) == 1 and research_plan[0]["step_description"].startswith("Failed")):
                 fail_reason = "Could not parse valid plan structure."
                 if research_plan and isinstance(research_plan, list) and research_plan[0].get("step_description", "").startswith("Failed"):
                      fail_reason = research_plan[0]["step_description"] # Use specific failure message if available
                 raw_snippet = f" Raw LLM Response Snippet: '{plan_response[:150]}...'" if plan_response else " (LLM Response was empty)"
                 yield from send_error_event(f"Failed to create or parse a valid research plan. Reason: {fail_reason}.{raw_snippet}")
                 return

            yield from send_progress(f"Generated {len(research_plan)} step plan:")
            log_limit = 5
            for i, step in enumerate(research_plan[:log_limit]):
                 kw_str = ', '.join(step.get('keywords',[]))
                 yield from send_progress(f"  {i+1}. {step['step_description']} (Keywords: {kw_str[:60]}{'...' if len(kw_str)>60 else ''})")
            if len(research_plan) > log_limit:
                 yield from send_progress(f"  ... and {len(research_plan) - log_limit} more steps.")

            # === Step 2a: Search and Collect URLs ===
            yield from send_progress("Starting web search to collect URLs...")
            start_search_time = time.time()
            urls_collected_count = 0
            all_urls_from_search = []
            search_step_errors = 0
            for i, step in enumerate(research_plan):
                step_desc = step.get('step_description', f'Unnamed Step {i+1}')
                keywords = step.get('keywords', [])

                yield from send_progress(f"Searching - Step {i+1}/{len(research_plan)}: '{step_desc[:70]}{'...' if len(step_desc)>70 else ''}'")
                if not keywords:
                    yield from send_progress("  -> No keywords provided, skipping search for this step.")
                    continue

                # Use search function with built-in delay and retry
                step_search_results = search_duckduckgo(
                    keywords,
                    MAX_SEARCH_RESULTS_PER_STEP,
                    initial_delay=0 # Don't add extra delay before the first attempt inside search_duckduckgo
                )

                if not step_search_results:
                    search_step_errors += 1
                    yield from send_progress(f"  -> Search failed or returned no results for keywords: {keywords}")

                all_urls_from_search.extend(step_search_results)

                # --- CHANGED: Add delay *between* different keyword searches ---
                if i < len(research_plan) - 1: # Don't sleep after the last search
                    # print(f"  -> Delaying {DDGS_SEARCH_DELAY_SECONDS}s before next search...") # Debug
                    time.sleep(DDGS_SEARCH_DELAY_SECONDS)


            yield from send_progress(f"Search phase completed in {time.time() - start_search_time:.2f}s.")
            yield from send_progress(f"Found {len(all_urls_from_search)} total URL results initially ({search_step_errors} search steps had issues).")

            # --- Filtering (Remains the same logic) ---
            unique_urls = list(dict.fromkeys(filter(None, all_urls_from_search))) # Filter None before unique
            yield from send_progress(f"Processing {len(unique_urls)} unique URLs...")

            for url in unique_urls:
                 if urls_collected_count >= MAX_TOTAL_URLS_TO_SCRAPE:
                      yield from send_progress(f"  -> Reached URL collection limit ({MAX_TOTAL_URLS_TO_SCRAPE}).")
                      break
                 if url in all_found_urls_set: continue # Should be redundant with unique_urls but safe

                 is_file = url.lower().split('?')[0].split('#')[0].endswith(('.pdf', '.jpg', '.png', '.gif', '.zip', '.mp4', '.mp3', '.docx', '.xlsx', '.pptx', '.webp', '.svg', '.xml', '.css', '.js', '.jpeg', '.doc', '.xls', '.ppt', '.txt', '.exe', '.dmg', '.iso', '.rar'))
                 is_mailto = url.lower().startswith('mailto:')
                 is_javascript = url.lower().startswith('javascript:')
                 is_ftp = url.lower().startswith('ftp:')
                 is_tel = url.lower().startswith('tel:')
                 is_valid_http = url.startswith(('http://', 'https://'))
                 # Add common non-content domains if needed
                 # is_social_media = any(domain in url for domain in ['youtube.com', 'facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com'])

                 if is_valid_http and not is_file and not is_mailto and not is_javascript and not is_ftp and not is_tel: # and not is_social_media:
                      urls_to_scrape_list.append(url)
                      all_found_urls_set.add(url) # Track added URLs
                      urls_collected_count += 1
                 # else: # Log skipped URLs if needed (can be verbose)
                 #    reason = "file" if is_file else "mailto" if is_mailto else "javascript" if is_javascript else "ftp" if is_ftp else "tel" if is_tel else "invalid http" if not is_valid_http else "other"
                 #    yield from send_progress(f"  -> Skipping filtered URL ({reason}): {url[:70]}...")

            yield from send_progress(f"Selected {len(urls_to_scrape_list)} valid, unique URLs for scraping (limit: {MAX_TOTAL_URLS_TO_SCRAPE}).")

            if not urls_to_scrape_list:
                 yield from send_error_event("No suitable URLs found to scrape after searching and filtering. Check search results or filtering logic.")
                 return

            # === Step 2b: Scrape URLs Concurrently ===
            yield from send_progress(f"Starting concurrent scraping of {len(urls_to_scrape_list)} URLs (Max workers: {MAX_WORKERS})...")
            start_scrape_time = time.time()
            total_scraped_successfully = 0
            processed_scrape_count = 0
            scraped_sources_list = [] # Re-initialize

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_url = {executor.submit(scrape_url, url): url for url in urls_to_scrape_list}

                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    processed_scrape_count += 1
                    try:
                        result_dict = future.result() # result() will raise exceptions from scrape_url
                        if result_dict and isinstance(result_dict, dict) and result_dict.get('content'):
                            scraped_sources_list.append(result_dict)
                            total_scraped_successfully += 1
                        # else: # Failure/empty scrape already logged inside scrape_url
                        #    pass
                    except Exception as exc:
                        # This catches errors *if* scrape_url itself failed unexpectedly
                        # It shouldn't happen often as scrape_url handles its errors
                        yield from send_progress(f"    -> Unexpected Error processing scrape result for {url[:60]}...: {exc}")

                    # Update progress less frequently
                    update_interval = max(1, len(urls_to_scrape_list) // 10) # Update ~10 times
                    if processed_scrape_count % update_interval == 0 or processed_scrape_count == len(urls_to_scrape_list):
                          progress_perc = (processed_scrape_count * 100) // len(urls_to_scrape_list)
                          yield from send_progress(f"  -> Scraping progress: {processed_scrape_count}/{len(urls_to_scrape_list)} ({progress_perc}%). Success: {total_scraped_successfully}")


            duration = time.time() - start_scrape_time
            yield from send_progress(f"Finished scraping. Successfully scraped {total_scraped_successfully}/{len(urls_to_scrape_list)} URLs in {duration:.2f} seconds.")
            if total_scraped_successfully < len(urls_to_scrape_list):
                 yield from send_progress(f"  -> Note: {len(urls_to_scrape_list) - total_scraped_successfully} URLs failed to scrape (check logs for HTTP errors like 403, timeouts, etc.).")


            if not scraped_sources_list:
                yield from send_error_event("Failed to scrape any content successfully. Check website accessibility or scraper logic.")
                return

            # Re-order scraped list (optional)
            scraped_url_map = {item['url']: item for item in scraped_sources_list}
            ordered_scraped_list = [scraped_url_map[url] for url in urls_to_scrape_list if url in scraped_url_map]
            scraped_sources_list = ordered_scraped_list

            # === Bibliography Map ===
            url_to_index_map, bibliography_prompt_list = generate_bibliography_map(scraped_sources_list)

            # === Step 3: Synthesize with Citations (Streaming) ===
            yield from send_progress(f"Synthesizing relevant information using {GOOGLE_MODEL_NAME}...")
            yield from send_event({'type': 'stream_start', 'target': 'synthesis'})

            # --- Context Preparation & Truncation (Improved) ---
            # Estimate token count roughly (1 char ~ 0.25 tokens, plus overhead)
            # Gemini 1.5 Flash has 1M token context, Pro has 1M (up to 2M)
            # Let's aim well below the limit to be safe and manage cost/time
            # Target max tokens for context (adjust based on model and budget)
            TARGET_MAX_CONTEXT_TOKENS = 750000
            CHARS_PER_TOKEN_ESTIMATE = 4 # Conservative estimate
            MAX_CONTEXT_CHARS = int(TARGET_MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN_ESTIMATE * 0.9) # Use 90%

            context_for_llm_structured = []
            current_chars = 0
            sources_included_count = 0
            original_source_count = len(scraped_sources_list)

            for source in scraped_sources_list:
                 # Estimate size: URL length + content length + JSON overhead (~50 chars)
                 source_len = len(source.get('url', '')) + len(source.get('content', '')) + 50
                 if current_chars + source_len <= MAX_CONTEXT_CHARS:
                     context_for_llm_structured.append(source)
                     current_chars += source_len
                     sources_included_count += 1
                 else:
                     yield from send_progress(f"  -> Warning: Estimated context limit (~{MAX_CONTEXT_CHARS // 1000}k chars) reached. Using first {sources_included_count}/{original_source_count} sources for synthesis.")
                     break # Stop adding sources

            estimated_chars = sum(len(item.get('content', '')) for item in context_for_llm_structured)
            estimated_tokens = estimated_chars / CHARS_PER_TOKEN_ESTIMATE
            yield from send_progress(f"  -> Synthesis using {len(context_for_llm_structured)} sources (~{estimated_chars // 1000}k chars / ~{estimated_tokens // 1000}k tokens).")

            # --- Synthesis Prompt (Remains the same) ---
            synthesis_prompt = f"""
            Analyze the following scraped web content related to the topic "{topic}", following the research plan provided.
            Your goal is to synthesize the key information relevant to each step of the plan, citing sources accurately using ONLY the provided URLs.

            Research Topic: {topic}

            Research Plan ({len(research_plan)} steps):
            ```json
            {json.dumps(research_plan, indent=2)}
            ```

            Scraped Source Content (List of {len(context_for_llm_structured)} JSON objects, each with 'url' and 'content'):
            ```json
            {json.dumps(context_for_llm_structured, indent=2, ensure_ascii=False)}
            ```

            Instructions:
            1. Process each step in the Research Plan sequentially.
            2. For each step, carefully review ALL the scraped source content to find relevant information. If multiple sources provide similar info, synthesize concisely.
            3. Extract and synthesize the most pertinent facts, findings, or data related *specifically* to that research step. Focus on unique information per step.
            4. **CRITICALLY IMPORTANT AND MANDATORY**: Immediately after stating **any** piece of information derived from a source, cite the source using the exact format: `[Source URL: <full_url_here>]`. Cite every distinct piece of information from its origin. Do not group citations.
            5. If you cannot find relevant information for a specific plan step within the provided sources, explicitly state: "No specific information found for this step in the provided sources." Do not invent information or cite irrelevant content.
            6. Structure your output clearly using Markdown. Use a heading (e.g., `### Step X: <Step Description>`) for each plan step. Use paragraphs for readability.
            7. Output ONLY the synthesized information with inline URL citations, structured by plan step. Do NOT include an introduction, conclusion, summary, bibliography, or any other text outside the step-by-step synthesis in this response. Separate each step's section clearly (e.g., using `---` between steps).
            """
            accumulated_synthesis = ""
            synthesis_stream_error = None
            # --- Synthesis Stream Processing ---
            try:
                stream_generator = stream_gemini(synthesis_prompt)
                for result in stream_generator:
                    if result['type'] == 'chunk':
                        yield from send_event({'type': 'llm_chunk', 'content': result['content'], 'target': 'synthesis'})
                        accumulated_synthesis += result['content']
                    elif result['type'] == 'stream_error':
                        synthesis_stream_error = result['message']
                        # Send error to client, but maybe don't terminate immediately?
                        # Let's terminate for now on critical errors like API key/quota
                        yield from send_error_event(f"LLM stream error during synthesis ({GOOGLE_MODEL_NAME}): {synthesis_stream_error}")
                        if "API key" in synthesis_stream_error or "quota" in synthesis_stream_error.lower():
                            return # Terminate on critical auth/quota issues
                        # Otherwise, maybe log and try to continue? For now, terminate.
                        return
                    elif result['type'] == 'stream_warning':
                         yield from send_progress(f"LLM Stream Warning (Synthesis): {result['message']}")
                         # Continue processing despite warnings
                    elif result['type'] == 'stream_end':
                         yield from send_progress(f"Synthesis stream finished.")
                         break # Exit loop

                if synthesis_stream_error: # Check again if loop exited due to non-critical error handled above
                     print("Terminating after non-critical stream error during synthesis.")
                     return

            except Exception as e:
                 yield from send_error_event(f"Fatal error processing LLM synthesis stream: {e}")
                 import traceback
                 traceback.print_exc()
                 return

            yield from send_progress(f"Synthesis generation completed.")
            if not accumulated_synthesis.strip():
                 yield from send_error_event("Synthesis resulted in empty content. Check LLM response, source relevance, or potential safety filtering.")
                 # Optionally dump the synthesis prompt for debugging
                 # print("--- Failed Synthesis Prompt ---")
                 # print(synthesis_prompt[:5000] + "...")
                 # print("--- End Failed Synthesis Prompt ---")
                 return

            # === Step 4: Generate Final Report (Streaming) ===
            yield from send_progress(f"Generating final report using {GOOGLE_MODEL_NAME}...")
            yield from send_event({'type': 'stream_start', 'target': 'report'})

            # --- Report Prompt Preparation (Truncation if needed) ---
            # Estimate size for report prompt (less likely to hit limit here)
            report_components_size = len(topic) + len(json.dumps(research_plan)) + len(accumulated_synthesis) + len(bibliography_prompt_list)
            report_estimated_tokens = report_components_size / CHARS_PER_TOKEN_ESTIMATE

            if report_estimated_tokens > TARGET_MAX_CONTEXT_TOKENS * 0.95: # Check if close to limit
                 yield from send_progress(f"  -> Warning: Inputs for final report prompt large (~{report_estimated_tokens // 1000}k est. tokens). Synthesis might be truncated in prompt.")
                 # Truncate synthesis if needed (same logic as before)
                 available_chars_for_synthesis = MAX_CONTEXT_CHARS - (len(topic) + len(json.dumps(research_plan)) + len(bibliography_prompt_list) + 2000) # Reserve buffer
                 if available_chars_for_synthesis < 0: available_chars_for_synthesis = 0
                 truncated_synthesis = accumulated_synthesis[:available_chars_for_synthesis] + "\n\n... [Synthesis truncated in report prompt due to length limit]"
                 yield from send_progress(f"  -> Truncated synthesis to ~{available_chars_for_synthesis // 1000}k chars for report prompt.")
            else:
                 truncated_synthesis = accumulated_synthesis


            # --- Report Prompt (Remains the same structure) ---
            report_prompt = f"""
            Create a comprehensive Markdown research report on the topic: "{topic}".

            You have the following inputs:
            1. The Original Research Plan ({len(research_plan)} steps):
               ```json
               {json.dumps(research_plan, indent=2)}
               ```

            2. Synthesized Information with Raw URL Citations (Result of previous step):
               ```markdown
               {truncated_synthesis}
               ```

            3. Bibliography Map (URL to Reference Number - {len(url_to_index_map)} sources):
               ```
               {bibliography_prompt_list}
               ```

            Instructions:
            1. Write a final research report in well-structured Markdown format. Use clear headings, paragraphs, and lists where appropriate.
            2. The report MUST include the following sections using Markdown headings:
                - `# Research Report: {topic}` (Main Title)
                - `## Introduction`: Briefly introduce the topic "{topic}", state the purpose of the report, and outline the research scope based on the plan's steps (list or describe them).
                - `## Findings`: Organize the main body strictly according to the {len(research_plan)} research plan steps. For each step:
                    - Use a subheading (e.g., `### Step X: <Step Description from Plan>`).
                    - Integrate the relevant "Synthesized Information" for that step provided above. Ensure smooth flow and readability.
                    - **CRITICALLY IMPORTANT AND MANDATORY: Replace EVERY inline URL citation `[Source URL: <full_url_here>]` from the "Synthesized Information" with its corresponding Markdown footnote marker `[^N]`. Use the provided "Bibliography Map" to find the correct number N for the EXACT URL. Ensure the mapping is precise.**
                    - If a URL citation appears in the synthesis that is *not* present in the Bibliography Map (e.g., due to context truncation, LLM hallucination, or if the source wasn't included in the synthesis context), OMIT that specific citation marker entirely. Do not invent footnote numbers or include broken references `[^?]`. Rephrase the sentence slightly if removing the citation makes it awkward.
                - `## Conclusion`: Summarize the key findings derived from the synthesis across the most important plan steps. Briefly mention any significant limitations encountered (e.g., steps where information was lacking in sources, potential biases identified, scraping difficulties, number of sources analyzed). Suggest potential areas for further research if applicable.
            3. Append a `## Bibliography` section at the very end of the report.
            4. In the Bibliography section, list all the sources from the "Bibliography Map" in numerical order (1, 2, 3...). Use the standard Markdown footnote definition format for each entry: `[^N]: <full_url_here>`. The URL should be the plain URL (Markdown parsers typically make these clickable).
            5. Ensure the final output is **only** the complete Markdown report, adhering strictly to the structure, formatting, citation replacement, and bibliography instructions. Do not include any commentary about the process, apologies, or explanations outside the report content itself.
            """
            final_report_markdown = "" # Reset before accumulating
            report_stream_error = None
            # --- Report Stream Processing ---
            try:
                stream_generator = stream_gemini(report_prompt)
                for result in stream_generator:
                    if result['type'] == 'chunk':
                        yield from send_event({'type': 'llm_chunk', 'content': result['content'], 'target': 'report'})
                        final_report_markdown += result['content']
                    elif result['type'] == 'stream_error':
                        report_stream_error = result['message']
                        yield from send_error_event(f"LLM stream error during report generation ({GOOGLE_MODEL_NAME}): {report_stream_error}")
                        if "API key" in report_stream_error or "quota" in report_stream_error.lower():
                             return # Terminate
                        # Otherwise, terminate for now
                        return
                    elif result['type'] == 'stream_warning':
                         yield from send_progress(f"LLM Stream Warning (Report): {result['message']}")
                         # Continue
                    elif result['type'] == 'stream_end':
                         yield from send_progress(f"Report stream finished.")
                         break # Exit loop

                if report_stream_error:
                    print("Terminating after non-critical stream error during report generation.")
                    return

            except Exception as e:
                 yield from send_error_event(f"Fatal error processing LLM report stream: {e}")
                 import traceback
                 traceback.print_exc()
                 return

            yield from send_progress(f"Report generation completed.")
            if not final_report_markdown.strip():
                 yield from send_error_event("Report generation resulted in empty content. Check LLM response or synthesis input.")
                 return

            # --- Final Processing and Completion ---
            yield from send_progress("Processing final report for display...")

            # Convert final report Markdown to HTML
            try:
                # Ensure footnotes extension is enabled, plus others for common Markdown features
                report_html = md_lib.markdown(
                    final_report_markdown,
                    extensions=['footnotes', 'fenced_code', 'tables', 'nl2br', 'attr_list', 'md_in_html']
                )
            except Exception as md_err:
                 yield from send_error_event(f"Failed to convert final Markdown report to HTML: {md_err}")
                 from html import escape
                 # Provide raw markdown in a <pre> tag as fallback
                 report_html = f"<h3>Markdown Conversion Error</h3><p>Could not render report as HTML using 'markdown' library. Error: {escape(str(md_err))}. Raw Markdown content below:</p><hr><pre style='white-space: pre-wrap; word-wrap: break-word;'>{escape(final_report_markdown)}</pre>"


            # Prepare final data payload
            preview_limit = 3000
            raw_data_preview_list = []
            current_preview_len = 0
            # Show preview of first few *successfully scraped* sources
            for src in scraped_sources_list:
                src_dump = json.dumps(src, indent=2, ensure_ascii=False)
                if current_preview_len + len(src_dump) < preview_limit:
                    raw_data_preview_list.append(src_dump)
                    current_preview_len += len(src_dump)
                else:
                    break # Stop adding previews if limit reached
            raw_data_preview = "[\n" + ",\n".join(raw_data_preview_list) + "\n]"
            if len(scraped_sources_list) > len(raw_data_preview_list):
                 raw_data_preview += f"\n... (Preview includes {len(raw_data_preview_list)}/{len(scraped_sources_list)} successfully scraped sources)"
            elif not scraped_sources_list:
                 raw_data_preview = "None (No sources scraped successfully)"


            final_data = {
                'type': 'complete',
                'report_html': report_html,
                'report_markdown': final_report_markdown, # Send raw markdown for download
                'raw_scraped_data_preview': raw_data_preview,
                'docx_available': PANDOC_AVAILABLE # Flag if DOCX download is possible
            }
            yield from send_event(final_data)
            end_time_total = time.time()
            yield from send_progress(f"Research process completed successfully in {end_time_total - start_time_total:.2f} seconds.")

        except Exception as e:
            print(f"An unexpected error occurred during stream generation: {e}")
            import traceback
            traceback.print_exc()
            error_msg = f"Unexpected server error during research process: {type(e).__name__} - {e}"
            yield from send_error_event(error_msg)
        finally:
             # Signal the end of the stream regardless of success/failure
             yield from send_event({'type': 'stream_terminated'})


    # Ensure headers prevent caching for SSE
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no', # Useful for Nginx proxying
        'Connection': 'keep-alive'
    }
    return Response(stream_with_context(generate_updates()), headers=headers)


# --- DOCX Download Route - Now checks PANDOC_AVAILABLE flag ---
@app.route('/download_docx', methods=['POST'])
def download_docx():
    """Converts the received Markdown report to DOCX and sends it as a download."""
    if not PANDOC_AVAILABLE:
        # Return JSON error if Pandoc wasn't found at startup
        return jsonify({"success": False, "message": "DOCX download is disabled: Pandoc executable not found or pypandoc setup issue. Install 'pypandoc-binary'."}), 400

    markdown_content = request.form.get('markdown_report')
    topic = request.form.get('topic', 'Research_Report') # Get topic for filename

    if not markdown_content:
        return jsonify({"success": False, "message": "Error: No Markdown content received for conversion."}), 400

    try:
        # Use pypandoc to convert Markdown text to DOCX bytes
        # Ensure input encoding is handled correctly, default is usually UTF-8
        # If using pypandoc-binary, it should find the bundled pandoc
        docx_bytes = pypandoc.convert_text(
            markdown_content,
            'docx',
            format='md',
            encoding='utf-8',
            # extra_args=['--reference-doc=my_template.docx'] # Optional: specify a template
        )

        buffer = BytesIO(docx_bytes)
        buffer.seek(0)

        # Sanitize topic for filename
        safe_filename_topic = re.sub(r'[^\w\s-]', '', topic).strip()
        safe_filename_topic = re.sub(r'[-\s]+', '_', safe_filename_topic)
        filename_base = f"{safe_filename_topic}_Research_Report"
        filename = f"{filename_base[:200]}.docx" # Limit filename length

        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

    except Exception as e:
        # Catch potential errors during conversion (even if PANDOC_AVAILABLE was true initially)
        print(f"Error converting Markdown to DOCX using pypandoc: {e}")
        # Check if it's specifically the "No pandoc was found" error again
        if "No pandoc was found" in str(e):
            msg = "Error during conversion: Pandoc executable could not be located by pypandoc at runtime. Ensure 'pypandoc-binary' is installed or pandoc is correctly in PATH."
        else:
            msg = f"An error occurred during DOCX conversion: {e}"
        # Log the error and return JSON
        return jsonify({"success": False, "message": msg}), 500


# --- Run the App ---
if __name__ == '__main__':
    # --- CHANGED: Updated installation comment ---
    # Ensure libraries are installed: pip install Flask python-dotenv google-generativeai duckduckgo-search beautifulsoup4 requests lxml Markdown pypandoc-binary flask[async]
    # Set GOOGLE_API_KEY in your .env file
    print(f"Using Google Model: {GOOGLE_MODEL_NAME}")
    print(f"Pandoc/DOCX available check at startup: {PANDOC_AVAILABLE}")
    # Recommended: Use a production WSGI server like gunicorn or uwsgi for deployment
    # Example: gunicorn --workers 4 --threads 2 --bind 0.0.0.0:5001 your_script_name:app
    app.run(debug=True, host='127.0.0.1', port=5001, threaded=True)