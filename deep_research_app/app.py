# --- START DEEP DEBUGGING ---
import sys
import os
print("--- Python Interpreter Info ---")
print(f"sys.executable: {sys.executable}")
print(f"os.path.dirname(sys.executable): {os.path.dirname(sys.executable)}")
print("\n--- sys.path ---")
for i, p in enumerate(sys.path):
    print(f"{i}: {p}")
print("----------------------------\n")

print("--- Attempting direct import of html2docx ---")
_html2docx_direct_import_worked = False
_html2docx_location = "Not found directly"
try:
    import html2docx
    _html2docx_direct_import_worked = True
    _html2docx_location = getattr(html2docx, '__file__', 'Location attribute missing')
    print(f"SUCCESS: Direct import of html2docx worked.")
    print(f"Location found: {_html2docx_location}")
except ImportError as e:
    print(f"FAILURE: Direct import failed. Error: {e}")
except Exception as e:
    print(f"FAILURE: Direct import failed with unexpected error: {type(e).__name__} - {e}")
print("-------------------------------------------\n")
# --- END DEEP DEBUGGING ---

# --- MOVE THE TRY/EXCEPT BLOCK IMMEDIATELY BELOW DEEP DEBUG ---
DOCX_CONVERSION_AVAILABLE = False
print("--- Attempting standard import (from html2docx import Html2Docx) ---")
try:
    # Try the specific import needed later
    from html2docx import Html2Docx
    print("SUCCESS: 'from html2docx import Html2Docx' worked immediately after deep debug.")
    DOCX_CONVERSION_AVAILABLE = True
except ImportError as e:
    print(f"FAILURE: 'from html2docx import Html2Docx' FAILED immediately after deep debug. Error: {e}")
    # If direct import worked but this failed, print extra info
    if _html2docx_direct_import_worked:
        print(f"    NOTE: Direct 'import html2docx' had previously SUCCEEDED, finding it at: {_html2docx_location}")
except Exception as e:
     print(f"FAILURE: 'from html2docx import Html2Docx' failed with UNEXPECTED error immediately after deep debug: {type(e).__name__} - {e}")
print("-----------------------------------------------------------------\n")

# --- NOW THE REST OF THE IMPORTS ---
#!python
# import os # No need to re-import
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, stream_with_context, send_file
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import markdown as md_lib
from urllib.parse import quote, unquote
import concurrent.futures
from io import BytesIO
import traceback

# --- ADDED DEBUG PRINT HERE ---
# This will now reflect the result of the import attempt done *earlier*
print(f"DEBUG: DOCX_CONVERSION_AVAILABLE set to: {DOCX_CONVERSION_AVAILABLE} at script start (based on immediate check).")
# -----------------------------


# --- Configuration ---
load_dotenv()
# --- Google API Key and Model Name ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Use a valid Google model name (e.g., gemini-1.5-pro-latest, gemini-1.5-flash-latest)
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash-latest") # Corrected model name

# --- Search Configuration ---
# Max results *per search engine* per research step
MAX_SEARCH_RESULTS_PER_ENGINE_STEP = 5 # Reduced to avoid overwhelming results quickly
MAX_TOTAL_URLS_TO_SCRAPE = 50 # Reduced default for faster testing/lower cost
MAX_WORKERS = 10
REQUEST_TIMEOUT = 12
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
PICO_CSS_CDN = "https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css"
# Delay between searches *within the same engine* (DDGS retry)
DDGS_RETRY_DELAY_SECONDS = 3.0
# Delay *between different search queries* (e.g., between steps or engines)
INTER_SEARCH_DELAY_SECONDS = 1.5

# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Initialize Google Generative AI Client ---
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to configure Google Generative AI: {e}")

# --- Helper Functions ---

# --- REFACTORED: call_gemini (unchanged) ---
def call_gemini(prompt, system_prompt=None, max_retries=3, delay=5):
    """Calls the specified Google Gemini model with retry logic."""
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    generation_config = genai.types.GenerationConfig(
        temperature=0.6,
        # max_output_tokens=8192 # Set if needed, depends on model
    )
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
            response = model.generate_content(effective_prompt)

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = str(response.prompt_feedback.block_reason)
                 safety_ratings_str = str(getattr(response.prompt_feedback, 'safety_ratings', 'N/A'))
                 raise ValueError(f"API response blocked by safety settings. Block Reason: {block_reason}. Safety Ratings: {safety_ratings_str}")

            if not response.candidates:
                 finish_details = getattr(response, 'finish_details', 'N/A')
                 raise ValueError(f"API response contained no candidates. Finish Details: {finish_details}")

            candidate = response.candidates[0]
            if candidate.finish_reason not in [None, 1, "STOP", "MAX_TOKENS"]:
                 raise ValueError(f"API response candidate finished unexpectedly. Finish Reason: {candidate.finish_reason}. Safety Ratings: {getattr(candidate, 'safety_ratings', 'N/A')}")

            if not candidate.content or not candidate.content.parts:
                 raise ValueError(f"API response candidate has no content parts. Finish Reason: {candidate.finish_reason}")

            response_content = response.text.strip()
            if not response_content:
                 raise ValueError("API response generated empty text content after successful call.")

            return response_content

        except Exception as e:
            print(f"Error calling Google Gemini API (Attempt {attempt + 1}/{max_retries}): {e}")
            if "API key not valid" in str(e):
                 print("Critical Error: Invalid Google API Key.")
                 raise
            if "quota" in str(e).lower():
                 print("Warning: Quota possibly exceeded.")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Failing LLM call.")
                raise

# --- REFACTORED: stream_gemini (unchanged) ---
def stream_gemini(prompt, system_prompt=None):
    """
    Calls the Google Gemini model with streaming enabled and yields content chunks.
    """
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    generation_config = genai.types.GenerationConfig(
        temperature=0.6,
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

        complete_response_text = ""
        stream_blocked = False

        for chunk in stream:
             if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                  block_reason_str = str(chunk.prompt_feedback.block_reason)
                  safety_ratings_str = str(getattr(chunk.prompt_feedback, 'safety_ratings', 'N/A'))
                  print(f"ERROR: LLM stream blocked by prompt safety filters. Reason: {block_reason_str}. Ratings: {safety_ratings_str}")
                  yield {'type': 'stream_error', 'message': f'LLM prompt blocked by safety filters (Reason: {block_reason_str}). Cannot generate response.'}
                  stream_blocked = True
                  break

             if not stream_blocked and chunk.candidates:
                 candidate = chunk.candidates[0]
                 if candidate.finish_reason and candidate.finish_reason not in [None, 1, "STOP", "MAX_TOKENS"]:
                     finish_reason_str = str(candidate.finish_reason)
                     safety_ratings_str = str(getattr(candidate, 'safety_ratings', 'N/A'))
                     print(f"WARNING: LLM stream stopped during generation. Reason: {finish_reason_str}. Safety: {safety_ratings_str}")
                     yield {'type': 'stream_warning', 'message': f'LLM stream may have been interrupted or blocked during generation (Reason: {finish_reason_str}). Output may be incomplete.'}
                     if finish_reason_str == "SAFETY":
                         stream_blocked = True
                         break

             if not stream_blocked and hasattr(chunk, 'text') and chunk.text:
                  chunk_text = chunk.text
                  complete_response_text += chunk_text
                  yield {'type': 'chunk', 'content': chunk_text}

        if not stream_blocked:
             yield {'type': 'stream_end', 'finish_reason': 'IterationComplete'}

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


# --- parse_research_plan (unchanged) ---
def parse_research_plan(llm_response):
    # ... (parsing logic remains the same) ...
    plan = []
    if not llm_response:
        print("Error: Received empty response from LLM for plan generation.")
        return [{"step_description": "Failed - Empty LLM response", "keywords": []}]

    raw_response = llm_response.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_response, re.MULTILINE)
    json_str = None
    if match:
        json_str = match.group(1).strip()
    else:
        if (raw_response.startswith('[') and raw_response.endswith(']')) or \
           (raw_response.startswith('{') and raw_response.endswith('}')):
            json_str = raw_response

    if json_str:
        try:
            data = json.loads(json_str)
            if isinstance(data, list) and data:
                if all(isinstance(item, dict) and 'step' in item and 'keywords' in item for item in data):
                    temp_plan = []
                    for item in data:
                        keywords_list = item.get('keywords', [])
                        if isinstance(keywords_list, str):
                            keywords_list = [k.strip() for k in re.split(r'[,\n]', keywords_list) if k.strip()]
                        elif not isinstance(keywords_list, list):
                            keywords_list = []
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
                        print("Parsed JSON structure, but failed to extract valid steps.")
                else:
                     print("Parsed JSON but structure/keys ('step', 'keywords') are incorrect or missing.")
            else:
                print("Parsed JSON but it's not a list or is empty.")
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing failed: {json_err}. Falling back.")

    print("Attempting Markdown/text parsing as fallback...")
    pattern_regex = re.compile(
        r"^\s*(?:\d+\.?\s*[:-]?|-|\*)\s*(.*?)"
        r"(?:\s*[\(\[]\s*Keywords?\s*[:\-]?\s*(.*?)\s*[\)\]]\s*)?$",
        re.MULTILINE | re.IGNORECASE
    )
    matches = pattern_regex.findall(raw_response)
    if matches:
        plan = []
        for desc, keys_str in matches:
            desc = desc.strip()
            desc = re.sub(r'\s*\(?Keywords?[:\-]?.*?\)?$', '', desc, flags=re.IGNORECASE).strip()
            keys = []
            if keys_str:
                 keys = [k.strip() for k in re.split(r'[,\n]', keys_str) if k.strip()]
            if desc:
                 plan.append({"step_description": desc, "keywords": keys})
        if plan:
            print("Successfully parsed research plan using regex.")
            return plan
        else:
             print("Regex matched structure but failed to extract valid steps/keywords.")

    print("Regex parsing failed. Trying simple line parsing.")
    lines = raw_response.strip().split('\n')
    plan = []
    current_step = None
    keyword_markers = ["Keywords:", "keywords:", "Search Terms:", "search terms:", "Keywords -"]
    for line in lines:
        line = line.strip()
        if not line: continue
        step_match = re.match(r"^\s*(?:step\s+)?(\d+)\s*[:.\-]?\s*(.*)|^\s*[-*+]\s+(.*)", line, re.IGNORECASE)
        if step_match:
            step_desc = (step_match.group(2) or step_match.group(3) or "").strip()
            keys = []
            for marker in keyword_markers:
                marker_lower = marker.lower()
                if marker_lower in step_desc.lower():
                    parts = re.split(marker, step_desc, maxsplit=1, flags=re.IGNORECASE)
                    if len(parts) == 2:
                        step_desc = parts[0].strip()
                        keys = [k.strip() for k in re.split(r'[,\n]', parts[1]) if k.strip()]
                        break
            if step_desc:
                current_step = {"step_description": step_desc, "keywords": keys}
                plan.append(current_step)
            else:
                current_step = None
        elif current_step and not current_step["keywords"]:
             is_keyword_line = False
             for marker in keyword_markers:
                 marker_lower = marker.lower()
                 if line.lower().startswith(marker_lower):
                     keys_str = line[len(marker):].strip()
                     current_step["keywords"] = [k.strip() for k in re.split(r'[,\n]', keys_str) if k.strip()]
                     is_keyword_line = True
                     break
    if plan:
        plan = [p for p in plan if p.get("step_description")]
        if plan:
             print("Parsed research plan using simple line-based approach.")
             return plan

    print("All parsing methods failed.")
    fail_msg = "Failed to parse plan structure from LLM response."
    if raw_response:
        fail_msg += f" Raw Response Snippet: '{raw_response[:200]}...'"
    return [{"step_description": fail_msg, "keywords": []}]


# --- Search Provider Functions (unchanged) ---
def search_duckduckgo_provider(keywords, max_results=MAX_SEARCH_RESULTS_PER_ENGINE_STEP, max_retries=2, retry_delay=DDGS_RETRY_DELAY_SECONDS):
    """Performs a search on DuckDuckGo with retry."""
    query = " ".join(keywords)
    urls = []
    if not query: return {"engine": "DuckDuckGo", "urls": [], "success": True, "error": None}
    print(f"  -> Searching DuckDuckGo for: '{query}' (max_results={max_results})")

    for attempt in range(max_retries + 1):
        try:
            with DDGS(timeout=15) as ddgs:
                # DDGS().text returns a generator, convert to list
                results = list(ddgs.text(query, max_results=max_results))
                urls = [r['href'] for r in results if r and 'href' in r]
                print(f"  -> DuckDuckGo Found {len(urls)} URLs for '{query}'")
                return {"engine": "DuckDuckGo", "urls": urls, "success": True, "error": None} # Success
        except Exception as e:
            is_rate_limit = "Ratelimit" in str(e) or "429" in str(e) or "202" in str(e) # 202 used by DDG sometimes
            if is_rate_limit and attempt < max_retries:
                current_delay = retry_delay * (2 ** attempt) # Exponential backoff
                print(f"  -> DuckDuckGo Rate limit detected for '{query}'. Retrying attempt {attempt + 1}/{max_retries} in {current_delay:.1f}s...")
                time.sleep(current_delay)
                continue
            else:
                err_msg = f"Error searching DuckDuckGo for '{query}' (Attempt {attempt+1}): {e}"
                if is_rate_limit:
                    err_msg = f"Max retries reached for rate limit on '{query}'."
                print(f"  -> {err_msg}")
                return {"engine": "DuckDuckGo", "urls": [], "success": False, "error": err_msg}

    return {"engine": "DuckDuckGo", "urls": [], "success": False, "error": "Max retries reached unexpectedly."}

# --- Unified Search Function (unchanged) ---
def perform_web_search(keywords, max_results_per_engine=MAX_SEARCH_RESULTS_PER_ENGINE_STEP):
    """
    Performs searches using configured providers and aggregates unique results.
    """
    all_unique_urls = set()
    search_errors = []

    # Define the search providers to use
    search_providers = [
        search_duckduckgo_provider,
        # search_google_provider, # Uncomment if implemented and API keys are set
    ]

    print(f"  -> Performing search with {len(search_providers)} engine(s) for keywords: {keywords}")

    for i, provider_func in enumerate(search_providers):
        provider_result = provider_func(keywords, max_results=max_results_per_engine)

        if provider_result["success"]:
            found_count = len(provider_result["urls"])
            new_urls = set(provider_result["urls"]) - all_unique_urls
            all_unique_urls.update(provider_result["urls"])
            print(f"  -> {provider_result['engine']} returned {found_count} URLs ({len(new_urls)} new).")
        else:
            print(f"  -> {provider_result['engine']} search failed: {provider_result['error']}")
            search_errors.append(f"{provider_result['engine']}: {provider_result['error']}")

        # Add delay between different search providers if more than one is active
        if len(search_providers) > 1 and i < len(search_providers) - 1:
            print(f"  -> Delaying {INTER_SEARCH_DELAY_SECONDS}s before next search engine...")
            time.sleep(INTER_SEARCH_DELAY_SECONDS)

    return list(all_unique_urls), search_errors

# --- scrape_url (unchanged) ---
def scrape_url(url):
    """
    Scrapes text content from a given URL. Handles common errors.
    Returns a dictionary {'url': url, 'content': text} on success, None on failure.
    """
    log_url = url[:75] + '...' if len(url) > 75 else url
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type and 'text' not in content_type:
            response.close()
            return None

        content_length = response.headers.get('content-length')
        MAX_CONTENT_LENGTH_MB = 10
        if content_length and int(content_length) > MAX_CONTENT_LENGTH_MB * 1024 * 1024:
            response.close()
            return None

        html_content = response.content
        response.close()

        if not content_length and len(html_content) > MAX_CONTENT_LENGTH_MB * 1024 * 1024:
             return None

        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except Exception:
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
            except Exception as parse_err:
                print(f"Failed to parse HTML for {log_url} with both lxml and html.parser: {parse_err}")
                return None

        for element in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "input", "textarea", "select", "img", "figure", "iframe", "video", "audio", "picture", "source", "noscript", "meta", "link", "svg", "canvas", "map", "area"]):
            element.decompose()

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
             body = soup.find('body')
             if body:
                 text = body.get_text(separator='\n', strip=True)
             else:
                 text = soup.get_text(separator='\n', strip=True)

        lines = (line.strip() for line in text.splitlines())
        chunks = (' '.join(phrase.split()) for line in lines for phrase in line.split("  ") if phrase.strip())
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk and len(chunk.split()) > 2)

        MIN_MEANINGFUL_WORDS = 50
        meaningful_word_count = len([word for word in cleaned_text.split() if len(word) > 1])
        if meaningful_word_count < MIN_MEANINGFUL_WORDS:
             return None

        return {'url': url, 'content': cleaned_text}

    except requests.exceptions.Timeout:
        print(f"Timeout error fetching URL {log_url}")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error {e.response.status_code} fetching URL {log_url}: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request Error fetching URL {log_url}: {e}")
    except Exception as e:
        print(f"Error parsing or processing URL {log_url}: {type(e).__name__} - {e}")
    finally:
        if 'response' in locals() and response and hasattr(response, 'close') and callable(response.close):
             try: response.close()
             except Exception: pass

    return None

# --- generate_bibliography_map (unchanged) ---
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
        topic = "Default Topic - No Topic Provided"

    def generate_updates():
        # --- Research state variables (unchanged) ---
        scraped_sources_list = []
        all_found_urls_set = set()
        urls_to_scrape_list = []
        research_plan = []
        accumulated_synthesis = ""
        final_report_markdown = ""
        url_to_index_map = {}
        start_time_total = time.time()

        # --- SSE Helper functions (unchanged) ---
        def send_event(data):
            try:
                payload = json.dumps(data)
                yield f"data: {payload}\n\n"
            except TypeError as e:
                print(f"Error serializing data for SSE: {e}. Data: {data}")
                try:
                    safe_data = {'type': data.get('type', 'error'), 'message': f"Serialization Error: {e}"}
                    payload = json.dumps(safe_data)
                    yield f"data: {payload}\n\n"
                except Exception:
                    yield "data: {\"type\": \"error\", \"message\": \"Internal server error during SSE serialization\"}\n\n"

        def send_progress(message):
            yield from send_event({'type': 'progress', 'message': message})

        def send_error_event(message):
            print(f"ERROR: {message}")
            yield from send_event({'type': 'error', 'message': message})

        try:
            # === Step 1: Generate Research Plan (unchanged) ===
            yield from send_progress(f"Generating research plan for: '{topic}' using {GOOGLE_MODEL_NAME}...")
            plan_prompt = f"""
            Create a detailed, step-by-step research plan with 5-10 distinct steps for the topic: "{topic}"
            Each step should represent a specific question or area of inquiry related to the topic.
            Format the output STRICTLY as a JSON list of objects within a ```json ... ``` block. Each object must have two keys:
            1. "step": A string describing the research step/question.
            2. "keywords": A list of 2-4 relevant keyword strings for searching about this step.

            Example format:
            ```json
            [
              {{"step": "Define the core concept", "keywords": ["term definition", "term explanation"]}},
              {{"step": "Explore historical origins", "keywords": ["history of term", "term origins"]}},
              ... (5-10 steps total)
            ]
            ```

            Ensure the output is ONLY the valid JSON list inside the markdown block. No introductory text, no explanations outside the JSON block.
            Generate 5 to 10 relevant steps for the specific topic: "{topic}".
            """
            try:
                plan_response = call_gemini(plan_prompt)
            except Exception as e:
                 yield from send_error_event(f"Failed to generate research plan from LLM ({GOOGLE_MODEL_NAME}): {e}")
                 return

            research_plan = parse_research_plan(plan_response)
            if not research_plan or not isinstance(research_plan, list) or not all(isinstance(step, dict) and 'step_description' in step and 'keywords' in step for step in research_plan) or \
               (len(research_plan) == 1 and research_plan[0]["step_description"].startswith("Failed")):
                 fail_reason = "Could not parse valid plan structure."
                 if research_plan and isinstance(research_plan, list) and research_plan[0].get("step_description", "").startswith("Failed"):
                      fail_reason = research_plan[0]["step_description"]
                 raw_snippet = f" Raw LLM Response Snippet: '{plan_response[:150]}...'" if plan_response else " (LLM Response was empty)"
                 yield from send_error_event(f"Failed to create or parse a valid research plan. Reason: {fail_reason}.{raw_snippet}")
                 return

            yield from send_progress(f"Generated {len(research_plan)} step plan:")
            for i, step in enumerate(research_plan):
                 yield from send_progress(f"  Step {i+1}: {step['step_description']} (Keywords: {step['keywords']})")


            # === Step 2a: Search and Collect URLs (unchanged) ===
            yield from send_progress("Starting web search to collect URLs...")
            start_search_time = time.time()
            urls_collected_count = 0
            all_urls_from_search_step = set() # Use a set to automatically handle duplicates across steps/engines
            total_search_errors = 0
            total_search_queries = 0

            for i, step in enumerate(research_plan):
                step_desc = step.get('step_description', f'Unnamed Step {i+1}')
                keywords = step.get('keywords', [])

                yield from send_progress(f"Searching - Step {i+1}/{len(research_plan)}: '{step_desc[:70]}{'...' if len(step_desc)>70 else ''}'")
                if not keywords:
                    yield from send_progress("  -> No keywords provided, skipping search for this step.")
                    continue

                total_search_queries += 1
                step_search_results_urls, step_search_errors = perform_web_search(
                    keywords,
                    max_results_per_engine=MAX_SEARCH_RESULTS_PER_ENGINE_STEP
                )

                if step_search_errors:
                    total_search_errors += len(step_search_errors)

                before_count = len(all_urls_from_search_step)
                all_urls_from_search_step.update(step_search_results_urls)
                new_urls_this_step = len(all_urls_from_search_step) - before_count
                yield from send_progress(f"  -> Step search added {new_urls_this_step} new unique URLs (Total unique: {len(all_urls_from_search_step)}).")


                # Add delay *between* different keyword searches (steps)
                if i < len(research_plan) - 1:
                    time.sleep(INTER_SEARCH_DELAY_SECONDS)

            yield from send_progress(f"Search phase completed in {time.time() - start_search_time:.2f}s.")
            yield from send_progress(f"Found {len(all_urls_from_search_step)} total unique URL results across {total_search_queries} search queries ({total_search_errors} engine errors encountered).")

            # --- Filtering (unchanged) ---
            yield from send_progress(f"Processing {len(all_urls_from_search_step)} unique URLs for filtering...")
            urls_to_scrape_list = [] # Reset list
            all_found_urls_set = set() # Reset set tracking *selected* URLs

            sorted_unique_urls = sorted(list(all_urls_from_search_step))

            for url in sorted_unique_urls:
                 if urls_collected_count >= MAX_TOTAL_URLS_TO_SCRAPE:
                      yield from send_progress(f"  -> Reached URL collection limit ({MAX_TOTAL_URLS_TO_SCRAPE}).")
                      break
                 is_file = url.lower().split('?')[0].split('#')[0].endswith(('.pdf', '.jpg', '.png', '.gif', '.zip', '.mp4', '.mp3', '.docx', '.xlsx', '.pptx', '.webp', '.svg', '.xml', '.css', '.js', '.jpeg', '.doc', '.xls', '.ppt', '.txt', '.exe', '.dmg', '.iso', '.rar'))
                 is_mailto = url.lower().startswith('mailto:')
                 is_javascript = url.lower().startswith('javascript:')
                 is_ftp = url.lower().startswith('ftp:')
                 is_tel = url.lower().startswith('tel:')
                 is_valid_http = url.startswith(('http://', 'https://'))

                 if is_valid_http and not is_file and not is_mailto and not is_javascript and not is_ftp and not is_tel:
                      urls_to_scrape_list.append(url)
                      all_found_urls_set.add(url) # Track added URLs
                      urls_collected_count += 1

            yield from send_progress(f"Selected {len(urls_to_scrape_list)} valid, filtered URLs for scraping (limit: {MAX_TOTAL_URLS_TO_SCRAPE}).")

            if not urls_to_scrape_list:
                 yield from send_error_event("No suitable URLs found to scrape after searching and filtering.")
                 return

            # === Step 2b: Scrape URLs Concurrently (unchanged) ===
            yield from send_progress(f"Starting concurrent scraping of {len(urls_to_scrape_list)} URLs (Max workers: {MAX_WORKERS})...")
            start_scrape_time = time.time()
            total_scraped_successfully = 0
            processed_scrape_count = 0
            scraped_sources_list = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_url = {executor.submit(scrape_url, url): url for url in urls_to_scrape_list}
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    processed_scrape_count += 1
                    try:
                        result_dict = future.result()
                        if result_dict and isinstance(result_dict, dict) and result_dict.get('content'):
                            scraped_sources_list.append(result_dict)
                            total_scraped_successfully += 1
                    except Exception as exc:
                        yield from send_progress(f"    -> Unexpected Error processing scrape result for {url[:60]}...: {exc}")

                    update_interval = max(1, len(urls_to_scrape_list) // 10)
                    if processed_scrape_count % update_interval == 0 or processed_scrape_count == len(urls_to_scrape_list):
                          progress_perc = (processed_scrape_count * 100) // len(urls_to_scrape_list)
                          yield from send_progress(f"  -> Scraping progress: {processed_scrape_count}/{len(urls_to_scrape_list)} ({progress_perc}%). Success: {total_scraped_successfully}")

            duration = time.time() - start_scrape_time
            yield from send_progress(f"Finished scraping. Successfully scraped {total_scraped_successfully}/{len(urls_to_scrape_list)} URLs in {duration:.2f} seconds.")
            if total_scraped_successfully < len(urls_to_scrape_list):
                 yield from send_progress(f"  -> Note: {len(urls_to_scrape_list) - total_scraped_successfully} URLs failed to scrape.")

            if not scraped_sources_list:
                yield from send_error_event("Failed to scrape any content successfully.")
                return

            scraped_url_map = {item['url']: item for item in scraped_sources_list}
            ordered_scraped_list = [scraped_url_map[url] for url in urls_to_scrape_list if url in scraped_url_map]
            scraped_sources_list = ordered_scraped_list

            # === Bibliography Map (unchanged) ===
            url_to_index_map, bibliography_prompt_list = generate_bibliography_map(scraped_sources_list)

            # === Step 3: Synthesize with Citations (Streaming) (unchanged) ===
            yield from send_progress(f"Synthesizing relevant information using {GOOGLE_MODEL_NAME}...")
            yield from send_event({'type': 'stream_start', 'target': 'synthesis'})

            TARGET_MAX_CONTEXT_TOKENS = 750000
            CHARS_PER_TOKEN_ESTIMATE = 4
            MAX_CONTEXT_CHARS = int(TARGET_MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN_ESTIMATE * 0.9)

            context_for_llm_structured = []
            current_chars = 0
            sources_included_count = 0
            original_source_count = len(scraped_sources_list)

            for source in scraped_sources_list:
                 source_len = len(source.get('url', '')) + len(source.get('content', '')) + 50
                 if current_chars + source_len <= MAX_CONTEXT_CHARS:
                     context_for_llm_structured.append(source)
                     current_chars += source_len
                     sources_included_count += 1
                 else:
                     yield from send_progress(f"  -> Warning: Estimated context limit reached. Using first {sources_included_count}/{original_source_count} sources for synthesis.")
                     break

            estimated_chars = sum(len(item.get('content', '')) for item in context_for_llm_structured)
            estimated_tokens = estimated_chars / CHARS_PER_TOKEN_ESTIMATE
            yield from send_progress(f"  -> Synthesis using {len(context_for_llm_structured)} sources (~{estimated_chars // 1000}k chars / ~{estimated_tokens // 1000}k tokens).")

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
            2. For each step, carefully review ALL the scraped source content to find relevant information.
            3. Extract and synthesize the most pertinent facts, findings, or data related *specifically* to that research step.
            4. **MANDATORY**: Immediately after stating **any** piece of information derived from a source, cite the source using the exact format: `[Source URL: <full_url_here>]`. Cite every distinct piece of information.
            5. If no relevant information is found for a step, state: "No specific information found for this step in the provided sources."
            6. Structure your output using Markdown. Use a heading (`### Step X: <Step Description>`) for each plan step.
            7. Output ONLY the synthesized information with inline URL citations, structured by plan step. Do NOT include introduction, conclusion, summary, or bibliography. Separate steps clearly (e.g., using `---`).
            """
            accumulated_synthesis = ""
            synthesis_stream_error = None
            try:
                stream_generator = stream_gemini(synthesis_prompt)
                for result in stream_generator:
                    if result['type'] == 'chunk':
                        yield from send_event({'type': 'llm_chunk', 'content': result['content'], 'target': 'synthesis'})
                        accumulated_synthesis += result['content']
                    elif result['type'] == 'stream_error':
                        synthesis_stream_error = result['message']
                        yield from send_error_event(f"LLM stream error during synthesis ({GOOGLE_MODEL_NAME}): {synthesis_stream_error}")
                        if "API key" in synthesis_stream_error or "quota" in synthesis_stream_error.lower(): return
                        # Allow continuing to report generation even if synthesis had minor issues
                        break
                    elif result['type'] == 'stream_warning':
                         yield from send_progress(f"LLM Stream Warning (Synthesis): {result['message']}")
                    elif result['type'] == 'stream_end':
                         yield from send_progress(f"Synthesis stream finished.")
                         break
                # Dont return here if error was just a warning or non-fatal

            except Exception as e:
                 yield from send_error_event(f"Fatal error processing LLM synthesis stream: {e}")
                 traceback.print_exc(); return # Fatal error

            yield from send_progress(f"Synthesis generation completed.")
            if not accumulated_synthesis.strip() and not synthesis_stream_error: # Check only if no error stopped it
                 yield from send_progress("Warning: Synthesis resulted in empty content. Proceeding to report generation.")
                 # Don't return, allow report generation to attempt


            # === Step 4: Generate Final Report (Streaming) (unchanged) ===
            yield from send_progress(f"Generating final report using {GOOGLE_MODEL_NAME}...")
            yield from send_event({'type': 'stream_start', 'target': 'report'})

            report_components_size = len(topic) + len(json.dumps(research_plan)) + len(accumulated_synthesis) + len(bibliography_prompt_list)
            report_estimated_tokens = report_components_size / CHARS_PER_TOKEN_ESTIMATE

            if report_estimated_tokens > TARGET_MAX_CONTEXT_TOKENS * 0.95:
                 yield from send_progress(f"  -> Warning: Inputs for final report prompt large. Synthesis might be truncated.")
                 available_chars_for_synthesis = MAX_CONTEXT_CHARS - (len(topic) + len(json.dumps(research_plan)) + len(bibliography_prompt_list) + 2000)
                 if available_chars_for_synthesis < 0: available_chars_for_synthesis = 0
                 truncated_synthesis = accumulated_synthesis[:available_chars_for_synthesis] + "\n\n... [Synthesis truncated]"
            else:
                 truncated_synthesis = accumulated_synthesis

            report_prompt = f"""
            Create a comprehensive Markdown research report on the topic: "{topic}".

            Inputs:
            1. Original Research Plan ({len(research_plan)} steps):
               ```json
               {json.dumps(research_plan, indent=2)}
               ```
            2. Synthesized Information with Raw URL Citations:
               ```markdown
               {truncated_synthesis if truncated_synthesis.strip() else "No synthesized information was generated."}
               ```
            3. Bibliography Map (URL to Reference Number - {len(url_to_index_map)} sources):
               ```
               {bibliography_prompt_list if bibliography_prompt_list else "No sources available for bibliography."}
               ```

            Instructions:
            1. Write a final Markdown report with sections: `# Research Report: {topic}`, `## Introduction`, `## Findings`, `## Conclusion`, `## Bibliography`.
            2. **Introduction**: Introduce "{topic}", state the report's purpose, outline research scope (plan steps).
            3. **Findings**: Organize by plan step (`### Step X: <Description>`). Integrate synthesized info for each step. If synthesis was empty, state this clearly.
            4. **CRITICAL CITATION**: Replace EVERY `[Source URL: <full_url_here>]` with its corresponding Markdown footnote `[^N]`. Use the Bibliography Map for the number N. If a URL isn't in the map, OMIT the citation marker.
            5. **Conclusion**: Summarize key findings (or lack thereof if synthesis failed). Mention limitations (e.g., empty synthesis, source count, potential scraping issues). Suggest further research if appropriate.
            6. **Bibliography**: List sources from the map numerically using Markdown footnote definitions: `[^N]: <full_url_here>`. If no sources, state "No sources cited."
            7. Output ONLY the complete Markdown report. No extra commentary.
            """
            final_report_markdown = ""
            report_stream_error = None
            try:
                stream_generator = stream_gemini(report_prompt)
                for result in stream_generator:
                    if result['type'] == 'chunk':
                        yield from send_event({'type': 'llm_chunk', 'content': result['content'], 'target': 'report'})
                        final_report_markdown += result['content']
                    elif result['type'] == 'stream_error':
                        report_stream_error = result['message']
                        yield from send_error_event(f"LLM stream error during report generation ({GOOGLE_MODEL_NAME}): {report_stream_error}")
                        if "API key" in report_stream_error or "quota" in report_stream_error.lower(): return # Fatal
                        break # Non-fatal, break loop
                    elif result['type'] == 'stream_warning':
                         yield from send_progress(f"LLM Stream Warning (Report): {result['message']}")
                    elif result['type'] == 'stream_end':
                         yield from send_progress(f"Report stream finished.")
                         break
                # Continue even if non-fatal stream error occurred

            except Exception as e:
                 yield from send_error_event(f"Fatal error processing LLM report stream: {e}")
                 traceback.print_exc(); return # Fatal

            yield from send_progress(f"Report generation completed.")
            if not final_report_markdown.strip():
                 # Check if we expected content based on input
                 if accumulated_synthesis.strip() or bibliography_prompt_list:
                     yield from send_error_event("Report generation resulted in empty content despite having input synthesis/sources.")
                 else:
                     yield from send_progress("Report generation resulted in empty content (as expected due to lack of synthesis/sources).")
                 # Don't return here, send the empty report structure

            # --- Final Processing and Completion (unchanged) ---
            yield from send_progress("Processing final report for display...")

            try:
                report_html = md_lib.markdown(
                    final_report_markdown if final_report_markdown.strip() else "# Research Report\n\n*No content generated.*",
                    extensions=['footnotes', 'fenced_code', 'tables', 'nl2br', 'attr_list', 'md_in_html']
                )
            except Exception as md_err:
                 yield from send_error_event(f"Failed to convert final Markdown report to HTML: {md_err}")
                 from html import escape
                 report_html = f"<h3>Markdown Conversion Error</h3><p>Error: {escape(str(md_err))}. Raw Markdown:</p><hr><pre style='white-space: pre-wrap; word-wrap: break-word;'>{escape(final_report_markdown)}</pre>"

            preview_limit = 3000
            raw_data_preview_list = []
            current_preview_len = 0
            for src in scraped_sources_list:
                src_dump = json.dumps(src, indent=2, ensure_ascii=False)
                if current_preview_len + len(src_dump) < preview_limit:
                    raw_data_preview_list.append(src_dump)
                    current_preview_len += len(src_dump)
                else: break
            raw_data_preview = "[\n" + ",\n".join(raw_data_preview_list) + "\n]"
            if len(scraped_sources_list) > len(raw_data_preview_list):
                 raw_data_preview += f"\n... ({len(raw_data_preview_list)}/{len(scraped_sources_list)} sources shown)"
            elif not scraped_sources_list: raw_data_preview = "None (No sources scraped)"

            # === DEBUG PRINT ADDED HERE ===
            print(f"DEBUG [Stream End]: DOCX_CONVERSION_AVAILABLE is currently: {DOCX_CONVERSION_AVAILABLE}")
            # ============================

            final_data = {
                'type': 'complete',
                'report_html': report_html,
                'report_markdown': final_report_markdown,
                'raw_scraped_data_preview': raw_data_preview,
                'docx_available': DOCX_CONVERSION_AVAILABLE # Use the flag's value at this point
            }
            yield from send_event(final_data)
            end_time_total = time.time()
            yield from send_progress(f"Research process completed successfully in {end_time_total - start_time_total:.2f} seconds.")

        except Exception as e:
            print(f"An unexpected error occurred during stream generation: {e}")
            traceback.print_exc() # Print full traceback
            error_msg = f"Unexpected server error: {type(e).__name__} - {e}"
            yield from send_error_event(error_msg)
        finally:
             yield from send_event({'type': 'stream_terminated'})

    headers = {
        'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no', 'Connection': 'keep-alive'
    }
    return Response(stream_with_context(generate_updates()), headers=headers)


# --- DOCX Download Route - REVISED to use html2docx ---
@app.route('/download_docx', methods=['POST'])
def download_docx():
    """Converts the received Markdown report to DOCX via HTML and sends it."""
    # Re-check availability just in case, though it should be consistent if set early
    if not DOCX_CONVERSION_AVAILABLE:
        print("DEBUG [/download_docx]: Route called but DOCX_CONVERSION_AVAILABLE is False.")
        return jsonify({"success": False, "message": "DOCX download is disabled: 'html2docx' library check failed at app start."}), 400

    print("DEBUG [/download_docx]: Route called and DOCX_CONVERSION_AVAILABLE is True.")
    markdown_content = request.form.get('markdown_report')
    topic = request.form.get('topic', 'Research_Report')

    if not markdown_content:
        print("ERROR [/download_docx]: No Markdown content received in POST request.")
        return jsonify({"success": False, "message": "Error: No Markdown content received."}), 400

    try:
        print("DEBUG [/download_docx]: Converting Markdown to HTML...")
        # 1. Convert Markdown to HTML first
        report_html = md_lib.markdown(
            markdown_content,
            extensions=['footnotes', 'fenced_code', 'tables', 'nl2br', 'attr_list', 'md_in_html']
        )
        print("DEBUG [/download_docx]: Markdown to HTML conversion done.")

        # 2. Convert HTML string to DOCX using html2docx
        print("DEBUG [/download_docx]: Initializing Html2Docx parser...")
        parser = Html2Docx()
        buffer = BytesIO()
        print("DEBUG [/download_docx]: Parsing HTML string...")
        parser.parse_html_string(report_html)
        print("DEBUG [/download_docx]: Saving DOCX to buffer...")
        parser.docx.save(buffer) # Save the python-docx object to the buffer
        buffer.seek(0) # Rewind the buffer
        print("DEBUG [/download_docx]: DOCX conversion complete, preparing to send file.")

        # 3. Send the buffer as a file download
        safe_filename_topic = re.sub(r'[^\w\s-]', '', topic).strip()
        safe_filename_topic = re.sub(r'[-\s]+', '_', safe_filename_topic)
        filename_base = f"{safe_filename_topic}_Research_Report"
        filename = f"{filename_base[:200]}.docx"

        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

    except Exception as e:
        print(f"ERROR [/download_docx]: Error converting Markdown to DOCX using html2docx:")
        traceback.print_exc()
        msg = f"An error occurred during DOCX conversion: {e}"
        return jsonify({"success": False, "message": msg}), 500


# --- Run the App ---
if __name__ == '__main__':
    # Ensure libraries are installed:
    # pip install Flask python-dotenv google-generativeai duckduckgo-search beautifulsoup4 requests lxml Markdown html2docx flask[async]
    # Set GOOGLE_API_KEY in your .env file
    print(f"Using Google Model: {GOOGLE_MODEL_NAME}")
    # This print now uses the value determined by the immediate import check
    print(f"DOCX conversion available (html2docx): {DOCX_CONVERSION_AVAILABLE}")
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True) # Consider threaded=False for simpler debugging if needed