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

# --- NEW: Import pypandoc ---
try:
    import pypandoc
    PANDOC_AVAILABLE = True
except ImportError:
    print("WARNING: pypandoc library not found. DOCX download will be disabled. Install with 'pip install pypandoc'")
    PANDOC_AVAILABLE = False
except OSError:
    # This catches if pypandoc is installed but the pandoc executable isn't found
    print("WARNING: Pandoc executable not found in system PATH. DOCX download will be disabled. Ensure Pandoc is installed and in PATH: https://pandoc.org/installing.html")
    PANDOC_AVAILABLE = False


# --- Configuration ---
load_dotenv()
# --- CHANGED: Use Google API Key and Model Name ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Use a valid Google model name (e.g., gemini-1.5-pro-latest, gemini-1.5-flash-latest, gemini-pro)
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-pro-latest")
# --- REMOVED: OpenRouter Base URL ---
# OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

MAX_SEARCH_RESULTS_PER_STEP = 10
MAX_TOTAL_URLS_TO_SCRAPE = 100
MAX_WORKERS = 10
REQUEST_TIMEOUT = 12
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36" # Updated UA string
PICO_CSS_CDN = "https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css"

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

# --- REFACTORED: call_gemini to use Google API ---
def call_gemini(prompt, system_prompt=None, max_retries=3, delay=5):
    """Calls the specified Google Gemini model with retry logic."""
    # Combine system prompt and user prompt if system prompt exists
    # Google's API often takes a single prompt string or specific system_instruction
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    generation_config = genai.types.GenerationConfig(
        # candidate_count=1, # Default is 1
        # stop_sequences=["..."], # Optional stop sequences
        # max_output_tokens=8192, # Optional max tokens for the response
        temperature=0.6,
        # top_p=0.9, # Optional nucleus sampling
        # top_k=40   # Optional top-k sampling
    )

    # Define safety settings (Optional, example: block less)
    # safety_settings = {
    #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    # }

    for attempt in range(max_retries):
        try:
            # Initialize the model for each attempt (or can be done once outside)
            model = genai.GenerativeModel(
                GOOGLE_MODEL_NAME,
                generation_config=generation_config,
                # safety_settings=safety_settings, # Apply safety settings if defined
                system_instruction=system_prompt if system_prompt else None # Use dedicated system instruction param if available and preferred
            )

            # Use the combined prompt if system_instruction isn't used or needs supplementing
            effective_prompt = prompt if system_prompt else full_prompt # Adjust based on how system_instruction is handled

            response = model.generate_content(effective_prompt) # Send the prompt

            # Check for blocking or empty response *before* accessing .text
            if not response.candidates:
                 block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
                 safety_ratings = getattr(response.prompt_feedback, 'safety_ratings', [])
                 raise ValueError(f"API response blocked or empty. Block Reason: {block_reason}. Safety Ratings: {safety_ratings}")

            # Check if the first candidate has content
            if not response.candidates[0].content or not response.candidates[0].content.parts:
                 finish_reason = getattr(response.candidates[0], 'finish_reason', 'Unknown')
                 # FinishReason often indicates issues like length, safety, recitation etc.
                 raise ValueError(f"API response candidate has no content parts. Finish Reason: {finish_reason}")


            # Access the text content
            response_content = response.text.strip()
            if not response_content:
                 raise ValueError("API response generated empty text content.")

            return response_content

        except Exception as e:
            # Catch potential Google API specific errors if needed, e.g., google.api_core.exceptions.GoogleAPIError
            print(f"Error calling Google Gemini API (Attempt {attempt + 1}/{max_retries}): {e}")
            # Check if the error suggests the prompt was too long (this might be model/API version specific)
            if "prompt" in str(e).lower() and ("too long" in str(e).lower() or "size" in str(e).lower()):
                print("Prompt might be too long for the model's context window.")
                # Optionally shorten the prompt here if feasible, or just fail
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Failing LLM call.")
                raise # Re-raise the last exception

# --- REFACTORED: stream_gemini to use Google API ---
def stream_gemini(prompt, system_prompt=None):
    """
    Calls the Google Gemini model with streaming enabled and yields content chunks.
    """
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    generation_config = genai.types.GenerationConfig(
        temperature=0.6,
        # max_output_tokens=8192 # Consider setting if needed
    )

    # Define safety settings (Optional, example: block none for less interruption, use with caution)
    # safety_settings = {
    #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    # }


    try:
        model = genai.GenerativeModel(
            GOOGLE_MODEL_NAME,
            generation_config=generation_config,
            # safety_settings=safety_settings, # Apply safety settings if defined
            system_instruction=system_prompt if system_prompt else None
        )

        effective_prompt = prompt if system_prompt else full_prompt

        # Start the streaming generation
        stream = model.generate_content(effective_prompt, stream=True)

        complete_response_text = ""
        finish_reason = None # Google's stream might not yield this per chunk easily

        for chunk in stream:
             # Check for blocking reasons within the chunk's prompt feedback
             if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                  block_reason_str = str(chunk.prompt_feedback.block_reason)
                  print(f"Warning: LLM stream blocked during generation. Reason: {block_reason_str}")
                  yield {'type': 'stream_warning', 'message': f'LLM stream may have been blocked due to safety filters (Reason: {block_reason_str}). Output may be incomplete.'}
                  # Depending on severity, you might want to break or return here
                  # break # Stop processing if blocked

             # Check if the chunk has text content
             if hasattr(chunk, 'text') and chunk.text:
                  chunk_text = chunk.text
                  complete_response_text += chunk_text
                  yield {'type': 'chunk', 'content': chunk_text}
             else:
                  # Log if a chunk seems empty but no block reason was given
                  # print(f"Debug: Received chunk with no text: {chunk}") # Uncomment for debugging
                  pass

        # After the loop, the stream is finished.
        # We can try to get the final finish reason from the resolved stream response,
        # though accessing it directly after iteration might require specific handling
        # or inspection of the 'stream' object's final state if the library supports it.
        # For now, we rely on detecting blocks during iteration.
        # Let's check the complete response length against potential limits (heuristic)
        # This is not a perfect replacement for the explicit 'length' finish_reason.
        # We might need to inspect `stream.candidates[0].finish_reason` if the stream object allows it after iteration.

        # Example (conceptual - library might behave differently):
        # try:
        #     final_candidate = stream.candidates[0] # Accessing after iteration might raise error
        #     finish_reason = str(final_candidate.finish_reason) if final_candidate else 'Unknown'
        # except Exception:
        #     finish_reason = 'Unknown (Iteration Complete)'

        # Simplified end event:
        yield {'type': 'stream_end', 'finish_reason': finish_reason or 'IterationComplete'}

    except Exception as e:
        # Handle potential API errors, including configuration or quota issues
        print(f"Error during Google Gemini stream: {e}")
        error_message = f"LLM stream error: {e}"
        # Check if error suggests context length issue
        if "prompt" in str(e).lower() and ("too long" in str(e).lower() or "size" in str(e).lower()):
             error_message = f"LLM stream error: Prompt likely too long for the model's context window. ({e})"
        elif "resource has been exhausted" in str(e).lower():
             error_message = f"LLM stream error: Quota likely exceeded. ({e})"

        yield {'type': 'stream_error', 'message': error_message}


# parse_research_plan - No changes needed from previous version
def parse_research_plan(llm_response):
    # --- (Keep the robust parser from previous step) ---
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


# search_duckduckgo - No changes needed
def search_duckduckgo(keywords, max_results=MAX_SEARCH_RESULTS_PER_STEP):
    """Performs a search on DuckDuckGo using the library."""
    query = " ".join(keywords)
    urls = []
    if not query: return [] # Handle empty keywords
    print(f"  -> DDGS Searching for: '{query}' (max_results={max_results})")
    try:
        with DDGS(timeout=15) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            urls = [r['href'] for r in results if r and 'href' in r]
            print(f"  -> DDGS Found {len(urls)} URLs for '{query}'")
    except Exception as e:
        print(f"Error searching DuckDuckGo for '{query}': {e}")
    return urls

# scrape_url - No changes needed from previous version
def scrape_url(url):
    """
    Scrapes text content from a given URL.
    Returns a dictionary {'url': url, 'content': text} on success, None on failure.
    """
    log_url = url[:75] + '...' if len(url) > 75 else url
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            print(f"Skipping non-HTML content: {log_url} (Type: {content_type})")
            response.close()
            return None

        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 10 * 1024 * 1024: # 10MB limit
            print(f"Skipping large file (>10MB): {log_url}")
            response.close()
            return None

        html_content = response.content # Read content
        response.close() # Close connection

        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except Exception:
            soup = BeautifulSoup(html_content, 'html.parser')

        for element in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "input", "textarea", "select", "img", "figure", "iframe", "video", "audio", "picture", "source", "noscript", "meta", "link"]):
            element.decompose()

        main_content = soup.find('main') or \
                       soup.find('article') or \
                       soup.find('div', attrs={'role': 'main'}) or \
                       soup.find('div', id='content') or \
                       soup.find('div', class_=re.compile(r'\b(content|main|post|entry)\b', re.I)) or \
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
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk and len(chunk.split()) > 3)

        meaningful_word_count = len([word for word in cleaned_text.split() if len(word) > 2])
        if meaningful_word_count < 75:
             # print(f"Skipping due to low meaningful content ({meaningful_word_count} words): {log_url}") # Less verbose
             return None

        # print(f"Successfully scraped: {log_url} ({len(cleaned_text)} chars)") # Less verbose
        return {'url': url, 'content': cleaned_text}

    except requests.exceptions.Timeout:
        print(f"Timeout error fetching URL {log_url}")
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            print(f"HTTP Error {e.response.status_code} fetching URL {log_url}: {e}")
        else:
            print(f"Request Error fetching URL {log_url}: {e}")
    except Exception as e:
        print(f"Error parsing or processing URL {log_url}: {e}")
    finally:
        if 'response' in locals() and response and not response.raw.closed:
            response.close()

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
        topic = "Default Topic"

    def generate_updates():
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
                error_payload = json.dumps({'type': 'error', 'message': f"Internal server error: Could not serialize data - {e}"})
                yield f"data: {error_payload}\n\n"

        def send_progress(message):
            yield from send_event({'type': 'progress', 'message': message})

        def send_error_event(message):
            """Sends an error event via SSE."""
            print(f"ERROR: {message}")
            yield from send_event({'type': 'error', 'message': message})

        try:
            # === Step 1: Generate Research Plan ===
            yield from send_progress(f"Generating research plan for: '{topic}' using {GOOGLE_MODEL_NAME}...")
            # Prompt remains the same, asking for JSON structure
            plan_prompt = f"""
            Create a detailed, step-by-step research plan with 10-15 distinct steps for the topic: "{topic}"
            Each step should represent a specific question or area of inquiry related to the topic.
            Format the output STRICTLY as a JSON list of objects. Each object must have two keys:
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

            Ensure the output is ONLY the valid JSON list, enclosed in ```json ... ``` markdown block. No introductory text, no explanations outside the JSON block.
            Generate 10 to 15 relevant steps for the specific topic: "{topic}".
            """
            try:
                # Use the refactored call_gemini
                plan_response = call_gemini(plan_prompt) # No system prompt needed here as it's in the main prompt
            except Exception as e:
                 yield from send_error_event(f"Failed to generate research plan from LLM ({GOOGLE_MODEL_NAME}): {e}")
                 return # Terminate generator

            research_plan = parse_research_plan(plan_response)

            if not research_plan or not isinstance(research_plan, list) or \
               (len(research_plan) == 1 and research_plan[0]["step_description"].startswith("Failed")):
                 fail_reason = research_plan[0]["step_description"] if (research_plan and isinstance(research_plan, list) and research_plan[0].get("step_description")) else "Could not parse plan from LLM response."
                 raw_snippet = f" Raw Response Snippet: '{plan_response[:150]}...'" if plan_response else ""
                 yield from send_error_event(f"Failed to create or parse a valid research plan. {fail_reason}{raw_snippet}")
                 return # Terminate generator

            yield from send_progress(f"Generated {len(research_plan)} step plan:")
            log_limit = 5
            for i, step in enumerate(research_plan[:log_limit]):
                 yield from send_progress(f"  {i+1}. {step['step_description']} (Keywords: {', '.join(step.get('keywords',[]))[:50]}...)")
            if len(research_plan) > log_limit:
                 yield from send_progress(f"  ... and {len(research_plan) - log_limit} more steps.")

            # === Step 2a: Search and Collect URLs ===
            yield from send_progress("Starting web search to collect URLs...")
            start_search_time = time.time()
            urls_collected_count = 0
            all_urls_from_search = []
            for i, step in enumerate(research_plan):
                step_desc = step.get('step_description', f'Unnamed Step {i+1}')
                keywords = step.get('keywords', [])

                yield from send_progress(f"Searching - Step {i+1}/{len(research_plan)}: '{step_desc}'")
                if not keywords:
                    yield from send_progress("  -> No keywords provided, skipping search for this step.")
                    continue

                step_search_results = search_duckduckgo(keywords, MAX_SEARCH_RESULTS_PER_STEP)
                all_urls_from_search.extend(step_search_results)
                time.sleep(0.1) # Small delay between searches

            yield from send_progress(f"Search phase completed in {time.time() - start_search_time:.2f}s. Found {len(all_urls_from_search)} total URLs initially.")

            unique_urls = list(dict.fromkeys(all_urls_from_search))
            yield from send_progress(f"Processing {len(unique_urls)} unique URLs...")

            for url in unique_urls:
                 if not url: continue
                 if urls_collected_count >= MAX_TOTAL_URLS_TO_SCRAPE:
                      yield from send_progress(f"  -> Reached URL collection limit ({MAX_TOTAL_URLS_TO_SCRAPE}).")
                      break
                 if url in all_found_urls_set: continue

                 # Basic URL filtering
                 is_file = url.lower().split('?')[0].split('#')[0].endswith(('.pdf', '.jpg', '.png', '.gif', '.zip', '.mp4', '.mp3', '.docx', '.xlsx', '.pptx', '.webp', '.svg', '.xml', '.css', '.js', '.jpeg', '.doc', '.xls', '.ppt', '.txt'))
                 is_mailto = url.lower().startswith('mailto:')
                 is_javascript = url.lower().startswith('javascript:')
                 is_ftp = url.lower().startswith('ftp:')
                 is_tel = url.lower().startswith('tel:')
                 is_valid_http = url.startswith(('http://', 'https://'))
                 # Add common non-content domains if needed (e.g., youtube, facebook, twitter)
                 # is_social_media = any(domain in url for domain in ['youtube.com', 'facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com'])

                 if is_valid_http and not is_file and not is_mailto and not is_javascript and not is_ftp and not is_tel: # and not is_social_media:
                      urls_to_scrape_list.append(url)
                      all_found_urls_set.add(url)
                      urls_collected_count += 1
                 #else:
                      # yield from send_progress(f"  -> Skipping filtered URL: {url[:70]}...") # Can be verbose
                 all_found_urls_set.add(url) # Track all encountered URLs

            yield from send_progress(f"Selected {len(urls_to_scrape_list)} valid, unique URLs for scraping (limit: {MAX_TOTAL_URLS_TO_SCRAPE}).")

            if not urls_to_scrape_list:
                 yield from send_error_event("No suitable URLs found to scrape after searching and filtering.")
                 return

            # === Step 2b: Scrape URLs Concurrently ===
            yield from send_progress(f"Starting concurrent scraping of {len(urls_to_scrape_list)} URLs using up to {MAX_WORKERS} workers...")
            start_scrape_time = time.time()
            total_scraped_successfully = 0
            processed_scrape_count = 0
            scraped_sources_list = [] # Re-initialize here

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
                        # else: # Don't log every failed scrape, scrape_url logs errors
                        #     yield from send_progress(f"    -> Failed or empty scrape for: {url[:60]}...")
                    except Exception as exc:
                        yield from send_progress(f"    -> Thread Error processing scrape result for {url[:60]}...: {exc}")
                    finally:
                         # Update progress less frequently to avoid flooding
                         if processed_scrape_count % 10 == 0 or processed_scrape_count == len(urls_to_scrape_list):
                              progress_perc = (processed_scrape_count * 100) // len(urls_to_scrape_list)
                              yield from send_progress(f"  -> Scraping progress: {processed_scrape_count}/{len(urls_to_scrape_list)} ({progress_perc}% complete). Success: {total_scraped_successfully}")

            duration = time.time() - start_scrape_time
            yield from send_progress(f"Finished scraping. Successfully scraped {total_scraped_successfully}/{len(urls_to_scrape_list)} URLs in {duration:.2f} seconds.")

            if not scraped_sources_list:
                yield from send_error_event("Failed to scrape any content successfully from the selected URLs.")
                return

            # Re-order scraped list to match the original scrape order (optional, but can be helpful)
            scraped_url_map = {item['url']: item for item in scraped_sources_list}
            ordered_scraped_list = [scraped_url_map[url] for url in urls_to_scrape_list if url in scraped_url_map]
            scraped_sources_list = ordered_scraped_list # Use the ordered list

            # === Bibliography Map ===
            url_to_index_map, bibliography_prompt_list = generate_bibliography_map(scraped_sources_list)

            # === Step 3: Synthesize with Citations (Streaming) ===
            yield from send_progress(f"Synthesizing relevant information using {GOOGLE_MODEL_NAME} (AI Generating...)")
            yield from send_event({'type': 'stream_start', 'target': 'synthesis'})

            # Prepare context for LLM
            # Ensure context fits within model limits. Truncate content if necessary.
            # Note: Gemini 1.5 Pro has a large context window, but cost and processing time increase.
            MAX_CONTEXT_CHARS = 900000 # Example limit (adjust based on model and expected input size)
            context_for_llm_structured = []
            current_chars = 0
            sources_included_count = 0
            original_source_count = len(scraped_sources_list)

            for source in scraped_sources_list:
                source_len = len(source.get('content', '')) + len(source.get('url', '')) + 50 # Estimate JSON overhead
                if current_chars + source_len <= MAX_CONTEXT_CHARS:
                    context_for_llm_structured.append(source)
                    current_chars += source_len
                    sources_included_count += 1
                else:
                    yield from send_progress(f"  -> Warning: Context limit ({MAX_CONTEXT_CHARS} chars) reached. Truncating sources for synthesis. Included {sources_included_count}/{original_source_count}.")
                    break # Stop adding sources

            estimated_chars = sum(len(item.get('content', '')) for item in context_for_llm_structured)
            yield from send_progress(f"  -> Synthesis context size: ~{estimated_chars // 1000}k chars from {len(context_for_llm_structured)} sources.")

            if estimated_chars > 500000 and GOOGLE_MODEL_NAME == "gemini-pro": # gemini-pro has smaller context
                 yield from send_progress("  -> Warning: Context size might be large for gemini-pro, consider gemini-1.5-pro if available.")


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
            synthesis_finish_reason = None
            try:
                # Use the refactored stream_gemini
                stream_generator = stream_gemini(synthesis_prompt)
                for result in stream_generator:
                    if result['type'] == 'chunk':
                        yield from send_event({'type': 'llm_chunk', 'content': result['content'], 'target': 'synthesis'})
                        accumulated_synthesis += result['content']
                    elif result['type'] == 'stream_error':
                        synthesis_stream_error = result['message']
                        yield from send_error_event(f"LLM stream error during synthesis ({GOOGLE_MODEL_NAME}): {synthesis_stream_error}")
                        # Decide whether to terminate based on error type (e.g., quota vs temporary issue)
                        return # Terminate for now on any stream error
                    elif result['type'] == 'stream_warning':
                         yield from send_progress(f"LLM Stream Warning (Synthesis): {result['message']}")
                         # Continue processing despite warnings like safety blocks, but be aware output might be affected
                    elif result['type'] == 'stream_end':
                         synthesis_finish_reason = result.get('finish_reason')
                         # Google API stream_end might not have detailed finish reason easily available here
                         # Log what we have
                         yield from send_progress(f"Synthesis stream finished.")
                         break # Exit loop on stream end

                if synthesis_stream_error:
                     return # Terminate if an error occurred

            except Exception as e:
                 yield from send_error_event(f"Fatal error processing LLM synthesis stream: {e}")
                 import traceback
                 traceback.print_exc()
                 return

            yield from send_progress(f"Synthesis generation completed.")
            if not accumulated_synthesis.strip():
                 yield from send_error_event("Synthesis resulted in empty content. Check LLM response or source relevance.")
                 return

            # === Step 4: Generate Final Report (Streaming) ===
            yield from send_progress(f"Generating final report using {GOOGLE_MODEL_NAME} (AI Generating...)")
            yield from send_event({'type': 'stream_start', 'target': 'report'})

            # Ensure the synthesis + bibliography map doesn't exceed limits for the report prompt
            # (Less likely to be an issue than the source content itself, but good practice)
            MAX_REPORT_PROMPT_CHARS = 950000 # Slightly less than synthesis to be safe

            report_components_size = len(topic) + len(json.dumps(research_plan)) + len(accumulated_synthesis) + len(bibliography_prompt_list)

            if report_components_size > MAX_REPORT_PROMPT_CHARS:
                 yield from send_progress(f"  -> Warning: Combined inputs for final report prompt potentially large (~{report_components_size // 1000}k chars). Synthesis might be truncated in prompt.")
                 # Simple truncation strategy: Prioritize keeping the plan and bibliography map intact
                 available_chars_for_synthesis = MAX_REPORT_PROMPT_CHARS - (len(topic) + len(json.dumps(research_plan)) + len(bibliography_prompt_list) + 1000) # Reserve buffer
                 if available_chars_for_synthesis < 0: available_chars_for_synthesis = 0
                 truncated_synthesis = accumulated_synthesis[:available_chars_for_synthesis] + "\n... [Synthesis truncated due to length limit]"
            else:
                 truncated_synthesis = accumulated_synthesis


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
            1. Write a final research report in well-structured Markdown format.
            2. The report MUST include the following sections using Markdown headings:
                - `# Research Report: {topic}` (Main Title)
                - `## Introduction`: Briefly introduce the topic "{topic}", state the purpose of the report, and outline the research scope based on the plan's steps (list or describe them).
                - `## Findings`: Organize the main body strictly according to the {len(research_plan)} research plan steps. For each step:
                    - Use a subheading (e.g., `### Step X: <Step Description from Plan>`).
                    - Integrate the relevant "Synthesized Information" for that step provided above.
                    - **CRITICALLY IMPORTANT AND MANDATORY: Replace EVERY inline URL citation `[Source URL: <full_url_here>]` from the "Synthesized Information" with its corresponding Markdown footnote marker `[^N]`. Use the provided "Bibliography Map" to find the correct number N for the EXACT URL. Ensure the mapping is precise.**
                    - If a URL citation appears in the synthesis that is *not* present in the Bibliography Map (e.g., due to context truncation or hallucination), OMIT that specific citation marker. Do not invent footnote numbers or include broken references. Rephrase slightly if needed.
                    - Ensure smooth flow and readability within each step's section.
                - `## Conclusion`: Summarize the key findings derived from the synthesis across the most important plan steps. Briefly mention any significant limitations encountered (e.g., steps where information was lacking in sources, potential biases). Suggest potential areas for further research if applicable.
            3. Append a `## Bibliography` section at the very end of the report.
            4. In the Bibliography section, list all the sources from the "Bibliography Map" in numerical order (1, 2, 3...). Use the standard Markdown footnote definition format for each entry: `[^N]: <full_url_here>`. Make the URL clickable (Markdown automatically does this if format is correct).
            5. Ensure the final output is **only** the complete Markdown report, adhering strictly to the structure, formatting, citation replacement, and bibliography instructions. Do not include any commentary, introductory text about the process, apologies, or explanations outside the report content itself.
            """
            final_report_markdown = "" # Reset before accumulating
            report_stream_error = None
            report_finish_reason = None
            try:
                # Use the refactored stream_gemini
                stream_generator = stream_gemini(report_prompt)
                for result in stream_generator:
                    if result['type'] == 'chunk':
                        yield from send_event({'type': 'llm_chunk', 'content': result['content'], 'target': 'report'})
                        final_report_markdown += result['content']
                    elif result['type'] == 'stream_error':
                        report_stream_error = result['message']
                        yield from send_error_event(f"LLM stream error during report generation ({GOOGLE_MODEL_NAME}): {report_stream_error}")
                        return # Terminate on error
                    elif result['type'] == 'stream_warning':
                         yield from send_progress(f"LLM Stream Warning (Report): {result['message']}")
                         # Continue processing
                    elif result['type'] == 'stream_end':
                         report_finish_reason = result.get('finish_reason')
                         yield from send_progress(f"Report stream finished.")
                         break # Exit loop

                if report_stream_error:
                     return # Terminate if error occurred

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

            # Convert final accumulated report Markdown to HTML
            try:
                # Ensure footnotes extension is enabled
                report_html = md_lib.markdown(final_report_markdown, extensions=['footnotes', 'fenced_code', 'tables', 'nl2br', 'attr_list'])
            except Exception as md_err:
                 yield from send_error_event(f"Failed to convert final Markdown report to HTML: {md_err}")
                 from html import escape
                 report_html = f"<h3>Markdown Conversion Error</h3><p>Could not render report as HTML. Raw Markdown below:</p><pre>{escape(final_report_markdown)}</pre>"


            # Prepare final data payload
            # Limit preview size
            preview_limit = 3000
            raw_data_preview = json.dumps(scraped_sources_list[:3], indent=2, ensure_ascii=False) # Show first 3 sources
            if len(raw_data_preview) > preview_limit:
                 raw_data_preview = raw_data_preview[:preview_limit] + f"\n... (Preview truncated, {len(scraped_sources_list)} total sources scraped)"
            elif not scraped_sources_list:
                 raw_data_preview = "None"

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
            # Ensure error is sent to client even if it happens outside main steps
            yield from send_error_event(error_msg)
        finally:
            # Signal the end of the stream to the client, even if errors occurred
            # This helps the client know the connection should close.
             yield from send_event({'type': 'stream_terminated'})


    # Ensure headers prevent caching for SSE
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no', # Useful for Nginx proxying
        'Connection': 'keep-alive'
    }
    return Response(stream_with_context(generate_updates()), headers=headers)


# --- DOCX Download Route - No changes needed ---
@app.route('/download_docx', methods=['POST'])
def download_docx():
    """Converts the received Markdown report to DOCX and sends it as a download."""
    if not PANDOC_AVAILABLE:
        # Return a JSON error response for the frontend to handle
        return jsonify({"success": False, "message": "DOCX download is disabled because Pandoc is not correctly installed or configured."}), 400

    markdown_content = request.form.get('markdown_report')
    topic = request.form.get('topic', 'Research Report') # Get topic for filename

    if not markdown_content:
        return jsonify({"success": False, "message": "Error: No Markdown content received for conversion."}), 400

    try:
        # Use pypandoc to convert Markdown text to DOCX bytes
        # Ensure input encoding is handled correctly, default is usually UTF-8
        docx_bytes = pypandoc.convert_text(
            markdown_content,
            'docx',
            format='md',
            encoding='utf-8',
            # extra_args=['--reference-doc=my_template.docx'] # Optional: specify a template
        )

        # Create a BytesIO buffer to hold the DOCX data
        buffer = BytesIO(docx_bytes)
        buffer.seek(0)

        # Sanitize topic for filename
        safe_filename_topic = re.sub(r'[^\w\s-]', '', topic).strip()
        safe_filename_topic = re.sub(r'[-\s]+', '_', safe_filename_topic)
        # Truncate filename base if too long (OS limits vary, ~200 chars is safe)
        filename_base = f"{safe_filename_topic}_Research_Report"
        filename = f"{filename_base[:200]}.docx"

        # Send the file
        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename, # Use download_name (or attachment_filename for older Flask)
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

    except FileNotFoundError as e:
         # Check if the error message specifically mentions 'pandoc'
         if 'pandoc' in str(e).lower():
             print("ERROR: Pandoc executable not found during conversion attempt.")
             msg = "Error: Pandoc executable not found. Please ensure it's installed and in the system PATH."
             return jsonify({"success": False, "message": msg}), 500
         else:
             # Handle other potential FileNotFoundError cases if necessary
             print(f"Error during DOCX conversion (FileNotFound): {e}")
             return jsonify({"success": False, "message": f"An unexpected file error occurred: {e}"}), 500
    except Exception as e:
        # Catch errors during pypandoc conversion itself
        print(f"Error converting Markdown to DOCX using pypandoc: {e}")
        # Include more specific error info if pypandoc provides it
        return jsonify({"success": False, "message": f"An error occurred during DOCX conversion: {e}"}), 500


# --- Run the App ---
if __name__ == '__main__':
    # Ensure libraries are installed: pip install Flask python-dotenv google-generativeai duckduckgo-search beautifulsoup4 requests lxml Markdown pypandoc flask[async]
    # Ensure Pandoc executable is installed and in PATH: https://pandoc.org/installing.html
    # Set GOOGLE_API_KEY in your .env file
    print(f"Using Google Model: {GOOGLE_MODEL_NAME}")
    print(f"Pandoc/DOCX available: {PANDOC_AVAILABLE}")
    app.run(debug=True, host='127.0.0.1', port=5001, threaded=True) # threaded=True is important for handling concurrent requests/SSE