#!python
import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, stream_with_context, send_file
from openai import OpenAI
from dotenv import load_dotenv
from duckduckgo_search import DDGS
# Use the Markdown library for better footnote support
import markdown as md_lib
from urllib.parse import quote, unquote # Add unquote
import concurrent.futures # Import the concurrent futures library
from io import BytesIO # Needed for sending file data

# --- NEW: Import pypandoc ---
try:
    import pypandoc
    PANDOC_AVAILABLE = True
except ImportError:
    print("WARNING: pypandoc library not found. DOCX download will be disabled. Install with 'pip install pypandoc'")
    PANDOC_AVAILABLE = False
except OSError:
    print("WARNING: Pandoc executable not found in system PATH. DOCX download will be disabled. Ensure Pandoc is installed: https://pandoc.org/installing.html")
    PANDOC_AVAILABLE = False


# --- Configuration ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "google/gemini-pro-1.5") # Consider models with larger context if needed for many steps
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
MAX_SEARCH_RESULTS_PER_STEP = 10
MAX_TOTAL_URLS_TO_SCRAPE = 100
MAX_WORKERS = 10
REQUEST_TIMEOUT = 12
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36" # Updated UA string
PICO_CSS_CDN = "https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css"

# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Initialize OpenRouter Client ---
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_API_BASE)

# --- Helper Functions ---

# call_gemini - No changes needed
def call_gemini(prompt, system_prompt=None, max_retries=3, delay=5):
    """Calls the specified Gemini model via OpenRouter with retry logic."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=OPENROUTER_MODEL_NAME,
                messages=messages,
                temperature=0.6,
                # Consider adding max_tokens if context becomes an issue
            )

            if completion is None: raise ValueError("API response object is None.")
            if not completion.choices:
                if completion.choices is None: raise ValueError("API response 'choices' attribute was None.")
                else: raise ValueError("API response 'choices' attribute was empty.")
            if not completion.choices[0]: raise ValueError("API response first choice is invalid.")
            if not completion.choices[0].message: raise ValueError("API response choice missing 'message' attribute.")
            if completion.choices[0].message.content is None:
                 finish_reason = getattr(completion.choices[0], 'finish_reason', 'Unknown')
                 # Check for length finish reason, could indicate context overflow
                 if finish_reason == 'length':
                     raise ValueError(f"API response message content is None (finish reason: {finish_reason}). Prompt might be too long.")
                 raise ValueError(f"API response message content is None (finish reason: {finish_reason}).")

            response_content = completion.choices[0].message.content.strip()
            return response_content

        except Exception as e:
            print(f"Error calling OpenRouter or processing response (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Failing LLM call.")
                raise

# stream_gemini - No changes needed
def stream_gemini(prompt, system_prompt=None):
    """
    Calls the Gemini model with streaming enabled and yields content chunks.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        stream = client.chat.completions.create(
            model=OPENROUTER_MODEL_NAME,
            messages=messages,
            temperature=0.6,
            stream=True,
            # Consider adding max_tokens if context becomes an issue
        )
        complete_response = "" # Added to check finish reason
        finish_reason = None
        for chunk in stream:
            delta_content = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
            if delta_content:
                complete_response += delta_content
                yield {'type': 'chunk', 'content': delta_content} # Yield chunk data
            if chunk.choices and chunk.choices[0].finish_reason:
                 finish_reason = chunk.choices[0].finish_reason


        # After loop, check finish reason (especially for length limit)
        if finish_reason == 'length':
             print(f"Warning: LLM stream finished due to length. Response might be truncated. Prompt length: ~{len(prompt)}")
             # Optionally yield an error/warning event here
             yield {'type': 'stream_warning', 'message': f'LLM stream may have been truncated due to length limits (finish reason: {finish_reason}).'}

        yield {'type': 'stream_end', 'finish_reason': finish_reason} # Include finish reason

    except Exception as e:
        print(f"Error during LLM stream: {e}")
        # Yield a specific error type that the main generator can catch
        yield {'type': 'stream_error', 'message': str(e)}
        # Don't raise here, let the main generator handle termination

# parse_research_plan - No changes needed
def parse_research_plan(llm_response):
    # --- (Keep the robust parser from previous step) ---
    plan = []
    if not llm_response:
        print("Error: Received empty response from LLM for plan generation.")
        return [{"step_description": "Failed - Empty LLM response", "keywords": []}]

    raw_response = llm_response.strip()

    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_response, re.MULTILINE)
    json_str = None
    if match:
        json_str = match.group(1).strip()
        print("Attempting to parse JSON found within markdown code block.")
    else:
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
                if all(isinstance(item, dict) and 'step' in item and 'keywords' in item for item in data):
                    temp_plan = []
                    for item in data:
                        keywords_list = item.get('keywords', [])
                        if isinstance(keywords_list, str):
                            # Handle keywords separated by comma or newline
                            keywords_list = [k.strip() for k in re.split(r'[,\n]', keywords_list) if k.strip()]
                        elif not isinstance(keywords_list, list):
                            print(f"Warning: Keywords for step '{item.get('step')}' was not a list or string, setting to empty.")
                            keywords_list = []
                        # Filter out empty strings again just in case
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

    print("Attempting Markdown/text parsing as fallback...")
    pattern_regex = re.compile(
        r"^\s*(?:\d+\.?\s*[:-]?|-)\s*(.*?)" # More flexible separator after number
        r"(?:\s*[\(\[]\s*Keywords?\s*[:\-]?\s*(.*?)\s*[\)\]]\s*)?$",
        re.MULTILINE | re.IGNORECASE
    )
    matches = pattern_regex.findall(raw_response)
    if matches:
        print(f"Regex parsing found {len(matches)} potential steps.")
        plan = []
        for desc, keys_str in matches:
            desc = desc.strip()
            # More robust keyword removal from description if not captured separately
            desc = re.sub(r'\s*\(?Keywords?[:\-]?.*?\)?$', '', desc, flags=re.IGNORECASE).strip()
            keys = []
            if keys_str:
                 # Handle comma or newline separated keywords here too
                 keys = [k.strip() for k in re.split(r'[,\n]', keys_str) if k.strip()]
            if desc:
                 plan.append({"step_description": desc, "keywords": keys})
        if plan:
            print("Successfully parsed research plan using regex.")
            return plan
        else:
            print("Regex matched structure but failed to extract valid steps/keywords.")


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
                current_step = None # Reset if description becomes empty
        elif current_step and not current_step["keywords"]:
             # Check if the line *only* contains keywords
             is_keyword_line = False
             for marker in keyword_markers:
                 marker_lower = marker.lower()
                 if line.lower().startswith(marker_lower):
                     keys_str = line[len(marker):].strip()
                     # Handle comma/newline separators for keys
                     current_step["keywords"] = [k.strip() for k in re.split(r'[,\n]', keys_str) if k.strip()]
                     is_keyword_line = True
                     break
             # If it wasn't *just* a keyword line, maybe it's part of the description?
             # This part is tricky, avoid appending random lines. Let's stick to explicit steps.
             # if not is_keyword_line and current_step:
             #    current_step["step_description"] += "\n" + line # Might append unrelated lines

    if plan:
        plan = [p for p in plan if p.get("step_description")] # Final cleanup
        if plan:
             print("Parsed research plan using simple line-based approach.")
             return plan

    print("All parsing methods failed to extract a valid research plan.")
    return [{"step_description": "Failed to parse plan structure from LLM response", "keywords": []}]


# search_duckduckgo - No changes needed
def search_duckduckgo(keywords, max_results=MAX_SEARCH_RESULTS_PER_STEP):
    """Performs a search on DuckDuckGo using the library."""
    query = " ".join(keywords)
    urls = []
    if not query: return [] # Handle empty keywords
    print(f"  -> DDGS Searching for: '{query}' (max_results={max_results})")
    try:
        # Ensure DDGS is instantiated correctly within the function scope
        # Use context manager for proper resource management
        with DDGS(timeout=15) as ddgs:
            # Use list() to force iteration and catch potential errors within the context manager
            results = list(ddgs.text(query, max_results=max_results))
            urls = [r['href'] for r in results if r and 'href' in r]
            print(f"  -> DDGS Found {len(urls)} URLs for '{query}'")
    except Exception as e:
        print(f"Error searching DuckDuckGo for '{query}': {e}")
        # Optionally retry or just return empty list
    return urls

# scrape_url - No changes needed
def scrape_url(url):
    """
    Scrapes text content from a given URL.
    Returns a dictionary {'url': url, 'content': text} on success, None on failure.
    """
    # Shorten URL for logging
    log_url = url[:75] + '...' if len(url) > 75 else url
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True, stream=True) # Use stream=True initially
        response.raise_for_status() # Check for 4xx/5xx errors

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            print(f"Skipping non-HTML content: {log_url} (Type: {content_type})")
            response.close() # Ensure connection is closed
            return None

        content_length = response.headers.get('content-length')
        # Check content length *before* reading the whole thing if possible
        if content_length and int(content_length) > 10 * 1024 * 1024: # 10MB limit
            print(f"Skipping large file (>10MB): {log_url}")
            response.close()
            return None

        # Read content now, respecting potential size limit again if needed
        # This part might need refinement if very large pages cause memory issues even after the header check
        html_content = response.content # Reads the entire content

        response.close() # Close the connection after reading

        try:
            soup = BeautifulSoup(html_content, 'lxml') # Try lxml first
        except Exception:
            soup = BeautifulSoup(html_content, 'html.parser') # Fallback to html.parser

        # Remove unwanted tags more aggressively
        for element in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "input", "textarea", "select", "img", "figure", "iframe", "video", "audio", "picture", "source", "noscript", "meta", "link"]):
            element.decompose()

        # Try finding common main content containers
        main_content = soup.find('main') or \
                       soup.find('article') or \
                       soup.find('div', attrs={'role': 'main'}) or \
                       soup.find('div', id='content') or \
                       soup.find('div', class_=re.compile(r'\b(content|main|post|entry)\b', re.I)) or \
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
        # Break multi-sentence lines into potential paragraphs, then strip whitespace
        chunks = (' '.join(phrase.split()) for line in lines for phrase in line.split("  ") if phrase.strip())
        # Rejoin, keeping meaningful lines (more than just a couple of short words)
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk and len(chunk.split()) > 3) # Increased min words per line

        # Final check for meaningful content length
        meaningful_word_count = len([word for word in cleaned_text.split() if len(word) > 2]) # Count words > 2 chars
        if meaningful_word_count < 75: # Increased threshold
             print(f"Skipping due to low meaningful content ({meaningful_word_count} words): {log_url}")
             return None

        # print(f"Successfully scraped: {log_url} ({len(cleaned_text)} chars)")
        return {'url': url, 'content': cleaned_text}

    except requests.exceptions.Timeout:
        print(f"Timeout error fetching URL {log_url}")
    except requests.exceptions.RequestException as e:
        # More specific error logging for HTTP errors
        if hasattr(e, 'response') and e.response is not None:
            print(f"HTTP Error {e.response.status_code} fetching URL {log_url}: {e}")
        else:
            print(f"Request Error fetching URL {log_url}: {e}")
    except Exception as e:
        print(f"Error parsing or processing URL {log_url}: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed debugging if needed
    finally:
        # Ensure response is closed in case of exceptions before reading content
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
    # URL-encode the topic for safe inclusion in the results page URL
    encoded_topic = quote(topic)
    # *** NEW: Pass unencoded topic separately for display ***
    return render_template('results.html', topic=topic, encoded_topic=encoded_topic, pico_css=PICO_CSS_CDN)

@app.route('/stream')
def stream():
    """The main SSE route that performs research and streams progress."""
    # Topic needs to be retrieved from query parameters now
    encoded_topic = request.args.get('topic', '') # Get encoded topic
    topic = unquote(encoded_topic) # Decode topic for internal use
    if not topic:
        topic = "Default Topic" # Fallback if somehow empty

    # SSE Generator function
    def generate_updates():
        scraped_sources_list = []
        all_found_urls_set = set() # Track all URLs found during search
        urls_to_scrape_list = [] # List of unique URLs to actually scrape
        research_plan = []
        accumulated_synthesis = ""
        final_report_markdown = "" # Store final markdown
        url_to_index_map = {} # Keep track of bibliography mapping
        start_time_total = time.time() # Track total time

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
            yield from send_progress(f"Generating research plan for: '{topic}'...")
            plan_prompt = f"""
            Create a detailed, step-by-step research plan with 10-15 distinct steps for the topic: "{topic}"
            Each step should represent a specific question or area of inquiry related to the topic.
            Format the output STRICTLY as a JSON list of objects. Each object must have two keys:
            1. "step": A string describing the research step/question.
            2. "keywords": A list of 2-4 relevant keyword strings for searching about this step.

            Example format:
            [
              {{"step": "Define the core concept", "keywords": ["term definition", "term explanation"]}},
              {{"step": "Explore historical origins", "keywords": ["history of term", "term origins"]}},
              ... (10-15 steps total)
            ]

            Ensure the output is ONLY the valid JSON list and nothing else. No introductory text, no explanations, just the JSON.
            Generate 10 to 15 relevant steps for the specific topic: "{topic}".
            """
            try:
                plan_response = call_gemini(plan_prompt)
            except Exception as e:
                 yield from send_error_event(f"Failed to generate research plan from LLM: {e}")
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
                time.sleep(0.1)

            yield from send_progress(f"Search phase completed in {time.time() - start_search_time:.2f}s. Found {len(all_urls_from_search)} total URLs initially.")

            unique_urls = list(dict.fromkeys(all_urls_from_search))
            yield from send_progress(f"Processing {len(unique_urls)} unique URLs...")

            for url in unique_urls:
                 if not url: continue
                 if urls_collected_count >= MAX_TOTAL_URLS_TO_SCRAPE:
                      yield from send_progress(f"  -> Reached URL collection limit ({MAX_TOTAL_URLS_TO_SCRAPE}).")
                      break
                 if url in all_found_urls_set: continue

                 is_file = url.lower().split('?')[0].split('#')[0].endswith(('.pdf', '.jpg', '.png', '.gif', '.zip', '.mp4', '.mp3', '.docx', '.xlsx', '.pptx', '.webp', '.svg', '.xml', '.css', '.js'))
                 is_mailto = url.lower().startswith('mailto:')
                 is_javascript = url.lower().startswith('javascript:')
                 is_ftp = url.lower().startswith('ftp:')
                 is_valid_http = url.startswith(('http://', 'https://'))

                 if not is_file and not is_mailto and not is_javascript and not is_ftp and is_valid_http:
                      urls_to_scrape_list.append(url)
                      all_found_urls_set.add(url)
                      urls_collected_count += 1
                 else:
                    all_found_urls_set.add(url) # Add even non-scrapeable ones to track

            yield from send_progress(f"Selected {len(urls_to_scrape_list)} valid, unique URLs for scraping (limit: {MAX_TOTAL_URLS_TO_SCRAPE}).")

            if not urls_to_scrape_list:
                 yield from send_error_event("No suitable URLs found to scrape after searching and filtering.")
                 return

            # === Step 2b: Scrape URLs Concurrently ===
            yield from send_progress(f"Starting concurrent scraping of {len(urls_to_scrape_list)} URLs using up to {MAX_WORKERS} workers...")
            start_scrape_time = time.time()
            total_scraped_successfully = 0
            processed_scrape_count = 0

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
                        yield from send_progress(f"    -> Thread Error scraping {url[:60]}...: {exc}")
                    finally:
                         if processed_scrape_count % 5 == 0 or processed_scrape_count == len(urls_to_scrape_list):
                              progress_perc = (processed_scrape_count * 100) // len(urls_to_scrape_list)
                              yield from send_progress(f"  -> Scraping progress: {processed_scrape_count}/{len(urls_to_scrape_list)} ({progress_perc}% complete). Success: {total_scraped_successfully}")

            duration = time.time() - start_scrape_time
            yield from send_progress(f"Finished scraping. Successfully scraped {total_scraped_successfully}/{len(urls_to_scrape_list)} URLs in {duration:.2f} seconds.")

            if not scraped_sources_list:
                yield from send_error_event("Failed to scrape any content successfully from the selected URLs.")
                return

            scraped_url_map = {item['url']: item for item in scraped_sources_list}
            scraped_sources_list = [scraped_url_map[url] for url in urls_to_scrape_list if url in scraped_url_map]

            # === Bibliography Map ===
            # *** MODIFIED: Store map for later use ***
            url_to_index_map, bibliography_prompt_list = generate_bibliography_map(scraped_sources_list)

            # === Step 3: Synthesize with Citations (Streaming) ===
            yield from send_progress("Synthesizing relevant information (AI Generating...)")
            yield from send_event({'type': 'stream_start', 'target': 'synthesis'})

            context_for_llm_structured = scraped_sources_list
            estimated_chars = sum(len(item.get('content', '')) for item in context_for_llm_structured)
            yield from send_progress(f"  -> Synthesis context size: ~{estimated_chars // 1000}k chars from {len(scraped_sources_list)} sources.")

            if estimated_chars > 500000:
                yield from send_progress("  -> Warning: Context size is large, synthesis might be slow or hit limits.")

            synthesis_prompt = f"""
            Analyze the following scraped web content related to the topic "{topic}", following the research plan provided.
            Your goal is to synthesize the key information relevant to each step of the plan, citing sources accurately.

            Research Topic: {topic}

            Research Plan ({len(research_plan)} steps):
            {json.dumps(research_plan, indent=2)}

            Scraped Source Content (List of {len(context_for_llm_structured)} JSON objects, each with 'url' and 'content'):
            ```json
            {json.dumps(context_for_llm_structured, indent=2, ensure_ascii=False)}
            ```

            Instructions:
            1. Process each step in the Research Plan sequentially.
            2. For each step, carefully review ALL the scraped source content to find relevant information.
            3. Extract and synthesize the most pertinent facts, findings, or data related *specifically* to that research step. Be concise and focus on unique information per step.
            4. **Crucially and Mandatorily**: Immediately after stating **any** piece of information derived from a source, cite the source using the exact format: `[Source URL: <full_url_here>]`. Cite every claim. Do not group citations at the end of sentences or paragraphs.
            5. If you cannot find relevant information for a specific plan step within the provided sources, explicitly state: "No specific information found for this step in the provided sources." Do not invent information.
            6. Structure your output clearly using Markdown. Use a heading (e.g., `### Step X: <Step Description>`) for each plan step.
            7. Output ONLY the synthesized information with inline citations, structured by plan step. Do NOT include an introduction, conclusion, summary, or any other text outside the step-by-step synthesis in this response. Separate each step's section clearly (e.g., using `---` between steps).
            """
            accumulated_synthesis = ""
            synthesis_stream_error = None
            synthesis_finish_reason = None
            try:
                stream_generator = stream_gemini(synthesis_prompt)
                for result in stream_generator:
                    if result['type'] == 'chunk':
                        yield from send_event({'type': 'llm_chunk', 'content': result['content'], 'target': 'synthesis'})
                        accumulated_synthesis += result['content']
                    elif result['type'] == 'stream_error':
                        synthesis_stream_error = result['message']
                        yield from send_error_event(f"LLM stream error during synthesis: {synthesis_stream_error}")
                        return
                    elif result['type'] == 'stream_warning':
                         yield from send_progress(f"LLM Stream Warning (Synthesis): {result['message']}")
                    elif result['type'] == 'stream_end':
                         synthesis_finish_reason = result.get('finish_reason')
                         if synthesis_finish_reason == 'length':
                              yield from send_progress("Warning: Synthesis output might be truncated due to LLM length limits.")
                         break

                if synthesis_stream_error:
                     return

            except Exception as e:
                 yield from send_error_event(f"Fatal error processing LLM synthesis stream: {e}")
                 return

            yield from send_progress(f"Synthesis stream finished. (Finish reason: {synthesis_finish_reason or 'Normal'})")
            if not accumulated_synthesis.strip():
                 yield from send_error_event("Synthesis resulted in empty content.")
                 return

            # === Step 4: Generate Final Report (Streaming) ===
            yield from send_progress("Generating final report (AI Generating...)")
            yield from send_event({'type': 'stream_start', 'target': 'report'})

            report_prompt = f"""
            Create a comprehensive Markdown research report on the topic: "{topic}".

            You have the following inputs:
            1. The Original Research Plan ({len(research_plan)} steps):
               ```json
               {json.dumps(research_plan, indent=2)}
               ```

            2. Synthesized Information with Raw URL Citations (Result of previous step):
               ```markdown
               {accumulated_synthesis}
               ```

            3. Bibliography Map (URL to Reference Number - {len(url_to_index_map)} sources):
               ```
               {bibliography_prompt_list}
               ```

            Instructions:
            1. Write a final research report in Markdown format.
            2. The report must include the following sections using Markdown headings:
                - `# Research Report: {topic}` (Main Title)
                - `## Introduction`: Briefly introduce the topic "{topic}" and outline the research scope based on the plan's steps.
                - `## Findings`: Organize the main body according to the {len(research_plan)} research plan steps. For each step:
                    - Use a subheading (e.g., `### Step X: <Step Description>`).
                    - Integrate the relevant "Synthesized Information" for that step provided above.
                    - **Crucially and Mandatorily: Replace every inline URL citation `[Source URL: <full_url_here>]` from the "Synthesized Information" with its corresponding Markdown footnote marker `[^N]`, using the "Bibliography Map" to find the correct number N.** Ensure the mapping is precise. If a URL citation appears in the synthesis that is *not* present in the Bibliography Map, you should omit that specific citation or rephrase the sentence slightly if the citation was critical. Do not invent footnote numbers.
                - `## Conclusion`: Summarize the key findings derived from the synthesis across the plan steps. Briefly mention any limitations (e.g., steps where information was lacking) or potential areas for further research.
            3. Append a `## Bibliography` section at the very end of the report.
            4. In the Bibliography section, list all the sources from the "Bibliography Map" in numerical order (1, 2, 3...). Use the standard Markdown footnote definition format for each entry: `[^N]: [URL](URL)`. Ensure the URL is clickable in the final rendered HTML.
            5. Ensure the final output is **only** the complete Markdown report, following the structure and formatting instructions precisely. Do not include any commentary, introductory text about the process, or explanations outside the report content itself.
            """
            final_report_markdown = "" # Reset before accumulating
            report_stream_error = None
            report_finish_reason = None
            try:
                stream_generator = stream_gemini(report_prompt)
                for result in stream_generator:
                    if result['type'] == 'chunk':
                        yield from send_event({'type': 'llm_chunk', 'content': result['content'], 'target': 'report'})
                        final_report_markdown += result['content'] # *** MODIFIED: Store raw markdown ***
                    elif result['type'] == 'stream_error':
                        report_stream_error = result['message']
                        yield from send_error_event(f"LLM stream error during report generation: {report_stream_error}")
                        return
                    elif result['type'] == 'stream_warning':
                         yield from send_progress(f"LLM Stream Warning (Report): {result['message']}")
                    elif result['type'] == 'stream_end':
                         report_finish_reason = result.get('finish_reason')
                         if report_finish_reason == 'length':
                              yield from send_progress("Warning: Final report output might be truncated due to LLM length limits.")
                         break

                if report_stream_error:
                     return

            except Exception as e:
                 yield from send_error_event(f"Fatal error processing LLM report stream: {e}")
                 return

            yield from send_progress(f"Report stream finished. (Finish reason: {report_finish_reason or 'Normal'})")
            if not final_report_markdown.strip():
                 yield from send_error_event("Report generation resulted in empty content.")
                 return

            # Convert final accumulated report Markdown to HTML
            try:
                report_html = md_lib.markdown(final_report_markdown, extensions=['footnotes', 'fenced_code', 'tables', 'nl2br', 'attr_list'])
            except Exception as md_err:
                 yield from send_error_event(f"Failed to convert final Markdown report to HTML: {md_err}")
                 from html import escape
                 report_html = f"<pre>{escape(final_report_markdown)}</pre>"


            # --- Send Final Completion Event ---
            final_data = {
                'type': 'complete',
                'report_html': report_html,
                # *** NEW: Send raw markdown for download feature ***
                'report_markdown': final_report_markdown,
                'raw_scraped_data_preview': json.dumps(scraped_sources_list[:2], indent=2, ensure_ascii=False)[:3000] + "..." if scraped_sources_list else "None",
                # *** NEW: Flag if DOCX download is possible ***
                'docx_available': PANDOC_AVAILABLE
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

    return Response(stream_with_context(generate_updates()), content_type='text/event-stream')


# --- NEW: DOCX Download Route ---
@app.route('/download_docx', methods=['POST'])
def download_docx():
    """Converts the received Markdown report to DOCX and sends it as a download."""
    if not PANDOC_AVAILABLE:
        return "DOCX download is disabled because Pandoc is not correctly installed or configured.", 400

    markdown_content = request.form.get('markdown_report')
    topic = request.form.get('topic', 'Research Report') # Get topic for filename

    if not markdown_content:
        return "Error: No Markdown content received for conversion.", 400

    try:
        # Use pypandoc to convert Markdown text to DOCX bytes
        docx_bytes = pypandoc.convert_text(
            markdown_content,
            'docx',
            format='md',
            # extra_args=['--reference-doc=my_template.docx'] # Optional: specify a template
        )

        # Create a BytesIO buffer to hold the DOCX data
        buffer = BytesIO(docx_bytes)
        buffer.seek(0)

        # Sanitize topic for filename
        safe_filename_topic = re.sub(r'[^\w\s-]', '', topic).strip()
        safe_filename_topic = re.sub(r'[-\s]+', '_', safe_filename_topic)
        filename = f"{safe_filename_topic[:50]}_Research_Report.docx" # Limit filename length

        # Send the file
        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename, # Use download_name for Flask >= 2.0
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

    except FileNotFoundError:
         print("ERROR: Pandoc executable not found during conversion attempt.")
         return "Error: Pandoc executable not found. Please ensure it's installed and in the system PATH.", 500
    except Exception as e:
        print(f"Error converting Markdown to DOCX: {e}")
        return f"An error occurred during DOCX conversion: {e}", 500


# --- Run the App ---
if __name__ == '__main__':
    # Ensure libraries are installed: pip install Flask python-dotenv openai duckduckgo-search beautifulsoup4 requests lxml Markdown pypandoc
    # Ensure Pandoc executable is installed and in PATH
    app.run(debug=True, host='127.0.0.1', port=5001, threaded=True)