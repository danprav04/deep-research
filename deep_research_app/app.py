#!python
import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, stream_with_context
from openai import OpenAI
from dotenv import load_dotenv
from duckduckgo_search import DDGS
# Use the Markdown library for better footnote support
import markdown as md_lib
from urllib.parse import quote

# --- Configuration ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "google/gemini-pro-1.5")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
MAX_SEARCH_RESULTS_PER_STEP = 5
MAX_TOTAL_URLS_TO_SCRAPE = 20
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
PICO_CSS_CDN = "https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css"

# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Initialize OpenRouter Client ---
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_API_BASE)

# --- Helper Functions ---

# call_gemini (Robust version from previous iteration)
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
            )

            if completion is None: raise ValueError("API response object is None.")
            if not completion.choices:
                if completion.choices is None: raise ValueError("API response 'choices' attribute was None.")
                else: raise ValueError("API response 'choices' attribute was empty.")
            if not completion.choices[0]: raise ValueError("API response first choice is invalid.")
            if not completion.choices[0].message: raise ValueError("API response choice missing 'message' attribute.")
            if completion.choices[0].message.content is None:
                 finish_reason = getattr(completion.choices[0], 'finish_reason', 'Unknown')
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

# stream_gemini (From previous iteration)
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
        )
        for chunk in stream:
            delta_content = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
            if delta_content:
                yield {'type': 'chunk', 'content': delta_content} # Yield chunk data

        yield {'type': 'stream_end'}

    except Exception as e:
        print(f"Error during LLM stream: {e}")
        yield {'type': 'stream_error', 'message': str(e)}
        raise

# parse_research_plan (Robust version from previous iteration)
def parse_research_plan(llm_response):
    # --- Include the full robust parse_research_plan function here ---
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
                            keywords_list = [k.strip() for k in keywords_list.split(',') if k.strip()]
                        elif not isinstance(keywords_list, list):
                            print(f"Warning: Keywords for step '{item.get('step')}' was not a list or string, setting to empty.")
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
                         print("Parsed JSON structure, but failed to extract valid step descriptions.")
                else:
                     print("Parsed JSON but structure/keys ('step', 'keywords') are incorrect or missing. Falling back.")
            else:
                print("Parsed JSON but it's not a list or is empty. Falling back.")
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing failed: {json_err}. Falling back to text parsing.")

    print("Attempting Markdown/text parsing as fallback...")
    pattern_regex = re.compile(
        r"^\s*(?:\d+\.?|-)\s*(.*?)"
        r"(?:\s*[\(\[]\s*Keywords?\s*[:\-]?\s*(.*?)\s*[\)\]]\s*)?$",
        re.MULTILINE | re.IGNORECASE
    )
    matches = pattern_regex.findall(raw_response)
    if matches:
        print(f"Regex parsing found {len(matches)} potential steps.")
        plan = []
        for desc, keys_str in matches:
            desc = desc.strip()
            desc = re.sub(r'\s*Keywords?\s*:.*$', '', desc, flags=re.IGNORECASE).strip()
            keys = []
            if keys_str:
                keys = [k.strip() for k in keys_str.split(',') if k.strip()]
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
        step_match = re.match(r"^\s*(?:step\s+)?(\d+)\s*[:.\-]?\s*(.*)", line, re.IGNORECASE)
        if step_match:
            step_num_str, step_text = step_match.groups()
            step_desc = step_text.strip()
            keys = []
            for marker in keyword_markers:
                marker_lower = marker.lower()
                if marker_lower in step_desc.lower():
                    parts = re.split(marker, step_desc, maxsplit=1, flags=re.IGNORECASE)
                    if len(parts) == 2:
                        step_desc = parts[0].strip()
                        keys = [k.strip() for k in parts[1].split(',') if k.strip()]
                        break
            if step_desc:
                current_step = {"step_description": step_desc, "keywords": keys}
                plan.append(current_step)
            else:
                current_step = None
        elif current_step and not current_step["keywords"]:
             for marker in keyword_markers:
                 marker_lower = marker.lower()
                 if line.lower().startswith(marker_lower):
                     keys_str = line[len(marker):].strip()
                     current_step["keywords"] = [k.strip() for k in keys_str.split(',') if k.strip()]
                     break
    if plan:
        plan = [p for p in plan if p.get("step_description")]
        if plan:
             print("Parsed research plan using simple line-based approach.")
             return plan

    print("All parsing methods failed to extract a valid research plan.")
    return [{"step_description": "Failed to parse plan structure from LLM response", "keywords": []}]
    # --- End of robust parser ---

# search_duckduckgo (Same as before)
def search_duckduckgo(keywords, max_results=MAX_SEARCH_RESULTS_PER_STEP):
    """Performs a search on DuckDuckGo using the library."""
    query = " ".join(keywords)
    urls = []
    try:
        with DDGS(timeout=15) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            urls = [r['href'] for r in results if 'href' in r]
    except Exception as e:
        print(f"Error searching DuckDuckGo for '{query}': {e}")
    return urls

# scrape_url (Same as before, returning dict)
def scrape_url(url):
    """
    Scrapes text content from a given URL.
    Returns a dictionary {'url': url, 'content': text} on success, None on failure.
    """
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type: return None
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 10 * 1024 * 1024: return None

        try: soup = BeautifulSoup(response.content, 'lxml')
        except Exception: soup = BeautifulSoup(response.content, 'html.parser')

        for element in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "input", "textarea", "select"]):
            element.decompose()

        main_content = soup.find('main') or soup.find('article') or \
                       soup.find('div', attrs={'role': 'main'}) or \
                       soup.find('div', id='content') or soup.find('div', class_='content') or \
                       soup.find('div', id='main') or soup.find('div', class_='main')

        if main_content: text = main_content.get_text(separator='\n', strip=True)
        else:
             body = soup.find('body')
             if body: text = body.get_text(separator='\n', strip=True)
             else: text = soup.get_text(separator='\n', strip=True)

        lines = (line.strip() for line in text.splitlines())
        chunks = (' '.join(phrase.split()) for line in lines for phrase in line.split("  ") if phrase.strip())
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk and len(chunk.split()) > 1)
        meaningful_word_count = len([word for word in cleaned_text.split() if len(word) > 3])
        if meaningful_word_count < 50: return None

        return {'url': url, 'content': cleaned_text}

    except requests.exceptions.Timeout: print(f"Timeout error fetching URL {url}")
    except requests.exceptions.RequestException as e: print(f"Error fetching URL {url}: {e}")
    except Exception as e: print(f"Error parsing or processing URL {url}: {e}")
    return None

# generate_bibliography_map (Same as before)
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
    return render_template('results.html', topic=topic, pico_css=PICO_CSS_CDN)


@app.route('/stream')
def stream():
    """The main SSE route that performs research and streams progress."""
    topic = request.args.get('topic', 'Default Topic')

    # SSE Generator function
    def generate_updates():
        scraped_sources_list = []
        all_scraped_urls_set = set()
        plan_details_for_template = []
        research_plan = []
        accumulated_synthesis = ""
        accumulated_report = ""

        def send_event(data):
            """Helper to format and yield SSE events."""
            yield f"data: {json.dumps(data)}\n\n"

        def send_progress(message):
            yield from send_event({'type': 'progress', 'message': message})

        def send_error(message, fatal=False):
            print(f"ERROR: {message}")
            yield from send_event({'type': 'error', 'message': message})
            if fatal: raise StopIteration("Fatal error occurred")

        try:
            # === Step 1: Generate Research Plan ===
            yield from send_progress(f"Generating research plan for: '{topic}'...")
            # *** FIXED F-STRING HERE ***
            plan_prompt = f"""
            Create a concise, 3-4 step research plan for: "{topic}"
            Format STRICTLY as a JSON list of objects with 'step' and 'keywords' keys.
            Example: [{{"step": "Origin", "keywords": ["kw1", "kw2"]}}]. NO extra text.
            Ensure the output is ONLY the valid JSON list. Use relevant steps/keywords for "{topic}". Keep the plan to 3-4 steps.
            """
            plan_response = call_gemini(plan_prompt)
            research_plan = parse_research_plan(plan_response)

            if not research_plan or (len(research_plan) == 1 and research_plan[0]["step_description"].startswith("Failed")):
                 fail_reason = research_plan[0]["step_description"] if research_plan else "Unknown"
                 raw_snippet = f" Raw: {plan_response[:100]}..." if plan_response else ""
                 yield from send_error(f"Failed plan: {fail_reason}.{raw_snippet}", fatal=True)
                 return

            yield from send_progress(f"Generated {len(research_plan)} step plan:")
            for i, step in enumerate(research_plan):
                 yield from send_progress(f"  {i+1}. {step['step_description']}")

            # === Step 2: Search and Scrape ===
            yield from send_progress("Starting web search & scraping...")
            total_scraped_successfully = 0
            for i, step in enumerate(research_plan):
                step_desc = step.get('step_description', f'Unnamed Step {i+1}')
                keywords = step.get('keywords', [])
                step_scraped_urls_list = []
                yield from send_progress(f"Step {i+1}: '{step_desc}'")
                if not keywords:
                    yield from send_progress("  -> No keywords, skipping search.")
                    plan_details_for_template.append({**step, "urls": [], "keywords": keywords})
                    continue

                yield from send_progress(f"  -> Searching: {', '.join(keywords)}")
                step_urls = search_duckduckgo(keywords, MAX_SEARCH_RESULTS_PER_STEP)
                yield from send_progress(f"  -> Found {len(step_urls)} URLs.")

                urls_to_scrape_this_step = 0
                for url in step_urls:
                    if total_scraped_successfully >= MAX_TOTAL_URLS_TO_SCRAPE:
                        yield from send_progress("  -> Max scrape limit reached.")
                        break
                    if url in all_scraped_urls_set: continue
                    if url.lower().endswith(('.pdf', '.jpg', '.png', '.gif', '.zip', '.mp4', '.mp3', '.docx', '.xlsx', '.pptx')):
                         all_scraped_urls_set.add(url)
                         continue
                    if urls_to_scrape_this_step >= MAX_SEARCH_RESULTS_PER_STEP:
                        yield from send_progress(f"  -> Step scrape limit reached.")
                        break

                    all_scraped_urls_set.add(url)
                    urls_to_scrape_this_step += 1
                    yield from send_progress(f"  -> Scraping ({urls_to_scrape_this_step}/{MAX_SEARCH_RESULTS_PER_STEP}): {url[:70]}...")
                    scraped_data = scrape_url(url)

                    if scraped_data and isinstance(scraped_data, dict) and scraped_data.get('content'):
                         scraped_sources_list.append(scraped_data)
                         step_scraped_urls_list.append(url)
                         total_scraped_successfully += 1
                         yield from send_progress(f"    -> OK ({len(scraped_data['content'])} chars). Total: {total_scraped_successfully}")
                plan_details_for_template.append({**step, "urls": step_scraped_urls_list, "keywords": keywords})
                time.sleep(0.2)

            if not scraped_sources_list:
                yield from send_error("Failed to scrape any content.", fatal=True)
                return
            yield from send_progress(f"Finished scraping. Total successful: {total_scraped_successfully}")

            # === Bibliography Map ===
            url_to_index_map, bibliography_prompt_list = generate_bibliography_map(scraped_sources_list)

            # === Step 3: Synthesize with Citations (Streaming) ===
            yield from send_progress("Synthesizing relevant information (AI Generating...)")
            yield from send_event({'type': 'stream_start', 'target': 'synthesis'})

            context_for_llm_structured = scraped_sources_list
            estimated_chars = sum(len(item['content']) for item in context_for_llm_structured)
            yield from send_progress(f"  -> Context size: ~{estimated_chars // 1000}k chars")

            # Note: Synthesis prompt uses regular f-string interpolation safely here
            synthesis_prompt = f"""
            Analyze scraped content about "{topic}" based on the plan, citing sources.
            Research Plan: {json.dumps(research_plan, indent=2)}
            Scraped Sources: {json.dumps(context_for_llm_structured, indent=2)}
            Instructions: For each plan step, synthesize relevant info. **Crucially:** Cite every fact with `[Source URL: http://...]` immediately after. Output ONLY the Markdown synthesis per step, separated by ---. If nothing found, state it. No intro/summary here.
            """
            accumulated_synthesis = ""
            try:
                stream_generator = stream_gemini(synthesis_prompt)
                for result in stream_generator:
                    if result['type'] == 'chunk':
                        yield from send_event({'type': 'llm_chunk', 'content': result['content'], 'target': 'synthesis'})
                        accumulated_synthesis += result['content']
                    elif result['type'] == 'stream_error':
                        yield from send_error(f"LLM stream error during synthesis: {result['message']}", fatal=True)
                        return
            except Exception as e:
                 yield from send_error(f"Error processing LLM synthesis stream: {e}", fatal=True)
                 return

            yield from send_progress("Synthesis stream finished.")
            if not accumulated_synthesis.strip():
                 yield from send_error("Synthesis resulted in empty content.", fatal=True)
                 return

            # === Step 4: Generate Final Report (Streaming) ===
            yield from send_progress("Generating final report (AI Generating...)")
            yield from send_event({'type': 'stream_start', 'target': 'report'})

            # Note: Report prompt uses regular f-string interpolation safely here
            report_prompt = f"""
            Create a Markdown research report on "{topic}".
            Plan: {json.dumps(research_plan, indent=2)}
            Synthesized Info w/ URL Citations: {accumulated_synthesis}
            Bibliography Map (URL to Number): {bibliography_prompt_list}
            Instructions: Write Intro, Findings (per plan step), Conclusion. **Replace** `[Source URL: http://...]` with footnote markers `[^N]` based on the Bibliography Map. Append a `## Bibliography` section listing sources as `[^N]: [URL](URL)`. Output ONLY the final Markdown report.
            """
            accumulated_report = ""
            try:
                stream_generator = stream_gemini(report_prompt)
                for result in stream_generator:
                    if result['type'] == 'chunk':
                        yield from send_event({'type': 'llm_chunk', 'content': result['content'], 'target': 'report'})
                        accumulated_report += result['content']
                    elif result['type'] == 'stream_error':
                        yield from send_error(f"LLM stream error during report generation: {result['message']}", fatal=True)
                        return
            except Exception as e:
                 yield from send_error(f"Error processing LLM report stream: {e}", fatal=True)
                 return

            yield from send_progress("Report stream finished.")
            if not accumulated_report.strip():
                 yield from send_error("Report generation resulted in empty content.", fatal=True)
                 return

            # Convert final accumulated report Markdown to HTML
            try:
                report_html = md_lib.markdown(accumulated_report, extensions=['footnotes', 'fenced_code', 'tables', 'nl2br'])
            except Exception as md_err:
                 yield from send_error(f"Failed to convert final Markdown report to HTML: {md_err}")
                 report_html = f"<pre>{accumulated_report}</pre>"


            # --- Send Final Completion Event ---
            final_data = {
                'type': 'complete',
                'report_html': report_html,
                'raw_scraped_data_preview': json.dumps(scraped_sources_list[:1], indent=2)[:2000]
            }
            yield from send_event(final_data)

        except StopIteration:
             yield from send_progress("Process stopped due to fatal error.")
        except Exception as e:
            print(f"An unexpected error occurred during stream generation: {e}")
            import traceback
            traceback.print_exc()
            error_msg = f"Unexpected server error: {type(e).__name__} - {e}"
            yield from send_event({'type': 'error', 'message': error_msg})

    return Response(stream_with_context(generate_updates()), content_type='text/event-stream')


# --- Run the App ---
if __name__ == '__main__':
    # Ensure Markdown library is installed: pip install Markdown
    app.run(debug=True, host='127.0.0.1', port=5001, threaded=True)