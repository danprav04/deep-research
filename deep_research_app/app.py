import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, stream_with_context # Added Response, stream_with_context
from openai import OpenAI
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import markdown
from urllib.parse import quote # For bibliography links

# --- Configuration --- (Same as before)
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "google/gemini-pro-1.5")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
MAX_SEARCH_RESULTS_PER_STEP = 5
MAX_TOTAL_URLS_TO_SCRAPE = 20 # Slightly reduced for faster demo
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flashing or session, good practice

# --- Initialize OpenRouter Client --- (Same as before)
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_API_BASE)

# --- Helper Functions ---

# call_gemini (Same as before)
def call_gemini(prompt, system_prompt=None, max_retries=3, delay=5):
    """Calls the specified Gemini model via OpenRouter with retry logic."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    # print(f"\n--- Calling LLM --- \nPrompt: {prompt[:500]}...\n-------------------\n") # DEBUG
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=OPENROUTER_MODEL_NAME,
                messages=messages,
                temperature=0.6, # Slightly lower temp for more factual synthesis/reporting
            )
            response_content = completion.choices[0].message.content.strip()
            # print(f"\n--- LLM Response --- \nResponse: {response_content[:500]}...\n--------------------\n") # DEBUG
            return response_content
        except Exception as e:
            print(f"Error calling OpenRouter (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Failing LLM call.")
                raise # Re-raise the exception after final attempt

# parse_research_plan (Same as before, maybe minor prompt tweaks reflected)
def parse_research_plan(llm_response):
    """
    Parses the LLM response to extract the research plan and keywords.
    Handles potential JSON in markdown code blocks and falls back to text parsing.
    (Using the robust version from the previous iteration)
    """
    # --- Using the robust parser from the previous iteration ---
    # (Include the full robust parse_research_plan function here)
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
    # print(f"Searching DDG for: '{query}' (Max results: {max_results})") # Covered by SSE update
    try:
        with DDGS(timeout=15) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            urls = [r['href'] for r in results if 'href' in r]
            # print(f"Found {len(urls)} URLs from DDG.") # Covered by SSE update
    except Exception as e:
        print(f"Error searching DuckDuckGo for '{query}': {e}") # Log error
        # Optionally yield an error message here?
    return urls


# scrape_url (MODIFIED to return dict)
def scrape_url(url):
    """
    Scrapes text content from a given URL.
    Returns a dictionary {'url': url, 'content': text} on success, None on failure.
    """
    # print(f"Attempting to scrape: {url}") # Covered by SSE update
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            # print(f"Skipping non-HTML content: {url} (Content-Type: {content_type})") # Minor info, maybe skip yielding
            return None

        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 10 * 1024 * 1024:
             # print(f"Skipping large file (>10MB): {url}")
             return None

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
        if meaningful_word_count < 50:
             # print(f"Skipping - likely low content (found {meaningful_word_count} long words): {url}")
             return None

        # print(f"Successfully scraped ~{len(cleaned_text)} characters from: {url}") # Covered by SSE
        return {'url': url, 'content': cleaned_text} # Return dict

    except requests.exceptions.Timeout:
        print(f"Timeout error fetching URL {url}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
    except Exception as e:
        print(f"Error parsing or processing URL {url}: {e}")
    return None # Return None on any scraping error

def format_bibliography(scraped_sources_dict):
     """Formats the bibliography section as HTML."""
     if not scraped_sources_dict:
         return ""

     html = "<h3>Bibliography</h3><ul>"
     for i, data in enumerate(scraped_sources_dict.values()):
         url = data['url']
         # Simple title extraction attempt (often fails or gets garbage)
         title = data.get('title', url) # Use URL as fallback title
         html += f'<li>[{i+1}] <a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a> ({url})</li>'
     html += "</ul>"
     return html

def generate_bibliography_map(scraped_sources_list):
    """Creates a map of {url: index} and a numbered list for prompts."""
    url_to_index = {data['url']: i + 1 for i, data in enumerate(scraped_sources_list)}
    numbered_list_str = "\n".join([f"{i+1}. {data['url']}" for i, data in enumerate(scraped_sources_list)])
    return url_to_index, numbered_list_str

# --- Flask Routes ---

@app.route('/')
def index():
    """Displays the main input form."""
    return render_template('index.html')

@app.route('/research', methods=['POST'])
def research_start():
    """
    Receives the topic, redirects to the results page which will
    then connect to the SSE stream.
    """
    topic = request.form.get('topic')
    if not topic:
        # Handle error - maybe flash message and redirect back
        return redirect(url_for('index')) # Simplest redirect

    # Redirect to the results page, passing the topic
    # The results page's JS will connect to /stream with the topic
    return render_template('results.html', topic=topic)


@app.route('/stream')
def stream():
    """The main SSE route that performs research and streams progress."""
    topic = request.args.get('topic', 'Default Topic') # Get topic from query param

    def generate_updates():
        """Generator function for SSE updates."""
        scraped_sources_list = [] # Holds {'url': url, 'content': text}
        all_scraped_urls_set = set() # Track unique URLs attempted/scraped
        plan_details_for_template = [] # Store plan steps attempted
        research_plan = []

        def send_progress(message):
            """Helper to format and yield progress updates."""
            yield f"data: {json.dumps({'type': 'progress', 'message': message})}\n\n"

        def send_error(message, fatal=False):
            """Helper to format and yield error updates."""
            print(f"ERROR: {message}") # Also log server-side
            yield f"data: {json.dumps({'type': 'error', 'message': message})}\n\n"
            if fatal:
                # Optionally raise an exception here to stop the generator if needed,
                # but yielding the error is often enough for the client.
                 raise StopIteration("Fatal error occurred")


        try:
            # === Step 1: Generate Research Plan ===
            yield from send_progress(f"Generating research plan for: '{topic}'...")
            # (Use the stricter JSON prompt from previous iteration)
            plan_prompt = f"""
            Create a concise, step-by-step research plan to investigate the following topic: "{topic}"
            Format the output STRICTLY as a JSON list of objects. Each object MUST have a 'step' key (string description) and a 'keywords' key (list of strings).
            Do NOT include any text before or after the JSON list. Just output the raw JSON list starting with '[' and ending with ']'.
            Example JSON format:
            [
              {{"step": "Origin and Early History", "keywords": ["watermelon origin", "history of watermelon africa", "ancient egypt watermelon", "early watermelon cultivation"]}},
              {{"step": "Spread and Varieties", "keywords": ["watermelon spread world", "watermelon united states history", "types of watermelon", "heirloom watermelon varieties"]}},
              {{"step": "Modern Cultivation and Significance", "keywords": ["modern watermelon farming", "watermelon production statistics", "cultural significance watermelon", "watermelon industry"]}}
            ]
            Ensure the output is ONLY the valid JSON list. Use relevant steps/keywords for "{topic}". Keep the plan to 3-4 steps.
            """
            plan_response = call_gemini(plan_prompt)
            # print("\n--- RAW LLM Plan Response ---") # Keep for server debug if needed
            # print(plan_response)
            # print("--- End RAW LLM Plan Response ---\n")
            research_plan = parse_research_plan(plan_response)

            if not research_plan or (len(research_plan) == 1 and research_plan[0]["step_description"].startswith("Failed")):
                 fail_reason = research_plan[0]["step_description"] if research_plan else "Unknown parsing error"
                 raw_resp_snippet = f" Raw Response Snippet: {plan_response[:200]}..." if plan_response else ""
                 yield from send_error(f"Failed to generate or parse research plan. Reason: '{fail_reason}'.{raw_resp_snippet}", fatal=True)
                 return # Stop generation

            yield from send_progress(f"Generated {len(research_plan)} step plan.")
            # Log plan steps for user
            for i, step in enumerate(research_plan):
                 yield from send_progress(f"  Plan Step {i+1}: {step['step_description']} (Keywords: {', '.join(step['keywords'])})")


            # === Step 2: Search and Scrape ===
            yield from send_progress("Starting web search and scraping...")
            total_urls_scraped_successfully = 0

            for i, step in enumerate(research_plan):
                step_desc = step.get('step_description', f'Unnamed Step {i+1}')
                keywords = step.get('keywords', [])
                step_scraped_urls_list = [] # URLs scraped in this step

                yield from send_progress(f"Processing Step {i+1}: '{step_desc}'")

                if not keywords:
                    yield from send_progress("  -> No keywords for this step, skipping search.")
                    plan_details_for_template.append({**step, "urls": [], "keywords": keywords})
                    continue

                yield from send_progress(f"  -> Searching for: {', '.join(keywords)}")
                step_urls = search_duckduckgo(keywords, MAX_SEARCH_RESULTS_PER_STEP)
                yield from send_progress(f"  -> Found {len(step_urls)} potential URLs.")

                urls_to_scrape_this_step = 0
                for url in step_urls:
                    if total_urls_scraped_successfully >= MAX_TOTAL_URLS_TO_SCRAPE:
                        yield from send_progress("  -> Reached max scraping limit, skipping remaining URLs for this step.")
                        break
                    if url in all_scraped_urls_set:
                        # yield from send_progress(f"  -> Skipping already processed URL: {url}")
                        continue
                    if url.lower().endswith(('.pdf', '.jpg', '.png', '.gif', '.zip', '.mp4', '.mp3', '.docx', '.xlsx', '.pptx')):
                         # yield from send_progress(f"  -> Skipping non-textual file type: {url}")
                         all_scraped_urls_set.add(url)
                         continue
                    if urls_to_scrape_this_step >= MAX_SEARCH_RESULTS_PER_STEP: # Limit attempts per step
                        yield from send_progress(f"  -> Reached scraping attempt limit for this step, skipping remaining URLs.")
                        break

                    all_scraped_urls_set.add(url) # Mark as attempted
                    urls_to_scrape_this_step += 1
                    yield from send_progress(f"  -> Scraping ({urls_to_scrape_this_step}/{MAX_SEARCH_RESULTS_PER_STEP}): {url}")
                    scraped_data = scrape_url(url) # Returns dict {'url': url, 'content': text} or None

                    if scraped_data and isinstance(scraped_data, dict) and scraped_data.get('content'):
                         # Store successful scrapes
                         scraped_sources_list.append(scraped_data)
                         step_scraped_urls_list.append(url)
                         total_urls_scraped_successfully += 1
                         yield from send_progress(f"    -> Success ({len(scraped_data['content'])} chars). Total scraped: {total_urls_scraped_successfully}")
                    else:
                         yield from send_progress(f"    -> Failed or skipped.")
                         # Optionally add URL to a 'failed_urls' list if needed

                plan_details_for_template.append({**step, "urls": step_scraped_urls_list, "keywords": keywords})
                time.sleep(0.5) # Small delay between steps


            if not scraped_sources_list:
                yield from send_error("Failed to scrape any content from the web.", fatal=True)
                return

            yield from send_progress(f"Finished scraping. Total successful scrapes: {total_urls_scraped_successfully}")

            # === Create Bibliography Map ===
            url_to_index_map, bibliography_prompt_list = generate_bibliography_map(scraped_sources_list)
            yield from send_progress("Generated bibliography map.")

            # === Step 3: Synthesize with Citations ===
            yield from send_progress("Synthesizing relevant information from scraped data (This may take a while)...")

            # Prepare context for LLM (list of dictionaries)
            context_for_llm_structured = scraped_sources_list # Pass the list directly

             # Estimate token count crudely (4 chars/token) + overhead
            estimated_chars = sum(len(item['content']) for item in context_for_llm_structured)
            yield from send_progress(f"  -> Preparing context for LLM (~{estimated_chars} chars)...")
            # NOTE: Context limits apply. Gemini 1.5 Pro has large limits, but check OpenRouter specifics.
            # Truncation might still be needed for extremely large scrapes. Implement if necessary.


            # **MODIFIED PROMPT FOR CITATION**
            synthesis_prompt = f"""
            You are a research assistant analyzing scraped web content about "{topic}".
            Your goal is to extract and synthesize information relevant to the research plan, **citing the source URL for each piece of information**.

            Research Plan:
            {json.dumps(research_plan, indent=2)}

            List of Scraped Sources (Content may be partial/cleaned):
            --- START SCRAPED CONTENT LIST ---
            {json.dumps(context_for_llm_structured, indent=2)}
            --- END SCRAPED CONTENT LIST ---

            Instructions:
            1.  Carefully read the research plan.
            2.  Go through the content from EACH scraped source.
            3.  Identify information directly related to *each* step of the plan.
            4.  Synthesize the findings for each step concisely. Focus on key facts, figures, concepts.
            5.  **Crucially:** For *every* significant piece of information you include in your synthesis, **you MUST indicate the source URL** it came from immediately after the information, formatted like this: `[Source URL: http://example.com/page]`. If information is synthesized from multiple sources, list them: `[Source URL: http://example.com/page1][Source URL: http://example.com/page2]`. Be precise.
            6.  Structure your output clearly using Markdown. For each step in the plan, create a section:
                ### Step: [Exact Step Description from Plan]
                [Synthesized information relevant to this step, with [Source URL: ...] citations embedded directly after the information they support.]
                ---
            7.  If NO relevant information for a specific step is found in the provided text, explicitly state: "No specific information found for this step in the provided text." under the corresponding step heading.
            8.  Do not add introductions or summaries in *this* response. Just provide the structured, synthesized, **cited** information per step.
            """
            synthesized_cited_info = call_gemini(synthesis_prompt)
            yield from send_progress("Synthesis complete.")

            # === Step 4: Generate Final Report with Bibliography ===
            yield from send_progress("Generating final report with bibliography (This may take a while)...")

            # **MODIFIED PROMPT FOR REPORT AND BIBLIOGRAPHY**
            report_prompt = f"""
            You are a research analyst preparing a comprehensive report on: "{topic}"

            Use the following materials:
            1.  Original Research Topic: "{topic}"
            2.  Research Plan Followed:
                {json.dumps(research_plan, indent=2)}
            3.  Synthesized Information with Source URL Citations:
                --- START SYNTHESIZED CITED INFO ---
                {synthesized_cited_info}
                --- END SYNTHESIZED CITED INFO ---
            4.  Bibliography Map (URL to Citation Number):
                --- START BIBLIOGRAPHY MAP ---
                {bibliography_prompt_list}
                --- END BIBLIOGRAPHY MAP ---

            Report Generation Instructions:
            *   Write a formal research report in Markdown format.
            *   Start with a clear title (e.g., "# Research Report: {topic}").
            *   Include a brief Introduction section summarizing the research topic and the planned approach.
            *   Create a main section for Findings, structured according to the research plan. For each step:
                *   Use a subheading (e.g., "## [Step Description from Plan]").
                *   Present the key findings based *only* on the 'Synthesized Information' provided above.
                *   **Critically:** Replace every `[Source URL: http://...]` marker in the synthesized text with the corresponding citation number formatted as a Markdown link `[^N]`, where `N` is the number from the 'Bibliography Map'. For example, if `http://example.com/page1` is number `1` in the map, replace `[Source URL: http://example.com/page1]` with `[^1]`. If multiple sources are cited, format as `[^1][^2]`.
                *   Elaborate slightly on the synthesized points for clarity and flow, maintaining an objective tone.
                *   If the synthesized info stated nothing was found, reflect that in the report for that step.
            *   Include a Conclusion section summarizing the main findings.
            *   **Finally, append a "Bibliography" section.** Under an `## Bibliography` heading, list all the sources from the 'Bibliography Map' in numerical order, formatted like this:
                [^N]: [URL](URL)
                For example:
                [^1]: [http://example.com/page1](http://example.com/page1)
                [^2]: [http://example.com/page2](http://example.com/page2)
            *   Format the entire output as a single Markdown document. Use standard Markdown for formatting (headings, lists, bold, etc.). Ensure the citation links `[^N]` correctly correspond to the bibliography entries `[^N]:`.
            *   Output ONLY the final Markdown report. No preamble or extra comments.
            """
            final_report_md = call_gemini(report_prompt)
            yield from send_progress("Report generated.")

            # Convert Markdown report to HTML for display
            # Using extensions that support footnotes (like python-markdown's footnotes)
            # Note: Standard 'markdown' library doesn't handle footnotes well.
            # Install 'Markdown' library: pip install Markdown
            try:
                # Using the 'Markdown' library for better footnote support
                import markdown as md_lib # Rename import to avoid conflict
                report_html = md_lib.markdown(final_report_md, extensions=['footnotes', 'fenced_code', 'tables', 'nl2br'])
                # The 'footnotes' extension handles the [^N] and [^N]: syntax
            except ImportError:
                 yield from send_progress("Markdown library not found, using basic conversion (footnotes may not render correctly). `pip install Markdown`")
                 report_html = markdown.markdown(final_report_md) # Fallback


            # Generate Bibliography HTML separately (optional, could be part of report)
            # bibliography_html = format_bibliography(scraped_sources_dict) # Using the map format generated by LLM now

            # Prepare final data payload
            final_data = {
                'type': 'complete',
                'report_html': report_html,
                'bibliography_html': "", # Bibliography is now part of report_html via footnotes
                'raw_scraped_data_preview': json.dumps(scraped_sources_list[:2], indent=2)[:5000] # Show first couple scraped items
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        except StopIteration as si:
             # Expected when send_error(fatal=True) is called
             print(f"Stopping stream due to fatal error: {si}")
        except Exception as e:
            # Catch any unexpected errors during the process
            print(f"An unexpected error occurred during stream generation: {e}")
            import traceback
            traceback.print_exc()
            # Send a final error message to the client
            error_msg = f"An unexpected server error occurred: {type(e).__name__} - {e}"
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"

    # Return the generator function wrapped in a Response object for SSE
    return Response(stream_with_context(generate_updates()), content_type='text/event-stream')


# --- Run the App ---
if __name__ == '__main__':
    # Make sure you have the Markdown library installed for footnote support:
    # pip install Markdown
    app.run(debug=True, host='127.0.0.1', port=5001, threaded=True) # Use threaded=True for handling concurrent requests during SSE