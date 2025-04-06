import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, jsonify
from openai import OpenAI # Use the OpenAI library for OpenRouter compatibility
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import markdown

# --- Configuration ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Default model, ensure it has a large context window for best results
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "google/gemini-pro-1.5")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

MAX_SEARCH_RESULTS_PER_STEP = 5 # Number of results per keyword set from DDG
MAX_TOTAL_URLS_TO_SCRAPE = 25 # Limit total URLs scraped across all steps
REQUEST_TIMEOUT = 10 # Timeout for scraping requests in seconds
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" # Be a good citizen

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Initialize OpenRouter Client ---
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_API_BASE,
)

# --- Helper Functions ---

def call_gemini(prompt, system_prompt=None, max_retries=3, delay=5):
    """Calls the specified Gemini model via OpenRouter with retry logic."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            # print(f"--- Sending Prompt to {OPENROUTER_MODEL_NAME} (Attempt {attempt+1}) ---")
            # print(f"System Prompt: {system_prompt}")
            # print(f"User Prompt (first 500 chars): {prompt[:500]}...")
            # print("--- End Prompt ---")

            completion = client.chat.completions.create(
                model=OPENROUTER_MODEL_NAME,
                messages=messages,
                temperature=0.7, # Adjust for creativity vs factualness
                # max_tokens=4096, # Often not needed for newer models
            )
            response_content = completion.choices[0].message.content.strip()
            # print(f"--- LLM Call Successful (Attempt {attempt+1}) ---")
            # print(f"Response (first 500 chars): {response_content[:500]}...") # Log truncated response
            return response_content
        except Exception as e:
            print(f"Error calling OpenRouter (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Failing LLM call.")
                raise  # Re-raise the exception after final attempt

# *** NEW/UPDATED PARSER ***
def parse_research_plan(llm_response):
    """
    Parses the LLM response to extract the research plan and keywords.
    Handles potential JSON in markdown code blocks and falls back to text parsing.
    """
    plan = []
    if not llm_response:
        print("Error: Received empty response from LLM for plan generation.")
        return [{"step_description": "Failed - Empty LLM response", "keywords": []}]

    raw_response = llm_response.strip() # Strip leading/trailing whitespace

    # Attempt 1: Extract JSON from markdown code blocks ```json ... ``` or ``` ... ```
    # Allows optional language specifier (like json) and flexible spacing
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_response, re.MULTILINE)
    json_str = None
    if match:
        json_str = match.group(1).strip()
        print("Attempting to parse JSON found within markdown code block.")
    else:
        # Attempt 2: Assume the entire response might be JSON (after stripping)
        # Check if it looks like JSON (starts with [ or { and ends with matching bracket)
        if (raw_response.startswith('[') and raw_response.endswith(']')) or \
           (raw_response.startswith('{') and raw_response.endswith('}')):
            json_str = raw_response
            print("No markdown code block found. Attempting to parse entire response as JSON.")
        else:
           print("Response doesn't appear to be JSON or within a code block. Proceeding to text parsing.")
           pass # Will proceed to regex/text parsing later

    if json_str:
        try:
            data = json.loads(json_str)
            if isinstance(data, list) and data: # Check if it's a non-empty list
                # Check structure more carefully
                if all(isinstance(item, dict) and 'step' in item and 'keywords' in item for item in data):
                    temp_plan = []
                    for item in data:
                        keywords_list = item.get('keywords', [])
                        # Ensure keywords is a list of strings
                        if isinstance(keywords_list, str):
                            # Handle common case of comma-separated string
                            keywords_list = [k.strip() for k in keywords_list.split(',') if k.strip()]
                        elif not isinstance(keywords_list, list):
                             # Handle cases where it might be None or other types
                            print(f"Warning: Keywords for step '{item.get('step')}' was not a list or string, setting to empty.")
                            keywords_list = []

                        # Ensure all items in list are strings and stripped
                        valid_keywords = [str(k).strip() for k in keywords_list if str(k).strip()]

                        step_desc = str(item.get('step', 'N/A')).strip()
                        if step_desc and step_desc != 'N/A': # Basic validation
                            temp_plan.append({
                                "step_description": step_desc,
                                "keywords": valid_keywords
                            })
                        else:
                             print(f"Warning: Skipping step with invalid description: {item.get('step')}")

                    if temp_plan: # Only return if we actually extracted valid steps
                        print("Successfully parsed research plan as JSON.")
                        return temp_plan
                    else:
                         print("Parsed JSON structure, but failed to extract valid step descriptions.")
                else:
                     print("Parsed JSON but structure/keys ('step', 'keywords') are incorrect or missing in some items. Falling back.")
            else:
                print("Parsed JSON but it's not a list or is empty. Falling back.")
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing failed: {json_err}. Falling back to text parsing.")
            # Fallthrough to text parsing below

    # If JSON parsing failed or wasn't attempted, try regex/text parsing on the ORIGINAL raw_response
    print("Attempting Markdown/text parsing as fallback...")

    # Fallback 1: Regex for numbered list with [Keywords: ...]
    # Made slightly more flexible regarding spacing and optional colon/brackets/parentheses
    # Captures step description and keywords separately
    pattern_regex = re.compile(
        r"^\s*(?:\d+\.?|-)\s*(.*?)" # Step number/bullet and description part
        r"(?:\s*[\(\[]\s*Keywords?\s*[:\-]?\s*(.*?)\s*[\)\]]\s*)?$", # Optional keywords part
        re.MULTILINE | re.IGNORECASE
    )
    matches = pattern_regex.findall(raw_response)

    if matches:
        print(f"Regex parsing found {len(matches)} potential steps.")
        plan = [] # Reset plan if we are using regex results
        for desc, keys_str in matches:
            desc = desc.strip()
            # Further clean description if keywords part was somehow included without brackets
            desc = re.sub(r'\s*Keywords?\s*:.*$', '', desc, flags=re.IGNORECASE).strip()

            keys = []
            if keys_str: # Check if keywords string was captured
                keys = [k.strip() for k in keys_str.split(',') if k.strip()]

            # Avoid adding steps with empty descriptions sometimes caught by regex
            if desc:
                 plan.append({
                    "step_description": desc,
                    "keywords": keys
                 })
        if plan:
            print("Successfully parsed research plan using regex.")
            return plan
        else:
            print("Regex matched structure but failed to extract valid steps/keywords.")


    # Fallback 2: Simple line parsing (less reliable)
    print("Regex parsing failed or yielded no results. Trying simple line parsing.")
    lines = raw_response.strip().split('\n')
    plan = [] # Reset plan
    current_step = None
    keyword_markers = ["Keywords:", "keywords:", "Search Terms:", "search terms:", "Keywords -"]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to detect a step start (number, dot/colon/dash, text)
        step_match = re.match(r"^\s*(?:step\s+)?(\d+)\s*[:.\-]?\s*(.*)", line, re.IGNORECASE)

        if step_match:
            step_num_str, step_text = step_match.groups()
            step_desc = step_text.strip()
            keys = []

            # Check if keywords are on the *same* line using defined markers
            for marker in keyword_markers:
                marker_lower = marker.lower()
                if marker_lower in step_desc.lower():
                    # Split carefully, case-insensitive
                    parts = re.split(marker, step_desc, maxsplit=1, flags=re.IGNORECASE)
                    if len(parts) == 2:
                        step_desc = parts[0].strip()
                        keys = [k.strip() for k in parts[1].split(',') if k.strip()]
                        break # Found keywords on this line

            # Only add if description is not empty
            if step_desc:
                current_step = {"step_description": step_desc, "keywords": keys}
                plan.append(current_step)
            else:
                current_step = None # Discard if no description text after potential keyword split

        # Check if the line contains keywords for the *previous* step (if keywords not found on step line)
        elif current_step and not current_step["keywords"]:
             for marker in keyword_markers:
                 marker_lower = marker.lower()
                 if line.lower().startswith(marker_lower):
                     keys_str = line[len(marker):].strip()
                     current_step["keywords"] = [k.strip() for k in keys_str.split(',') if k.strip()]
                     # print(f"Found keywords on separate line for step: {current_step['step_description']}")
                     break # Found keywords for the previous step

    if plan:
        # Final filter for empty descriptions
        plan = [p for p in plan if p.get("step_description")]
        if plan:
             print("Parsed research plan using simple line-based approach.")
             return plan

    # If all parsing fails
    print("All parsing methods failed to extract a valid research plan.")
    # Return a structure indicating failure, but still a list of dicts
    return [{"step_description": "Failed to parse plan structure from LLM response", "keywords": []}]


def search_duckduckgo(keywords, max_results=MAX_SEARCH_RESULTS_PER_STEP):
    """Performs a search on DuckDuckGo using the library."""
    query = " ".join(keywords)
    urls = []
    print(f"Searching DDG for: '{query}' (Max results: {max_results})")
    try:
        # Use context manager for DDGS
        with DDGS(timeout=15) as ddgs: # Increased timeout slightly
            results = list(ddgs.text(query, max_results=max_results))
            urls = [r['href'] for r in results if 'href' in r]
            print(f"Found {len(urls)} URLs from DDG.")
    except Exception as e:
        print(f"Error searching DuckDuckGo for '{query}': {e}")
    return urls

def scrape_url(url):
    """Scrapes text content from a given URL."""
    print(f"Attempting to scrape: {url}")
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Check content type - only parse HTML
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            print(f"Skipping non-HTML content: {url} (Content-Type: {content_type})")
            return None

        # Check size - avoid excessively large files if needed (e.g., > 10MB)
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 10 * 1024 * 1024:
             print(f"Skipping large file (>10MB): {url}")
             return None

        # Specify parser, 'html.parser' is built-in, 'lxml' is faster if installed
        try:
             soup = BeautifulSoup(response.content, 'lxml') # Try lxml first
        except Exception: # Fallback if lxml not installed
             soup = BeautifulSoup(response.content, 'html.parser')


        # Remove script, style, nav, footer, header, aside elements more aggressively
        for element in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "input", "textarea", "select"]):
            element.decompose()

        # Get text, trying different common content containers
        # Prioritize main content areas if identifiable
        main_content = soup.find('main') or \
                       soup.find('article') or \
                       soup.find('div', attrs={'role': 'main'}) or \
                       soup.find('div', id='content') or \
                       soup.find('div', class_='content') or \
                       soup.find('div', id='main') or \
                       soup.find('div', class_='main') # Add more specific selectors if needed

        if main_content:
             text = main_content.get_text(separator='\n', strip=True)
        else:
             # Fallback to body if no main content found
             body = soup.find('body')
             if body:
                 text = body.get_text(separator='\n', strip=True)
             else:
                 text = soup.get_text(separator='\n', strip=True) # Full fallback


        # Basic cleaning - improved slightly
        lines = (line.strip() for line in text.splitlines())
        # Handle multiple spaces between words better
        chunks = (' '.join(phrase.split()) for line in lines for phrase in line.split("  ") if phrase.strip())
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk and len(chunk.split()) > 1) # Keep lines with more than one word

        # Very basic check to filter out boilerplate/low-content pages
        meaningful_word_count = len([word for word in cleaned_text.split() if len(word) > 3])
        if meaningful_word_count < 50: # Heuristic: Require at least 50 words longer than 3 chars
             print(f"Skipping - likely low content (found {meaningful_word_count} long words): {url}")
             return None

        print(f"Successfully scraped ~{len(cleaned_text)} characters from: {url}")
        return cleaned_text

    except requests.exceptions.Timeout:
        print(f"Timeout error fetching URL {url}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
    except Exception as e:
        # Catch potential BeautifulSoup errors or others
        print(f"Error parsing or processing URL {url}: {e}")
    return None

# --- Flask Routes ---

@app.route('/')
def index():
    """Displays the main input form."""
    return render_template('index.html')

@app.route('/research', methods=['POST'])
def research():
    """Handles the research request."""
    topic = request.form.get('topic')
    if not topic:
        return render_template('index.html', error="Research topic cannot be empty.")

    print(f"\n--- Starting Research for Topic: {topic} ---")
    full_scraped_text = ""
    plan_details_for_template = [] # Store details for results page
    all_scraped_urls = set() # Use a set to avoid scraping duplicates
    plan_response = "" # Initialize for potential error message

    try:
        # === Step 1: Generate Research Plan & Keywords ===
        print("Step 1: Generating research plan and keywords...")
        plan_prompt = f"""
        Create a concise, step-by-step research plan to investigate the following topic: "{topic}"

        For each step in the plan, provide:
        1. A clear description of the research goal for that step.
        2. A list of 3-5 specific search keywords relevant to achieving that step's goal.

        Format the output STRICTLY as a JSON list of objects. Each object MUST have a 'step' key (string description) and a 'keywords' key (list of strings).
        Do NOT include any text before or after the JSON list (like "Here is the JSON:" or ```json). Just output the raw JSON list starting with '[' and ending with ']'.

        Example JSON format:
        [
          {{
            "step": "Understand the basic definition and history of Watermelons",
            "keywords": ["define Watermelon", "history of Watermelons", "Watermelon origin", "what are Watermelons", "ancient Watermelons"]
          }},
          {{
            "step": "Identify key varieties and cultivation of Watermelons",
            "keywords": ["Watermelon varieties", "types of Watermelons", "growing Watermelons", "Watermelon cultivation", "Watermelon farming"]
          }},
          {{
            "step": "Explore nutritional value and health benefits of Watermelons",
            "keywords": ["Watermelon nutrition facts", "Watermelon health benefits", "is Watermelon good for you", "vitamins in Watermelon", "Watermelon hydration"]
          }}
        ]

        Ensure the output is ONLY the valid JSON list. Replace the examples with relevant steps and keywords for "{topic}".
        Keep the plan focused, 3-5 steps maximum.
        """

        try: # Add a try/except specifically around the LLM call for better debugging
            plan_response = call_gemini(plan_prompt) # Get the raw response
            print("\n--- RAW LLM Plan Response ---") # Log the raw response
            print(plan_response)
            print("--- End RAW LLM Plan Response ---\n")
        except Exception as llm_err:
            print(f"Error during LLM call for plan generation: {llm_err}")
            # Raise a more specific error that can be caught below
            raise ValueError(f"Failed to get response from LLM for plan generation: {llm_err}")

        # Parse the response using the robust parser
        research_plan = parse_research_plan(plan_response)

        # Check if parsing fundamentally failed (returned the specific failure message)
        if not research_plan or (len(research_plan) == 1 and research_plan[0]["step_description"].startswith("Failed")):
             print(f"Failed Plan Parsing. Raw response was:\n{plan_response}") # Log again if parsing failed
             # Provide a more informative error message including the parser's conclusion
             fail_reason = research_plan[0]["step_description"] if research_plan else "Unknown parsing error"
             raise ValueError(f"Failed to parse a valid research plan from the LLM. Reason: '{fail_reason}'. Check console logs for raw LLM response.")
        print(f"Successfully Parsed Plan: {research_plan}")


        # === Step 2: Search and Scrape ===
        print("\nStep 2: Searching and scraping based on plan...")
        total_urls_scraped = 0
        scraped_data_by_step = {} # Store text associated with the step that found it

        for i, step in enumerate(research_plan):
            step_desc = step.get('step_description', f'Unnamed Step {i+1}')
            keywords = step.get('keywords', [])
            print(f"\n-- Processing Step {i+1}: {step_desc} --")
            print(f"   Keywords: {keywords}")

            if not keywords or not isinstance(keywords, list) or len(keywords) == 0:
                print("   Skipping step due to missing or invalid keywords.")
                plan_details_for_template.append({**step, "urls": [], "keywords": keywords}) # Add to template data even if skipped
                continue

            # Search using keywords for this step
            step_urls = search_duckduckgo(keywords, MAX_SEARCH_RESULTS_PER_STEP)
            valid_urls_for_step = []

            step_scraped_content = ""
            step_scraped_urls_list = [] # Keep track of URLs scraped for this specific step

            # Scrape results for this step, respecting total limit and avoiding duplicates
            urls_to_scrape_this_step = 0
            for url in step_urls:
                if total_urls_scraped >= MAX_TOTAL_URLS_TO_SCRAPE:
                    print("   Reached maximum total URL scraping limit.")
                    break
                if url in all_scraped_urls:
                    print(f"   Skipping already processed URL: {url}")
                    # Still add to template list if desired, or skip? Let's skip adding again.
                    continue
                # Basic check for file extensions often not useful for scraping text
                if url.lower().endswith(('.pdf', '.jpg', '.png', '.gif', '.zip', '.mp4', '.mp3', '.docx', '.xlsx', '.pptx')):
                     print(f"   Skipping non-textual file type: {url}")
                     all_scraped_urls.add(url) # Add to prevent re-check, but don't scrape
                     continue

                # Limit scraping attempts per step too
                if urls_to_scrape_this_step >= MAX_SEARCH_RESULTS_PER_STEP:
                    print(f"   Reached scraping attempt limit for this step ({MAX_SEARCH_RESULTS_PER_STEP}).")
                    break

                content = scrape_url(url)
                all_scraped_urls.add(url) # Add to set regardless of success to avoid retrying
                urls_to_scrape_this_step += 1

                if content:
                    step_scraped_content += f"\n\n--- Content from {url} (related to step: '{step_desc}') ---\n{content}"
                    total_urls_scraped += 1
                    step_scraped_urls_list.append(url) # Add successful URL to step list

            if step_scraped_content:
                scraped_data_by_step[step_desc] = step_scraped_content
                full_scraped_text += step_scraped_content # Append to the global context

            # Add step details for template, including the actual keywords used and scraped URLs
            plan_details_for_template.append({**step, "urls": step_scraped_urls_list, "keywords": keywords})

            print(f"-- Step {i+1} finished. Scraped {len(step_scraped_urls_list)} new URLs for this step. Total URLs scraped: {total_urls_scraped} --")


        if not full_scraped_text:
            # Check if we had a plan but just failed to scrape anything
            if research_plan and len(research_plan) > 0 and not research_plan[0]['step_description'].startswith("Failed"):
                 error_message = "Successfully generated a research plan, but failed to scrape any relevant content from the web search results. This could be due to network issues, anti-scraping measures on websites, or irrelevant search results."
            else:
                 # If plan generation also failed, stick to that error
                 error_message = "Failed to generate a research plan, and therefore could not proceed with web scraping."
            raise ValueError(error_message)


        print(f"\nTotal scraped text length: {len(full_scraped_text)} characters.")

        # === Step 3: Extract Relevant Data (Focusing Step) ===
        print("\nStep 3: Asking LLM to synthesize relevant information from scraped data...")
        # Adjust context size limits based on the model (Gemini 1.5 Pro has a large limit)
        # OpenRouter typically uses token limits, converting characters is approximate.
        # Assume ~4 chars/token. 1M token limit ~ 4M chars. Let's use a safer limit.
        MAX_CONTEXT_CHARS = 1_500_000 # Example: 1.5 million characters limit for safety

        truncated = False
        if len(full_scraped_text) > MAX_CONTEXT_CHARS:
            print(f"Warning: Scraped text ({len(full_scraped_text)} chars) exceeds limit ({MAX_CONTEXT_CHARS} chars), truncating.")
            context_for_llm = full_scraped_text[:MAX_CONTEXT_CHARS]
            truncated = True
        else:
            context_for_llm = full_scraped_text

        extraction_prompt = f"""
        You are a research assistant analyzing scraped web content about "{topic}".
        Your goal is to extract and synthesize information relevant to the research plan provided below.

        Research Plan:
        {json.dumps(research_plan, indent=2)}

        Scraped Web Content:
        --- START SCRAPED CONTENT ---
        {context_for_llm}
        --- END SCRAPED CONTENT ---
        {'*Note: The provided scraped content was truncated due to length limitations.*' if truncated else ''}

        Instructions:
        1. Carefully read the research plan.
        2. Go through the scraped web content and identify information directly related to *each* step of the plan.
        3. Synthesize the findings for each step concisely. Focus on the key facts, figures, and concepts relevant to the step's goal.
        4. Structure your output clearly. Use Markdown. For each step in the plan, create a section:
           ### Step: [Exact Step Description from Plan]
           [Synthesized information relevant to this step found in the text. Be factual and draw *only* from the provided text. Write in complete sentences/paragraphs.]
           ---
        5. If NO relevant information for a specific step is found in the provided text, explicitly state: "No specific information found for this step in the provided text." under the corresponding step heading.
        6. Do not add introductions, conclusions, or summaries in *this* response. Just provide the structured, synthesized information per step.
        """
        relevant_info_response = call_gemini(extraction_prompt)
        print("LLM synthesis/extraction complete.")

        # === Step 4: Generate Final Report ===
        print("\nStep 4: Generating the final research report...")
        report_prompt = f"""
        You are a research analyst preparing a comprehensive report on: "{topic}"

        Use the following materials:
        1.  Original Research Topic: "{topic}"
        2.  Research Plan Followed:
            {json.dumps(research_plan, indent=2)}
        3.  Synthesized Information from Web Research (organized by plan step):
            --- START SYNTHESIZED INFO ---
            {relevant_info_response}
            --- END SYNTHESIZED INFO ---

        Report Generation Instructions:
        *   Write a formal research report in Markdown format.
        *   Start with a clear title (e.g., "# Research Report: {topic}").
        *   Include a brief Introduction section summarizing the research topic and the planned approach (mention the steps briefly).
        *   Create a main section for Findings, structured according to the research plan. For each step:
            *   Use a subheading (e.g., "## [Step Description from Plan]").
            *   Present the key findings synthesized for that step, based *only* on the 'Synthesized Information' provided above.
            *   Elaborate slightly on the synthesized points for clarity, maintaining an objective tone. Ensure the information flows well.
            *   If the synthesized info stated nothing was found, reflect that in the report for that step (e.g., "The research yielded no specific information on this aspect from the sources analyzed.").
        *   Include a Conclusion section summarizing the main findings across all research steps. Briefly reiterate the key takeaways regarding "{topic}".
        *   Format the entire output as a single Markdown document.
        *   Crucially: Output ONLY the Markdown report. Do not include any conversational text, preamble like "Here is the report:", or post-report comments.
        """
        final_report_md = call_gemini(report_prompt)
        print("Final report generated by LLM.")

        # Convert Markdown report to HTML for display
        # Add extensions for better formatting like tables and fenced code blocks
        report_html = markdown.markdown(final_report_md, extensions=['fenced_code', 'tables', 'nl2br', 'toc'])

        # --- Render Results ---
        return render_template('results.html',
                               topic=topic,
                               plan_details=plan_details_for_template, # Pass the detailed plan execution
                               scraped_content_length=len(full_scraped_text),
                               raw_scraped_data_preview=full_scraped_text[:5000], # Show a preview
                               report_html=report_html,
                               error=None)

    except ValueError as ve:
         # Catches explicit ValueErrors raised, like plan parsing failure or scraping failure
         print(f"Value Error during research process: {ve}")
         # Pass the raw plan response to the template ONLY if it was a parsing error
         error_details = str(ve)
         if "Failed to parse" in error_details and plan_response:
             error_details += f"\n\n--- Raw LLM Response causing parse error ---\n{plan_response[:1000]}..." # Show preview in error

         return render_template('results.html', topic=topic, error=error_details, plan_details=plan_details_for_template) # Pass plan details even on error if available

    except requests.exceptions.RequestException as re:
        print(f"Network Error during research process: {re}")
        return render_template('results.html', topic=topic, error=f"A network error occurred during web scraping or API calls: {re}", plan_details=plan_details_for_template)

    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback to console for debugging
        # Provide a generic error message but include exception type/message
        error_msg = f"An unexpected error occurred: {type(e).__name__} - {e}"
        return render_template('results.html', topic=topic, error=error_msg, plan_details=plan_details_for_template)


# --- Run the App ---
if __name__ == '__main__':
    # Set host='0.0.0.0' to make it accessible on your network (use with caution)
    # Use debug=True only for development, not production
    # Choose a different port if 5001 is in use
    app.run(debug=True, host='127.0.0.1', port=5001)