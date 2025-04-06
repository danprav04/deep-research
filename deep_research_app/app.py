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
            completion = client.chat.completions.create(
                model=OPENROUTER_MODEL_NAME,
                messages=messages,
                temperature=0.7, # Adjust for creativity vs factualness
                # max_tokens=4096, # Adjust based on model and expected output size - Often not needed for newer models
            )
            response_content = completion.choices[0].message.content.strip()
            # print(f"--- LLM Call Successful (Attempt {attempt+1}) ---")
            # print(f"Prompt: {prompt[:200]}...") # Log truncated prompt
            # print(f"Response: {response_content[:200]}...") # Log truncated response
            return response_content
        except Exception as e:
            print(f"Error calling OpenRouter (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Failing LLM call.")
                raise  # Re-raise the exception after final attempt

def parse_research_plan(llm_response):
    """
    Parses the LLM response to extract the research plan and keywords.
    Expects a specific format, e.g., JSON or Markdown list.
    Adjust parsing based on the prompt's requested format.
    """
    plan = []
    try:
        # Try parsing as JSON first (robust if prompt asks for JSON)
        data = json.loads(llm_response)
        if isinstance(data, list) and all('step' in item and 'keywords' in item for item in data):
             # Simple heuristic: check for required keys
            for item in data:
                 # Ensure keywords is a list of strings
                keywords_list = item.get('keywords', [])
                if isinstance(keywords_list, str):
                    # Handle comma-separated string if needed
                    keywords_list = [k.strip() for k in keywords_list.split(',') if k.strip()]
                elif not isinstance(keywords_list, list):
                    keywords_list = [] # Default to empty list if format is wrong

                plan.append({
                    "step_description": str(item.get('step', 'N/A')),
                    "keywords": keywords_list
                })
            print("Successfully parsed research plan as JSON.")
            return plan
        else:
             print("Parsed JSON but unexpected structure. Falling back to text parsing.")
             raise ValueError("Unexpected JSON structure")


    except json.JSONDecodeError:
        print("LLM response is not valid JSON. Attempting Markdown/text parsing.")
        # Fallback: Try parsing a numbered list format
        # Example:
        # 1. Step Description 1 [Keywords: keyword1, keyword2]
        # 2. Step Description 2 [Keywords: keyword3, keyword4]
        # This regex is basic and might need adjustment based on actual LLM output
        pattern = re.compile(r"^\s*\d+\.\s*(.*?)\s*\[Keywords?\s*:\s*(.*?)\]", re.MULTILINE | re.IGNORECASE)
        matches = pattern.findall(llm_response)

        if matches:
            for desc, keys_str in matches:
                keys = [k.strip() for k in keys_str.split(',') if k.strip()]
                plan.append({
                    "step_description": desc.strip(),
                    "keywords": keys
                })
            if plan:
                 print("Successfully parsed research plan using regex.")
                 return plan

        # If regex fails, try a simpler line-based approach (less reliable)
        print("Regex parsing failed. Trying simple line parsing.")
        lines = llm_response.strip().split('\n')
        current_step = None
        for line in lines:
            line = line.strip()
            if re.match(r"^\s*\d+\.", line): # Starts with number and dot
                 # Assume description is everything after the number/dot, before keywords marker
                desc_match = re.match(r"^\s*\d+\.\s*(.*)", line)
                if desc_match:
                    current_step = {"step_description": desc_match.group(1).strip(), "keywords": []}
                    # Try to find keywords on the same line or next lines
                    kw_match = re.search(r"\[Keywords?\s*:\s*(.*?)\]", current_step["step_description"], re.IGNORECASE)
                    if kw_match:
                        keys_str = kw_match.group(1)
                        current_step["keywords"] = [k.strip() for k in keys_str.split(',') if k.strip()]
                        # Remove keyword part from description
                        current_step["step_description"] = current_step["step_description"][:kw_match.start()].strip()
                    plan.append(current_step)

            elif current_step and ("Keywords:" in line or "keywords:" in line):
                 keys_str = line.split(":", 1)[-1]
                 current_step["keywords"] = [k.strip() for k in keys_str.split(',') if k.strip()]

        if plan:
            print("Parsed research plan using simple line-based approach.")
            return plan
        else:
            print("Could not parse research plan from LLM response.")
            # Return a default structure indicating failure
            return [{"step_description": "Failed to parse plan", "keywords": []}]


def search_duckduckgo(keywords, max_results=MAX_SEARCH_RESULTS_PER_STEP):
    """Performs a search on DuckDuckGo using the library."""
    query = " ".join(keywords)
    urls = []
    print(f"Searching DDG for: '{query}' (Max results: {max_results})")
    try:
        with DDGS() as ddgs:
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

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style", "nav", "footer", "aside"]):
            script_or_style.decompose()

        # Get text, trying different common content containers
        # Prioritize main content areas if identifiable
        main_content = soup.find('main') or \
                       soup.find('article') or \
                       soup.find('div', attrs={'role': 'main'}) or \
                       soup.find('div', id='content') or \
                       soup.find('div', class_='content') # Add more specific selectors if needed

        if main_content:
             text = main_content.get_text(separator='\n', strip=True)
        else:
             # Fallback to body if no main content found
             body = soup.find('body')
             if body:
                 text = body.get_text(separator='\n', strip=True)
             else:
                 text = soup.get_text(separator='\n', strip=True) # Full fallback


        # Basic cleaning
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)

        if len(cleaned_text) < 100: # Skip pages with very little extracted text
             print(f"Skipping - very little text extracted from: {url}")
             return None

        print(f"Successfully scraped {len(cleaned_text)} characters from: {url}")
        return cleaned_text

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
    except Exception as e:
        print(f"Error parsing URL {url}: {e}")
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

    try:
        # === Step 1: Generate Research Plan & Keywords ===
        print("Step 1: Generating research plan and keywords...")
        plan_prompt = f"""
        Create a concise, step-by-step research plan to investigate the following topic: "{topic}"

        For each step in the plan, provide:
        1. A clear description of the research goal for that step.
        2. A list of 3-5 specific search keywords relevant to achieving that step's goal.

        Format the output as a JSON list of objects, where each object has a 'step' key (string description) and a 'keywords' key (list of strings).
        Example JSON format:
        [
          {{
            "step": "Understand the basic definition and history of [Topic]",
            "keywords": ["define [Topic]", "history of [Topic]", "[Topic] origin", "what is [Topic]"]
          }},
          {{
            "step": "Identify key components or aspects of [Topic]",
            "keywords": ["[Topic] components", "aspects of [Topic]", "key features [Topic]", "[Topic] structure"]
          }},
          {{
            "step": "Explore current applications or examples of [Topic]",
            "keywords": ["[Topic] applications", "[Topic] examples", "uses of [Topic]", "current [Topic] projects"]
          }}
        ]
        Ensure the output is ONLY the valid JSON list, without any introductory text or explanation before or after the JSON structure.
        Replace '[Topic]' with appropriate terms related to "{topic}".
        Keep the plan focused, maybe 3-5 steps maximum unless the topic is very complex.
        """
        plan_response = call_gemini(plan_prompt)
        research_plan = parse_research_plan(plan_response)

        if not research_plan or research_plan[0]["step_description"] == "Failed to parse plan":
             raise ValueError("Failed to generate or parse a valid research plan from the LLM.")
        print(f"Generated Plan: {research_plan}")


        # === Step 2: Search and Scrape ===
        print("\nStep 2: Searching and scraping based on plan...")
        total_urls_scraped = 0
        scraped_data_by_step = {} # Store text associated with the step that found it

        for i, step in enumerate(research_plan):
            step_desc = step['step_description']
            keywords = step['keywords']
            print(f"\n-- Processing Step {i+1}: {step_desc} --")
            print(f"   Keywords: {keywords}")

            if not keywords:
                print("   Skipping step due to missing keywords.")
                plan_details_for_template.append({**step, "urls": []}) # Add to template data
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
                    continue
                if urls_to_scrape_this_step >= MAX_SEARCH_RESULTS_PER_STEP: # Limit per step as well
                    print(f"   Reached scraping limit for this step ({MAX_SEARCH_RESULTS_PER_STEP}).")
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

            plan_details_for_template.append({**step, "urls": step_scraped_urls_list}) # Add step details for template

            print(f"-- Step {i+1} finished. Scraped {len(step_scraped_urls_list)} new URLs for this step. Total URLs scraped: {total_urls_scraped} --")


        if not full_scraped_text:
            raise ValueError("Failed to scrape any relevant content from the web search results.")

        print(f"\nTotal scraped text length: {len(full_scraped_text)} characters.")

        # === Step 3: Extract Relevant Data (Optional but Recommended for large context) ===
        # This step asks the LLM to filter the large scraped text based on the plan.
        # If the context window allows, you could skip this and feed everything to Step 4.
        # However, for very large scraped data, this helps focus the final report generation.
        print("\nStep 3: Asking LLM to extract relevant information from scraped data...")
        extraction_prompt = f"""
        You are a research assistant. Analyze the following scraped web content based on the provided research plan for the topic "{topic}".

        Research Plan:
        {json.dumps(research_plan, indent=2)}

        Scraped Web Content:
        --- START SCRAPED CONTENT ---
        {full_scraped_text[:500000]}
        --- END SCRAPED CONTENT ---
        (Note: Content might be truncated if very large)

        Your task is to go through the scraped content and extract the information that is MOST RELEVANT to answering EACH step of the research plan.
        Synthesize the relevant findings for each step. Discard irrelevant information.
        Present the extracted information clearly, organized by the research plan step it addresses.
        Be concise and focus on the core information related to the plan. If no relevant information is found for a step in the provided text, state that clearly.

        Output Format:
        For each step in the research plan, provide a section starting with "### Step: [Step Description]" followed by the synthesized relevant information found in the text for that step.
        """
        # Limit context size if needed - check model's limit. Gemini 1.5 Pro has a very large limit.
        # A simple truncation is used here as an example. More sophisticated chunking might be needed.
        if len(full_scraped_text) > 700000: # Example limit - adjust based on model!
            print("Warning: Scraped text is very long, truncating for extraction prompt.")
            relevant_info_response = call_gemini(extraction_prompt.replace(full_scraped_text[:500000], full_scraped_text[:700000])) # Crude truncation
        else:
             relevant_info_response = call_gemini(extraction_prompt)

        print("LLM extraction complete.")
        # print(f"Extracted Info Response (preview): {relevant_info_response[:500]}") # Debugging

        # === Step 4: Generate Final Report ===
        print("\nStep 4: Generating the final research report...")
        report_prompt = f"""
        You are a research analyst tasked with creating a comprehensive research report on the topic: "{topic}"

        Use the following inputs to generate the report:
        1.  The Original Research Topic: "{topic}"
        2.  The Research Plan Used:
            {json.dumps(research_plan, indent=2)}
        3.  Synthesized Relevant Information Extracted from Web Research (organized by plan step):
            --- START EXTRACTED INFO ---
            {relevant_info_response}
            --- END EXTRACTED INFO ---

        Instructions for the Report:
        *   Structure the report logically, following the flow of the research plan.
        *   Start with an introduction briefly stating the research topic and the approach (mentioning the plan).
        *   For each step of the research plan:
            *   Clearly state the research step/question.
            *   Present the key findings and information relevant to that step, drawing *only* from the 'Synthesized Relevant Information' provided above.
            *   Explain the findings clearly and concisely. Do not invent information not present in the provided context.
            *   If the extracted information indicates nothing was found for a step, state that.
        *   Include a concluding summary that synthesizes the main findings across all steps.
        *   Format the report using Markdown for clear readability (headings, lists, bold text, etc.).
        *   Ensure the tone is objective and informative.
        *   The final output should be ONLY the Markdown report, starting with a suitable title (e.g., "# Research Report: [Topic]"). Do not include introductory phrases like "Here is the report:".
        """
        final_report_md = call_gemini(report_prompt)
        print("Final report generated by LLM.")

        # Convert Markdown report to HTML for display
        report_html = markdown.markdown(final_report_md, extensions=['fenced_code', 'tables'])

        # --- Render Results ---
        return render_template('results.html',
                               topic=topic,
                               plan_details=plan_details_for_template,
                               scraped_content_length=len(full_scraped_text),
                               raw_scraped_data_preview=full_scraped_text[:5000], # Show a preview
                               report_html=report_html,
                               error=None)

    except ValueError as ve:
         print(f"Value Error during research process: {ve}")
         return render_template('results.html', topic=topic, error=f"A configuration or parsing error occurred: {ve}")
    except requests.exceptions.RequestException as re:
        print(f"Network Error during research process: {re}")
        return render_template('results.html', topic=topic, error=f"A network error occurred during web scraping or API calls: {re}")
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback to console for debugging
        return render_template('results.html', topic=topic, error=f"An unexpected error occurred: {e}")


# --- Run the App ---
if __name__ == '__main__':
    # Set host='0.0.0.0' to make it accessible on your network
    # Use debug=True only for development, not production
    app.run(debug=True, host='127.0.0.1', port=5001)