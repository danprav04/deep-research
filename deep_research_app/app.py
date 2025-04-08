# app.py
import sys
import os
import re
import time
import json
import traceback
from urllib.parse import quote, unquote
import concurrent.futures
from io import BytesIO
from html import escape # For escaping error messages in HTML

from flask import (
    Flask, render_template, request, redirect, url_for, jsonify,
    Response, stream_with_context, send_file
)
import google.generativeai as genai

# Import configuration and outsourced modules
import config
from llm_interface import call_gemini, stream_gemini
from web_research import perform_web_search, scrape_url
from utils import (
    parse_research_plan, generate_bibliography_map,
    convert_markdown_to_html, convert_html_to_docx, # Ensure all needed utils are imported
    DOCX_CONVERSION_AVAILABLE # Import the flag from utils
)

# --- Check for Optional Dependencies ---
if DOCX_CONVERSION_AVAILABLE:
    print("INFO: 'html2docx' library found. DOCX download will be available.")
else:
    print("WARN: 'html2docx' library not found. DOCX download will be disabled.")
    print("      Install it using: pip install html2docx")


# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # Good practice

# --- Initialize Google Generative AI Client ---
if not config.GOOGLE_API_KEY:
     raise ValueError("FATAL: GOOGLE_API_KEY environment variable not set in .env or environment.")
try:
     genai.configure(api_key=config.GOOGLE_API_KEY)
     # Ensure a model name is loaded, provide a default or raise error if missing
     if not config.GOOGLE_MODEL_NAME:
         raise ValueError("FATAL: GOOGLE_MODEL_NAME not set in .env or environment.")
     print(f"INFO: Google Generative AI configured with model: {config.GOOGLE_MODEL_NAME}")
except Exception as e:
     raise RuntimeError(f"FATAL: Failed to configure Google Generative AI: {e}")


# --- Flask Routes ---

@app.route('/')
def index():
    """Displays the main input form."""
    return render_template('index.html', pico_css=config.PICO_CSS_CDN)

@app.route('/research', methods=['POST'])
def research_start():
    """Redirects to the results page which connects to the SSE stream."""
    topic = request.form.get('topic', '').strip()
    if not topic:
        return redirect(url_for('index'))
    # Encode the topic safely for use in URL query parameters
    encoded_topic = quote(topic)
    # Pass topic, encoded_topic and css link to results.html
    return render_template('results.html',
                           topic=topic,
                           encoded_topic=encoded_topic,
                           pico_css=config.PICO_CSS_CDN)

@app.route('/stream')
def stream():
    """The main SSE route that performs research and streams progress."""
    encoded_topic = request.args.get('topic', '')
    topic = unquote(encoded_topic) # Decode topic from URL
    if not topic:
        topic = "Default Topic - No Topic Provided" # Fallback

    def generate_updates():
        # --- Research state variables ---
        scraped_sources_list = []
        research_plan = []
        accumulated_synthesis_md = "" # Store raw markdown for synthesis
        final_report_markdown = "" # Store raw markdown for final report
        url_to_index_map = {}
        start_time_total = time.time()

        # --- SSE Helper functions ---
        def send_event(data):
            """Safely serializes data and yields an SSE formatted event."""
            try:
                payload = json.dumps(data)
                yield f"data: {payload}\n\n"
            except TypeError as e:
                print(f"Error serializing data for SSE: {e}. Data: {data}")
                try:
                    # Attempt to send a safe error message back
                    safe_data = {'type': data.get('type', 'error'), 'message': f"Serialization Error: {e}"}
                    payload = json.dumps(safe_data)
                    yield f"data: {payload}\n\n"
                except Exception: # Fallback if even error serialization fails
                    yield "data: {\"type\": \"error\", \"message\": \"Internal server error during SSE event serialization.\"}\n\n"

        def send_progress(message, is_error=False, is_fatal=False):
            """Sends a progress or error update event via SSE."""
            event_type = 'error' if is_error else 'progress'
            event_data = {'type': event_type, 'message': message}
            if is_error:
                print(f"{'FATAL ' if is_fatal else ''}ERROR (SSE Stream): {message}")
                event_data['fatal'] = is_fatal # Add fatal flag if applicable
            yield from send_event(event_data)


        # --- Main Research Workflow ---
        try:
            # === Step 1: Generate Research Plan ===
            yield from send_progress(f"Generating research plan for: '{topic}'...")
            plan_prompt = f"""
            Create a detailed, step-by-step research plan with 5-10 distinct steps for the topic: "{topic}"
            Each step should represent a specific question or area of inquiry.
            Format the output STRICTLY as a JSON list of objects within a ```json ... ``` block.
            Each object needs keys "step" (string description) and "keywords" (list of 2-4 search strings).

            Example:
            ```json
            [
              {{"step": "Define core concept", "keywords": ["term definition", "term explanation"]}},
              {{"step": "Explore history", "keywords": ["history of term", "term origins"]}}
            ]
            ```
            Output ONLY the valid JSON list inside the markdown block. Generate 5 to 10 steps for: "{topic}".
            """
            try:
                plan_response = call_gemini(plan_prompt)
                research_plan = parse_research_plan(plan_response)
            except Exception as e:
                 yield from send_progress(f"LLM Error: Failed to generate research plan. Details: {e}", is_error=True, is_fatal=True)
                 return

            # Check if parsing failed or returned the failure indicator
            if not research_plan or (len(research_plan) == 1 and research_plan[0]["step"].startswith("Failed")):
                 fail_reason = research_plan[0]["step"] if research_plan else "Could not parse plan structure."
                 raw_snippet = f" Raw LLM Response Snippet: '{plan_response[:150]}...'" if plan_response else " (LLM Response was empty)"
                 yield from send_progress(f"Failed to create/parse research plan. Reason: {fail_reason}.{raw_snippet}", is_error=True, is_fatal=True)
                 return

            yield from send_progress(f"Generated {len(research_plan)} step plan.")
            for i, step in enumerate(research_plan):
                 yield from send_progress(f"  Step {i+1}: {step['step']} (Keywords: {step.get('keywords', 'N/A')})")


            # === Step 2a: Search and Collect URLs ===
            yield from send_progress("Starting web search...")
            start_search_time = time.time()
            all_urls_from_search_step = set()
            total_search_errors = 0
            total_search_queries = 0

            for i, step in enumerate(research_plan):
                step_desc = step.get('step', f'Unnamed Step {i+1}')
                keywords = step.get('keywords', [])

                yield from send_progress(f"Searching - Step {i+1}/{len(research_plan)}: '{step_desc[:70]}{'...' if len(step_desc)>70 else ''}'")
                if not keywords:
                    yield from send_progress("  -> No keywords provided for this step, skipping search.")
                    continue

                total_search_queries += 1
                step_urls, step_errors = perform_web_search(keywords)

                if step_errors:
                    total_search_errors += len(step_errors)
                    for err in step_errors: yield from send_progress(f"    -> Search Warning: {err}", is_error=True) # Report non-fatal search errors

                new_urls_count = len(set(step_urls) - all_urls_from_search_step)
                all_urls_from_search_step.update(step_urls)
                yield from send_progress(f"  -> Found {len(step_urls)} URLs for step keywords, {new_urls_count} new. Total unique URLs so far: {len(all_urls_from_search_step)}.")

                if i < len(research_plan) - 1:
                    time.sleep(config.INTER_SEARCH_DELAY_SECONDS) # Avoid hitting search rate limits

            search_duration = time.time() - start_search_time
            yield from send_progress(f"Search phase completed in {search_duration:.2f}s.")
            yield from send_progress(f"Collected {len(all_urls_from_search_step)} total unique URLs across all steps ({total_search_errors} search engine errors encountered).")

            # Filter URLs
            urls_to_scrape_list = []
            yield from send_progress("Filtering URLs for scraping...")
            for url in sorted(list(all_urls_from_search_step)):
                 if len(urls_to_scrape_list) >= config.MAX_TOTAL_URLS_TO_SCRAPE:
                      yield from send_progress(f"  -> Reached URL scraping limit ({config.MAX_TOTAL_URLS_TO_SCRAPE}). Skipping remaining URLs.")
                      break
                 # Basic filtering (can be expanded)
                 lower_url = url.lower()
                 path_part = lower_url.split('?')[0].split('#')[0]
                 is_file = path_part.endswith(('.pdf', '.jpg', '.png', '.gif', '.zip', '.mp4', '.mp3', '.docx', '.xlsx', '.pptx', '.webp', '.svg', '.xml', '.css', '.js', '.jpeg', '.doc', '.xls', '.ppt', '.txt', '.exe', '.dmg', '.iso', '.rar', '.gz', '.tar', '.bz2', '.7z'))
                 is_mailto = lower_url.startswith('mailto:')
                 is_javascript = lower_url.startswith('javascript:')
                 is_ftp = lower_url.startswith('ftp:')
                 is_tel = lower_url.startswith('tel:')
                 is_local = lower_url.startswith(('file:', 'localhost', '127.0.0.1')) # Avoid local paths
                 is_valid_http = url.startswith(('http://', 'https://'))

                 if is_valid_http and not any([is_file, is_mailto, is_javascript, is_ftp, is_tel, is_local]):
                      urls_to_scrape_list.append(url)
                      # yield from send_progress(f"  -> Keeping URL: {url[:80]}...") # Can be noisy
                 # else:
                 #     yield from send_progress(f"  -> Filtering out: {url[:80]}...") # Can be noisy

            yield from send_progress(f"Selected {len(urls_to_scrape_list)} URLs for scraping after filtering (limit was {config.MAX_TOTAL_URLS_TO_SCRAPE}).")

            if not urls_to_scrape_list:
                 yield from send_progress("No suitable URLs found to scrape after search and filtering. Cannot proceed.", is_error=True, is_fatal=True)
                 return


            # === Step 2b: Scrape URLs Concurrently ===
            yield from send_progress(f"Starting concurrent scraping ({config.MAX_WORKERS} workers)...")
            start_scrape_time = time.time()
            scraped_sources_list = []
            processed_scrape_count = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                future_to_url = {executor.submit(scrape_url, url): url for url in urls_to_scrape_list}
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    processed_scrape_count += 1
                    try:
                        result_dict = future.result() # Result is {'url': url, 'content': text} or None
                        if result_dict:
                            scraped_sources_list.append(result_dict)
                            # yield from send_progress(f"    -> Success: Scraped {url[:60]}...") # Noisy
                        # else: result was None (filtered by scrape_url)
                    except Exception as exc:
                        yield from send_progress(f"    -> Scrape Error for {url[:60]}...: {escape(str(exc))}", is_error=True) # Log specific future error

                    # Send progress update periodically
                    if processed_scrape_count % 5 == 0 or processed_scrape_count == len(urls_to_scrape_list):
                          progress_perc = (processed_scrape_count * 100) // len(urls_to_scrape_list)
                          yield from send_progress(f"  -> Scraping Progress: {processed_scrape_count}/{len(urls_to_scrape_list)} URLs processed ({progress_perc}%). Successful scrapes: {len(scraped_sources_list)}")

            scrape_duration = time.time() - start_scrape_time
            yield from send_progress(f"Scraping finished in {scrape_duration:.2f}s. Successfully scraped content from {len(scraped_sources_list)} out of {len(urls_to_scrape_list)} selected URLs.")

            if not scraped_sources_list:
                yield from send_progress("Failed to scrape any content successfully. Cannot proceed with synthesis.", is_error=True, is_fatal=True)
                return

            # Ensure order matches original scrape list if needed (useful for consistent bibliography)
            scraped_url_map = {item['url']: item for item in scraped_sources_list}
            ordered_scraped_list = [scraped_url_map[url] for url in urls_to_scrape_list if url in scraped_url_map]
            scraped_sources_list = ordered_scraped_list


            # === Step 3: Generate Bibliography Map ===
            url_to_index_map, bibliography_prompt_list = generate_bibliography_map(scraped_sources_list)
            yield from send_progress(f"Generated bibliography map for {len(url_to_index_map)} successfully scraped sources.")


            # === Step 4: Synthesize Information (Streaming) ===
            yield from send_progress(f"Synthesizing information from scraped content using {config.GOOGLE_MODEL_NAME}...")
            yield from send_event({'type': 'stream_start', 'target': 'synthesis'}) # Signal UI to start displaying synthesis stream

            # Prepare context, respecting estimated token limits
            context_for_llm_structured = []
            current_chars = 0
            sources_included_count = 0
            for source in scraped_sources_list:
                 # Estimate size: URL length + content length + some JSON overhead
                 source_len = len(source.get('url', '')) + len(source.get('content', '')) + 50
                 if current_chars + source_len <= config.MAX_CONTEXT_CHARS:
                     context_for_llm_structured.append(source)
                     current_chars += source_len
                     sources_included_count += 1
                 else:
                     yield from send_progress(f"  -> Warning: Context limit ({config.MAX_CONTEXT_CHARS // 1000}k chars) reached during synthesis prep. Using first {sources_included_count}/{len(scraped_sources_list)} sources.", is_error=True)
                     break

            estimated_tokens = current_chars / config.CHARS_PER_TOKEN_ESTIMATE
            yield from send_progress(f"  -> Synthesizing based on {len(context_for_llm_structured)} sources (~{current_chars // 1000}k chars / ~{estimated_tokens / 1000:.1f}k estimated tokens).")

            synthesis_prompt = f"""
            Analyze the provided web content about "{topic}" based on the research plan.
            Synthesize key information relevant to each plan step, citing sources accurately using ONLY the provided URLs.

            Research Topic: {topic}
            Research Plan:
            ```json
            {json.dumps(research_plan, indent=2)}
            ```
            Source Content ({len(context_for_llm_structured)} sources):
            ```json
            {json.dumps(context_for_llm_structured, indent=2, ensure_ascii=False)}
            ```
            Instructions:
            1. Address each plan step sequentially.
            2. Find relevant information for each step across ALL provided sources.
            3. Synthesize findings concisely in **Markdown format**. Use headings (`### Step X:`), lists, bold, italics etc. for clarity.
            4. **MANDATORY CITATION**: Immediately after ANY information derived from a specific source, cite it using the exact format: `[Source URL: <full_url_here>]`. Cite every distinct piece of information derived from a source. Use the full URL provided in the Source Content section.
            5. If no relevant information is found for a specific plan step in the provided sources, state clearly under the step heading: "No specific information found for this step in the provided sources."
            6. Structure the output strictly by plan steps. Start each step's synthesis with `### Step X: <Step Description from Plan>`. Use `---` as a separator ONLY between the synthesis for different plan steps.
            7. Output ONLY the synthesized Markdown content structured by plan steps. Do NOT include an introduction, conclusion, summary, or bibliography in this output.
            """
            accumulated_synthesis_md = "" # Store raw markdown from synthesis stream
            synthesis_stream_error = None
            try:
                stream_generator = stream_gemini(synthesis_prompt)
                for result in stream_generator:
                    if result['type'] == 'chunk':
                        # Send raw markdown chunk for intermediate display in UI
                        yield from send_event({'type': 'llm_chunk', 'content': result['content'], 'target': 'synthesis'})
                        accumulated_synthesis_md += result['content']
                    elif result['type'] == 'stream_error':
                        synthesis_stream_error = result['message']
                        # Determine if error is fatal (e.g., API key, quota)
                        is_fatal_err = "API key" in synthesis_stream_error or "quota" in synthesis_stream_error.lower() or "resource has been exhausted" in synthesis_stream_error.lower()
                        yield from send_progress(f"LLM stream error during synthesis: {synthesis_stream_error}", is_error=True, is_fatal=is_fatal_err)
                        if is_fatal_err: return # Stop if fatal API key/quota error
                        break # Break loop on non-fatal stream error, report generation will proceed but may lack synthesis
                    elif result['type'] == 'stream_warning':
                         yield from send_progress(f"LLM Stream Warning (Synthesis): {result['message']}", is_error=True) # Log warnings
                    elif result['type'] == 'stream_end':
                         break # Normal stream end
            except Exception as e:
                 yield from send_progress(f"Fatal error processing LLM synthesis stream: {escape(str(e))}", is_error=True, is_fatal=True)
                 traceback.print_exc()
                 return # Stop processing

            yield from send_progress("Synthesis generation finished.")
            if not accumulated_synthesis_md.strip() and not synthesis_stream_error:
                 yield from send_progress("Warning: Synthesis resulted in empty content. The final report might lack detailed findings.", is_error=True)


            # === Step 5: Generate Final Report (Streaming) ===
            yield from send_progress(f"Generating final report using {config.GOOGLE_MODEL_NAME}...")
            yield from send_event({'type': 'stream_start', 'target': 'report'}) # Signal UI for report stream

            # Prepare inputs, handle potential truncation if synthesis is very large
            report_prompt_components_base_size = len(topic) + len(json.dumps(research_plan)) + len(bibliography_prompt_list)
            # Add buffer for prompt instructions and formatting overhead
            available_chars_for_synthesis = config.MAX_CONTEXT_CHARS - report_prompt_components_base_size - 3000 # Increased safety buffer

            if len(accumulated_synthesis_md) > available_chars_for_synthesis:
                yield from send_progress(f"  -> Warning: Accumulated synthesis text ({len(accumulated_synthesis_md)} chars) is potentially too large for the report generation context limit. Truncating synthesis input.", is_error=True)
                # Truncate from the end to preserve earlier steps if possible
                truncated_synthesis_md = accumulated_synthesis_md[:available_chars_for_synthesis] + "\n\n... [Synthesis truncated due to context limits]"
            else:
                truncated_synthesis_md = accumulated_synthesis_md

            report_prompt = f"""
            Create a comprehensive research report in Markdown format on the topic: "{topic}".

            You are provided with the following inputs:
            1. The Original Research Plan:
               ```json
               {json.dumps(research_plan, indent=2)}
               ```
            2. Synthesized Information (in Markdown, potentially incomplete, with raw URL citations `[Source URL: ...]`:
               ```markdown
               {truncated_synthesis_md if truncated_synthesis_md.strip() else "No synthesized information was generated or provided."}
               ```
            3. A Bibliography Map (mapping URLs to citation numbers):
               ```
               {bibliography_prompt_list if bibliography_prompt_list else "No sources available for bibliography."}
               ```

            **Instructions for Generating the Final Markdown Report:**

            1.  **Structure:** Create the report with the following sections using Markdown headings:
                *   `# Research Report: {topic}` (Main Title)
                *   `## Introduction`
                *   `## Findings` (This section will contain subsections for each plan step)
                *   `## Conclusion`
                *   `## Bibliography`
            2.  **Introduction:** Briefly introduce the research topic "{topic}". State the purpose of the report (to synthesize information based on the plan and sources). List or briefly describe the key steps from the Research Plan.
            3.  **Findings:**
                *   Under the `## Findings` heading, create a subsection for each step in the original Research Plan using `### Step X: <Step Description>`.
                *   For each step's subsection, integrate the relevant information from the "Synthesized Information" provided above. Rephrase and structure the information clearly.
                *   **Crucially, perform citation replacement:** Find every instance of a raw URL citation `[Source URL: <full_url_here>]` within the Synthesized Information you are using for this step. Replace **each** instance with the corresponding Markdown footnote reference `[^N]`, where `N` is the number associated with `<full_url_here>` in the provided Bibliography Map.
                *   **Handling Missing Citations:** If a URL inside `[Source URL: ...]` tag from the synthesis is NOT found as a key in the Bibliography Map, **OMIT** the footnote marker `[^N]` for that specific instance. Do not invent numbers.
                *   If the "Synthesized Information" section was empty, or if no relevant synthesis was provided for a specific plan step, clearly state this under the relevant `### Step X:` heading (e.g., "No specific findings were synthesized for this step based on the available sources.").
            4.  **Conclusion:** Summarize the key findings presented in the report (or note the lack thereof). Briefly mention any limitations encountered (e.g., limited number of sources successfully scraped, gaps in information found, potential synthesis truncation). Suggest potential areas for further research if appropriate.
            5.  **Bibliography:**
                *   Under the `## Bibliography` heading, list all the sources from the Bibliography Map.
                *   Use the standard Markdown footnote definition format: `[^N]: <full_url_here>`
                *   Each footnote definition **must** be on its own line.
                *   Ensure the numbers `N` match those used in the Findings section.
                *   If the Bibliography Map was empty, simply state "No sources were cited in this report." under the heading.
            6.  **Formatting:** Use standard Markdown for clarity (headings, lists, bold, italics, paragraphs). Ensure proper spacing between sections and paragraphs.
            7.  **Output:** Generate ONLY the complete Markdown report according to these instructions. Do not include any preliminary remarks, explanations, or text outside the defined report structure.

            Generate the Markdown report now for topic: "{topic}".
            """

            final_report_markdown = "" # Store the final markdown report stream
            report_stream_error = None
            try:
                stream_generator = stream_gemini(report_prompt)
                for result in stream_generator:
                    if result['type'] == 'chunk':
                         # Send raw markdown chunk for intermediate display in UI
                        yield from send_event({'type': 'llm_chunk', 'content': result['content'], 'target': 'report'})
                        final_report_markdown += result['content'] # Accumulate raw markdown
                    elif result['type'] == 'stream_error':
                        report_stream_error = result['message']
                        is_fatal_err = "API key" in report_stream_error or "quota" in report_stream_error.lower() or "resource has been exhausted" in report_stream_error.lower()
                        yield from send_progress(f"LLM stream error during report generation: {report_stream_error}", is_error=True, is_fatal=is_fatal_err)
                        if is_fatal_err: return
                        break # Break on non-fatal stream error
                    elif result['type'] == 'stream_warning':
                         yield from send_progress(f"LLM Stream Warning (Report): {result['message']}", is_error=True)
                    elif result['type'] == 'stream_end':
                         break # Normal end
            except Exception as e:
                 yield from send_progress(f"Fatal error processing LLM report stream: {escape(str(e))}", is_error=True, is_fatal=True)
                 traceback.print_exc()
                 return

            yield from send_progress("Report generation finished.")
            if not final_report_markdown.strip() and not report_stream_error:
                 yield from send_progress("Warning: Final report generation resulted in empty content.", is_error=True)
                 # Create a placeholder if completely empty to avoid errors downstream
                 final_report_markdown = f"# Research Report: {topic}\n\n*Report generation failed or produced no content. Check logs for details.*"


            # === Step 6: Final Processing and Completion ===
            yield from send_progress("Processing final report for display and download...")

            # Convert final Markdown to HTML using the utility function
            # This handles footnote syntax conversion correctly
            report_html = convert_markdown_to_html(final_report_markdown)
            if report_html.startswith("<pre>Error during Markdown conversion"):
                yield from send_progress("Error: Failed to convert final Markdown report to HTML for display.", is_error=True)
                # Send the raw markdown anyway, wrapped in pre, so something shows up
                report_html = f"<h2>Report Display Error</h2><p>Could not convert report Markdown to HTML. Displaying raw Markdown:</p><pre><code>{escape(final_report_markdown)}</code></pre>"


            # Prepare raw data preview (limited size)
            preview_limit_chars = 3000
            raw_data_preview_list = []
            current_preview_len = 0
            yield from send_progress("Preparing raw scraped data preview...")
            for src in scraped_sources_list:
                # Use compact JSON for preview to save space, handle potential errors
                try:
                    src_dump = json.dumps(src, separators=(',', ':'), ensure_ascii=False)
                except Exception as json_err:
                    src_dump = f'{{"error": "Could not serialize source", "url": "{src.get("url", "N/A")}", "details": "{escape(str(json_err))}"}}'

                if current_preview_len + len(src_dump) < preview_limit_chars:
                    raw_data_preview_list.append(src_dump)
                    current_preview_len += len(src_dump)
                else:
                    break # Stop adding if preview limit exceeded
            # Format the preview nicely
            if raw_data_preview_list:
                 raw_data_preview = "[\n  " + ",\n  ".join(raw_data_preview_list) + "\n]"
            else:
                 raw_data_preview = "[]"

            if len(scraped_sources_list) > len(raw_data_preview_list):
                 raw_data_preview += f"\n\n... ({len(raw_data_preview_list)} out of {len(scraped_sources_list)} sources shown due to preview size limit of {preview_limit_chars} chars)"
            elif not scraped_sources_list:
                 raw_data_preview = "[] (No sources were successfully scraped or retained)"
            else:
                raw_data_preview += f"\n\n({len(raw_data_preview_list)} sources shown)"


            # Send final data package to the client
            yield from send_progress("Sending final results to client...")
            final_data = {
                'type': 'complete',
                'report_html': report_html, # Send the converted HTML for display
                'report_markdown': final_report_markdown, # Send raw markdown for copy/download base
                'raw_scraped_data_preview': raw_data_preview,
                'docx_available': DOCX_CONVERSION_AVAILABLE # Use the imported flag
            }
            yield from send_event(final_data)

            end_time_total = time.time()
            yield from send_progress(f"Research process completed successfully in {end_time_total - start_time_total:.2f} seconds.")

        except Exception as e:
            # Catch any unexpected errors in the main workflow
            print(f"FATAL: An unexpected error occurred during stream generation:")
            traceback.print_exc() # Print full traceback to server logs
            error_msg = f"Unexpected server error during research: {type(e).__name__} - {escape(str(e))}"
            # Use the send_progress helper for errors
            yield from send_progress(error_msg, is_error=True, is_fatal=True)
        finally:
             # Signal that the stream is definitively finished, regardless of success/error
             # This helps the client know when to stop waiting even if 'complete' wasn't sent
             yield from send_event({'type': 'stream_terminated'})


    # Set headers for Server-Sent Events
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no', # Important for Nginx buffering issues
        'Connection': 'keep-alive'
    }
    return Response(stream_with_context(generate_updates()), headers=headers)


@app.route('/download_docx', methods=['POST'])
def download_docx():
    """Converts the received Markdown report to DOCX and sends it as a download."""
    if not DOCX_CONVERSION_AVAILABLE: # Use the imported flag
        print("Error [/download_docx]: Attempted download when DOCX conversion is unavailable.")
        # Return a user-friendly JSON error
        return jsonify({"success": False, "message": "DOCX download failed: The required 'html2docx' library is not installed or loaded on the server."}), 400

    markdown_content = request.form.get('markdown_report') # Get raw markdown from hidden input
    topic = request.form.get('topic', 'Research_Report') # Get topic for filename

    if not markdown_content:
        print("Error [/download_docx]: No Markdown content received in the request.")
        return jsonify({"success": False, "message": "Error: No report content was received for DOCX conversion."}), 400

    try:
        # 1. Convert Markdown to HTML *first* using the same utility as for display
        # This ensures footnotes and other Markdown features are handled correctly
        print("Info [/download_docx]: Converting Markdown to HTML for DOCX generation...")
        report_html = convert_markdown_to_html(markdown_content)
        if not report_html or report_html.startswith("<pre>Error during Markdown conversion"):
             print("Error [/download_docx]: Failed to convert provided Markdown to HTML before DOCX conversion.")
             # Extract error message if possible
             error_detail = "Markdown conversion failed."
             if report_html.startswith("<pre>"):
                 match = re.search(r"Error during Markdown conversion: (.*?)\n", report_html)
                 if match: error_detail = match.group(1)
             return jsonify({"success": False, "message": f"Error: Could not prepare report content for DOCX conversion. {error_detail}"}), 500

        # 2. Convert the resulting HTML to DOCX in memory buffer
        print("Info [/download_docx]: Converting HTML to DOCX...")
        docx_buffer = convert_html_to_docx(report_html) # Pass the generated HTML
        if docx_buffer is None:
            # convert_html_to_docx should have printed details
            print("Error [/download_docx]: The convert_html_to_docx utility returned None.")
            return jsonify({"success": False, "message": "Error: Failed during the HTML to DOCX conversion process on the server."}), 500

        # 3. Prepare filename
        # Sanitize topic for filename: remove invalid chars, replace spaces
        print("Info [/download_docx]: Preparing filename...")
        safe_filename_topic = re.sub(r'[^\w\s-]', '', topic).strip()
        safe_filename_topic = re.sub(r'[-\s]+', '_', safe_filename_topic)
        # Truncate if too long to prevent filesystem issues
        filename_base = f"{safe_filename_topic}_Research_Report"
        filename = f"{filename_base[:config.DOWNLOAD_FILENAME_MAX_LENGTH]}.docx"
        print(f"Info [/download_docx]: Sending file as '{filename}'")

        # 4. Send the buffer as a file download
        return send_file(
            docx_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

    except ImportError: # Should be caught by initial check, but as fallback
         print("Error [/download_docx]: html2docx library missing during conversion attempt (ImportError).")
         return jsonify({"success": False, "message": "Internal Server Error: DOCX conversion library is unexpectedly missing."}), 500
    except Exception as e:
        print(f"ERROR [/download_docx]: An unexpected error occurred during DOCX download preparation:")
        traceback.print_exc()
        # Provide a generic but informative error to the user
        msg = f"An unexpected error occurred on the server during DOCX conversion: {escape(str(e))}"
        return jsonify({"success": False, "message": msg}), 500


# --- Run the App ---
if __name__ == '__main__':
    # Final check before running
    if not config.GOOGLE_API_KEY:
        print("FATAL ERROR: GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")
        sys.exit(1)
    if not config.GOOGLE_MODEL_NAME:
        print("FATAL ERROR: GOOGLE_MODEL_NAME not found. Please set it in your .env file or environment variables.")
        sys.exit(1)

    print(f"INFO: Starting Flask server...")
    # Use threaded=True for handling multiple requests concurrently, especially during SSE
    # debug=False for production/demonstration. Set debug=True for development (enables auto-reloading).
    # Use host='0.0.0.0' to make the server accessible on your network.
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
    # For production deployment, consider using a proper WSGI server like Gunicorn or Waitress:
    # Example: gunicorn --workers 4 --bind 0.0.0.0:5001 app:app
    # Example: waitress-serve --host 0.0.0.0 --port 5001 app:app