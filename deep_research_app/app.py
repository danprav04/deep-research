# app.py
import sys
import os
import re
import time
import json
import traceback
from urllib.parse import quote, unquote
import concurrent.futures
# from io import BytesIO # No longer needed for DOCX
from html import escape # For escaping error messages in HTML
import tempfile

from flask import (
    Flask, render_template, request, redirect, url_for, jsonify,
    Response, stream_with_context # send_file no longer needed
)
import google.generativeai as genai

# Import configuration and outsourced modules
import config
from llm_interface import call_gemini, stream_gemini
from web_research import perform_web_search, scrape_url
from utils import (
    parse_research_plan, generate_bibliography_map,
    convert_markdown_to_html # convert_html_to_docx removed
    # DOCX_CONVERSION_AVAILABLE flag removed
)

# --- REMOVED Check for Optional Dependencies (DOCX) ---


# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # Good practice

# --- Initialize Google Generative AI Client ---
if not config.GOOGLE_API_KEY:
     raise ValueError("FATAL: GOOGLE_API_KEY environment variable not set in .env or environment.")
try:
     genai.configure(api_key=config.GOOGLE_API_KEY)
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
    encoded_topic = quote(topic)
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
        scraped_source_metadata_list = []
        temp_files_to_clean = []
        research_plan = []
        accumulated_synthesis_md = ""
        final_report_markdown = ""
        url_to_index_map = {}
        start_time_total = time.time()

        # --- SSE Helper functions ---
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
                    yield "data: {\"type\": \"error\", \"message\": \"Internal server error during SSE event serialization.\"}\n\n"

        def send_progress(message, is_error=False, is_fatal=False):
            event_type = 'error' if is_error else 'progress'
            event_data = {'type': event_type, 'message': message}
            if is_error:
                print(f"{'FATAL ' if is_fatal else ''}ERROR (SSE Stream): {message}")
                event_data['fatal'] = is_fatal
            yield from send_event(event_data)


        # --- Main Research Workflow ---
        try: # <<< START OF MAIN TRY BLOCK >>>
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
                    time.sleep(config.INTER_SEARCH_DELAY_SECONDS)

            search_duration = time.time() - start_search_time
            yield from send_progress(f"Search phase completed in {search_duration:.2f}s.")
            yield from send_progress(f"Collected {len(all_urls_from_search_step)} total unique URLs across all steps ({total_search_errors} search engine errors encountered).")


            # --- Filter URLs ---
            urls_to_scrape_list = []
            yield from send_progress("Filtering URLs for scraping...")
            for url in sorted(list(all_urls_from_search_step)):
                 if len(urls_to_scrape_list) >= config.MAX_TOTAL_URLS_TO_SCRAPE:
                      yield from send_progress(f"  -> Reached URL scraping limit ({config.MAX_TOTAL_URLS_TO_SCRAPE}). Skipping remaining URLs.")
                      break
                 lower_url = url.lower()
                 path_part = lower_url.split('?')[0].split('#')[0]
                 is_file = path_part.endswith(('.pdf', '.jpg', '.png', '.gif', '.zip', '.mp4', '.mp3', '.docx', '.xlsx', '.pptx', '.webp', '.svg', '.xml', '.css', '.js', '.jpeg', '.doc', '.xls', '.ppt', '.txt', '.exe', '.dmg', '.iso', '.rar', '.gz', '.tar', '.bz2', '.7z'))
                 is_mailto = lower_url.startswith('mailto:')
                 is_javascript = lower_url.startswith('javascript:')
                 is_ftp = lower_url.startswith('ftp:')
                 is_tel = lower_url.startswith('tel:')
                 is_local = lower_url.startswith(('file:', 'localhost', '127.0.0.1'))
                 is_valid_http = url.startswith(('http://', 'https://'))

                 if is_valid_http and not any([is_file, is_mailto, is_javascript, is_ftp, is_tel, is_local]):
                      urls_to_scrape_list.append(url)

            yield from send_progress(f"Selected {len(urls_to_scrape_list)} URLs for scraping after filtering (limit was {config.MAX_TOTAL_URLS_TO_SCRAPE}).")

            if not urls_to_scrape_list:
                 yield from send_progress("No suitable URLs found to scrape after search and filtering. Cannot proceed.", is_error=True, is_fatal=True)
                 return

            # === Step 2b: Scrape URLs Concurrently ===
            yield from send_progress(f"Starting concurrent scraping ({config.MAX_WORKERS} workers)...")
            start_scrape_time = time.time()
            scraped_source_metadata_list = []
            temp_files_to_clean = []
            processed_scrape_count = 0
            successful_scrape_count = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                future_to_url = {executor.submit(scrape_url, url): url for url in urls_to_scrape_list}
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    processed_scrape_count += 1
                    try:
                        result_dict = future.result()
                        if result_dict and 'temp_filepath' in result_dict:
                            scraped_source_metadata_list.append(result_dict)
                            temp_files_to_clean.append(result_dict['temp_filepath'])
                            successful_scrape_count += 1
                    except Exception as exc:
                        yield from send_progress(f"    -> Scrape Error for {url[:60]}...: {escape(str(exc))}", is_error=True)

                    if processed_scrape_count % 5 == 0 or processed_scrape_count == len(urls_to_scrape_list):
                          progress_perc = (processed_scrape_count * 100) // len(urls_to_scrape_list)
                          yield from send_progress(f"  -> Scraping Progress: {processed_scrape_count}/{len(urls_to_scrape_list)} URLs processed ({progress_perc}%). Successful scrapes: {successful_scrape_count}")

            scrape_duration = time.time() - start_scrape_time
            yield from send_progress(f"Scraping finished in {scrape_duration:.2f}s. Successfully scraped content from {successful_scrape_count} URLs (saved to temp files).")

            if not scraped_source_metadata_list:
                yield from send_progress("Failed to scrape any content successfully. Cannot proceed with synthesis.", is_error=True, is_fatal=True)
                return

            scraped_url_map = {item['url']: item for item in scraped_source_metadata_list}
            ordered_scraped_metadata_list = [scraped_url_map[url] for url in urls_to_scrape_list if url in scraped_url_map]
            scraped_source_metadata_list = ordered_scraped_metadata_list

            # === Step 3: Generate Bibliography Map ===
            url_to_index_map, bibliography_prompt_list = generate_bibliography_map(scraped_source_metadata_list)
            yield from send_progress(f"Generated bibliography map for {len(url_to_index_map)} successfully scraped sources.")

            # === Step 4: Synthesize Information (Streaming) ===
            yield from send_progress(f"Synthesizing information from scraped content using {config.GOOGLE_MODEL_NAME}...")
            yield from send_event({'type': 'stream_start', 'target': 'synthesis'})

            context_for_llm_structured = []
            current_chars = 0
            sources_included_count = 0
            estimated_total_chars = 0

            yield from send_progress(f"  -> Preparing context for synthesis from {len(scraped_source_metadata_list)} temp files...")

            for source_metadata in scraped_source_metadata_list:
                filepath = source_metadata.get('temp_filepath')
                url = source_metadata.get('url')
                if not filepath or not url or not os.path.exists(filepath):
                     yield from send_progress(f"  -> Warning: Skipping source, missing temp file or metadata for URL {url or 'Unknown'}", is_error=True)
                     continue

                try:
                     file_size = os.path.getsize(filepath)
                     estimated_total_chars += file_size

                     if current_chars + file_size <= config.MAX_CONTEXT_CHARS:
                         with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                             content = f.read()
                         if content:
                              context_for_llm_structured.append({'url': url, 'content': content})
                              current_chars += file_size
                              sources_included_count += 1
                         else:
                              yield from send_progress(f"  -> Warning: Read empty content from temp file {os.path.basename(filepath)} for {url[:60]}...", is_error=True)
                     else:
                          yield from send_progress(f"  -> Context limit ({config.MAX_CONTEXT_CHARS // 1000}k chars) likely reached. Preparing synthesis using first {sources_included_count}/{len(scraped_source_metadata_list)} sources.", is_error=True)
                          break
                except OSError as e:
                     yield from send_progress(f"  -> Error accessing temp file {os.path.basename(filepath)} for {url[:60]}...: {e}", is_error=True)
                except Exception as e:
                     yield from send_progress(f"  -> Unexpected error reading temp file {os.path.basename(filepath)} for {url[:60]}...: {e}", is_error=True)

            estimated_tokens = current_chars / config.CHARS_PER_TOKEN_ESTIMATE
            yield from send_progress(f"  -> Synthesizing based on {sources_included_count} sources (~{current_chars // 1000}k chars / ~{estimated_tokens / 1000:.1f}k estimated tokens).")

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

            accumulated_synthesis_md = ""
            synthesis_stream_error = None
            try:
                stream_generator = stream_gemini(synthesis_prompt)
                for result in stream_generator:
                    if result['type'] == 'chunk':
                        yield from send_event({'type': 'llm_chunk', 'content': result['content'], 'target': 'synthesis'})
                        accumulated_synthesis_md += result['content']
                    elif result['type'] == 'stream_error':
                        synthesis_stream_error = result['message']
                        is_fatal_err = "API key" in synthesis_stream_error or "quota" in synthesis_stream_error.lower() or "resource has been exhausted" in synthesis_stream_error.lower()
                        yield from send_progress(f"LLM stream error during synthesis: {synthesis_stream_error}", is_error=True, is_fatal=is_fatal_err)
                        if is_fatal_err: return
                        break
                    elif result['type'] == 'stream_warning':
                         yield from send_progress(f"LLM Stream Warning (Synthesis): {result['message']}", is_error=True)
                    elif result['type'] == 'stream_end':
                         break
            except Exception as e:
                 yield from send_progress(f"Fatal error processing LLM synthesis stream: {escape(str(e))}", is_error=True, is_fatal=True)
                 traceback.print_exc()
                 return

            yield from send_progress("Synthesis generation finished.")
            if not accumulated_synthesis_md.strip() and not synthesis_stream_error:
                 yield from send_progress("Warning: Synthesis resulted in empty content. The final report might lack detailed findings.", is_error=True)

            # === Step 5: Generate Final Report (Streaming) ===
            yield from send_progress(f"Generating final report using {config.GOOGLE_MODEL_NAME}...")
            yield from send_event({'type': 'stream_start', 'target': 'report'})

            report_prompt_components_base_size = len(topic) + len(json.dumps(research_plan)) + len(bibliography_prompt_list)
            available_chars_for_synthesis = config.MAX_CONTEXT_CHARS - report_prompt_components_base_size - 3000

            if len(accumulated_synthesis_md) > available_chars_for_synthesis:
                yield from send_progress(f"  -> Warning: Accumulated synthesis text ({len(accumulated_synthesis_md)} chars) is potentially too large for report context. Truncating.", is_error=True)
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
            6.  **Formatting:** Use standard Markdown for clarity (headings, lists, bold, italics, paragraphs). Ensure proper spacing between sections and paragraphs. Use bullet points (`*` or `-`) for lists where appropriate.
            7.  **Output:** Generate ONLY the complete Markdown report according to these instructions. Do not include any preliminary remarks, explanations, or text outside the defined report structure.

            Generate the Markdown report now for topic: "{topic}".
            """
            final_report_markdown = ""
            report_stream_error = None
            try:
                stream_generator = stream_gemini(report_prompt)
                for result in stream_generator:
                    if result['type'] == 'chunk':
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
                 final_report_markdown = f"# Research Report: {topic}\n\n*Report generation failed or produced no content. Check logs for details.*"


            # === Step 6: Final Processing and Completion ===
            yield from send_progress("Processing final report for display...")

            # Convert final Markdown to HTML
            # Add print statements to debug the content before/after conversion
            # print(f"DEBUG [app.py]: Final Markdown before conversion (first 500 chars):\n{final_report_markdown[:500]}\n---")
            report_html = convert_markdown_to_html(final_report_markdown)
            # print(f"DEBUG [app.py]: Converted HTML (first 500 chars):\n{report_html[:500]}\n---")

            # Check if conversion returned an error message (starts with <pre><strong>Error...)
            if report_html.strip().lower().startswith(('<pre><strong>error', '<p><em>markdown conversion resulted', '<p><em>report content is empty')):
                yield from send_progress("Error: Failed to convert final Markdown report to HTML for display. See logs for details.", is_error=True)
                # The report_html already contains the error message/fallback from the utility function
            elif not report_html.strip():
                 yield from send_progress("Error: Markdown conversion resulted in empty HTML without specific error message.", is_error=True)
                 report_html = f"<h2>Report Display Error</h2><p>Could not convert report Markdown to HTML. The conversion resulted in empty content.</p><p>Raw Markdown was:</p><pre><code>{escape(final_report_markdown)}</code></pre>"


            # Send final data package to the client
            yield from send_progress("Sending final results to client...")
            final_data = {
                'type': 'complete',
                'report_html': report_html
                # Removed report_markdown, raw_scraped_data_preview, docx_available
            }
            yield from send_event(final_data)

            end_time_total = time.time()
            yield from send_progress(f"Research process completed successfully in {end_time_total - start_time_total:.2f} seconds.")

        except Exception as e:
            # Catch any unexpected errors in the main workflow
            print(f"FATAL: An unexpected error occurred during stream generation:")
            traceback.print_exc() # Print full traceback to server logs
            error_msg = f"Unexpected server error during research: {type(e).__name__} - {escape(str(e))}"
            yield from send_progress(error_msg, is_error=True, is_fatal=True)
        finally:
            # --- Clean up temporary files ---
            if temp_files_to_clean:
                 print(f"INFO: Cleaning up {len(temp_files_to_clean)} temporary scrape files...")
                 cleaned_count = 0
                 failed_count = 0
                 for fpath in temp_files_to_clean:
                     try:
                         if os.path.exists(fpath):
                              os.remove(fpath)
                              cleaned_count += 1
                     except OSError as e:
                         print(f"  Warning: Failed to remove temp file {os.path.basename(fpath)}: {e}")
                         failed_count += 1
                 print(f"INFO: Temp file cleanup complete. Removed: {cleaned_count}, Failed: {failed_count}")

            # Signal that the stream is definitively finished
            yield from send_event({'type': 'stream_terminated'})


    # Set headers for Server-Sent Events
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no', # Important for Nginx buffering issues
        'Connection': 'keep-alive'
    }
    return Response(stream_with_context(generate_updates()), headers=headers)


# --- REMOVED download_docx route ---


# --- Run the App ---
if __name__ == '__main__':
    if not config.GOOGLE_API_KEY:
        print("FATAL ERROR: GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")
        sys.exit(1)
    if not config.GOOGLE_MODEL_NAME:
        print("FATAL ERROR: GOOGLE_MODEL_NAME not found. Please set it in your .env file or environment variables.")
        sys.exit(1)

    print(f"INFO: Starting Flask server...")
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)