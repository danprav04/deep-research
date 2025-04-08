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
    convert_markdown_to_html, convert_html_to_docx
)

# --- Check for Optional Dependencies ---
DOCX_CONVERSION_AVAILABLE = False
try:
    import html2docx
    DOCX_CONVERSION_AVAILABLE = True
    print("INFO: 'html2docx' library found. DOCX download will be available.")
except ImportError:
    print("WARN: 'html2docx' library not found. DOCX download will be disabled.")
    print("      Install it using: pip install html2docx")


# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # Used for session management if needed, good practice

# --- Initialize Google Generative AI Client ---
if not config.GOOGLE_API_KEY:
    raise ValueError("FATAL: GOOGLE_API_KEY environment variable not set in .env or environment.")
try:
    genai.configure(api_key=config.GOOGLE_API_KEY)
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
        # Handle empty topic submission gracefully (e.g., redirect back with message)
        # For simplicity, redirecting to index. Could add flash message.
        return redirect(url_for('index'))
    # Encode the topic safely for use in URL query parameters
    encoded_topic = quote(topic)
    return render_template('results.html', topic=topic, encoded_topic=encoded_topic, pico_css=config.PICO_CSS_CDN)

@app.route('/stream')
def stream():
    """The main SSE route that performs research and streams progress."""
    encoded_topic = request.args.get('topic', '')
    topic = unquote(encoded_topic) # Decode topic from URL
    if not topic:
        topic = "Default Topic - No Topic Provided" # Fallback if decoding fails or empty

    def generate_updates():
        # --- Research state variables ---
        scraped_sources_list = []
        research_plan = []
        accumulated_synthesis = ""
        final_report_markdown = ""
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
                # Attempt to send a safe error message back
                try:
                    safe_data = {'type': data.get('type', 'error'), 'message': f"Serialization Error: {e}"}
                    payload = json.dumps(safe_data)
                    yield f"data: {payload}\n\n"
                except Exception: # Fallback if even error serialization fails
                    yield "data: {\"type\": \"error\", \"message\": \"Internal server error during SSE event serialization.\"}\n\n"

        def send_progress(message):
            """Sends a progress update event."""
            yield from send_event({'type': 'progress', 'message': message})

        def send_error_event(message, is_fatal=False):
            """Sends an error event and prints to server log."""
            print(f"ERROR (SSE Stream): {message}")
            yield from send_event({'type': 'error', 'message': message, 'fatal': is_fatal})

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
            except Exception as e:
                 yield from send_error_event(f"Failed to generate research plan from LLM: {e}", is_fatal=True)
                 return # Stop processing if plan generation fails

            research_plan = parse_research_plan(plan_response)

            # Check if parsing failed or returned the failure indicator
            if not research_plan or (len(research_plan) == 1 and research_plan[0]["step_description"].startswith("Failed")):
                 fail_reason = research_plan[0]["step_description"] if research_plan else "Could not parse plan."
                 raw_snippet = f" Raw LLM Response Snippet: '{plan_response[:150]}...'" if plan_response else " (LLM Response was empty)"
                 yield from send_error_event(f"Failed to create/parse research plan. Reason: {fail_reason}.{raw_snippet}", is_fatal=True)
                 return

            yield from send_progress(f"Generated {len(research_plan)} step plan.")
            for i, step in enumerate(research_plan):
                 yield from send_progress(f"  Step {i+1}: {step['step_description']} (Keywords: {step['keywords']})")


            # === Step 2a: Search and Collect URLs ===
            yield from send_progress("Starting web search...")
            start_search_time = time.time()
            all_urls_from_search_step = set()
            total_search_errors = 0
            total_search_queries = 0

            for i, step in enumerate(research_plan):
                step_desc = step.get('step_description', f'Unnamed Step {i+1}')
                keywords = step.get('keywords', [])

                yield from send_progress(f"Searching - Step {i+1}/{len(research_plan)}: '{step_desc[:70]}{'...' if len(step_desc)>70 else ''}'")
                if not keywords:
                    yield from send_progress("  -> No keywords, skipping search.")
                    continue

                total_search_queries += 1
                step_urls, step_errors = perform_web_search(keywords)

                if step_errors:
                    total_search_errors += len(step_errors)
                    for err in step_errors: yield from send_progress(f"    -> Search Warning: {err}") # Report non-fatal search errors

                new_urls_count = len(set(step_urls) - all_urls_from_search_step)
                all_urls_from_search_step.update(step_urls)
                yield from send_progress(f"  -> Found {len(step_urls)} URLs for step, {new_urls_count} new. Total unique: {len(all_urls_from_search_step)}.")

                if i < len(research_plan) - 1:
                    time.sleep(config.INTER_SEARCH_DELAY_SECONDS) # Delay between step searches

            search_duration = time.time() - start_search_time
            yield from send_progress(f"Search phase completed in {search_duration:.2f}s.")
            yield from send_progress(f"Found {len(all_urls_from_search_step)} total unique URLs ({total_search_errors} engine errors).")

            # Filter URLs
            urls_to_scrape_list = []
            for url in sorted(list(all_urls_from_search_step)):
                 if len(urls_to_scrape_list) >= config.MAX_TOTAL_URLS_TO_SCRAPE:
                      yield from send_progress(f"  -> Reached URL limit ({config.MAX_TOTAL_URLS_TO_SCRAPE}).")
                      break
                 # Basic filtering (can be expanded)
                 is_file = url.lower().split('?')[0].split('#')[0].endswith(('.pdf', '.jpg', '.png', '.gif', '.zip', '.mp4', '.mp3', '.docx', '.xlsx', '.pptx', '.webp', '.svg', '.xml', '.css', '.js', '.jpeg', '.doc', '.xls', '.ppt', '.txt', '.exe', '.dmg', '.iso', '.rar'))
                 is_mailto = url.lower().startswith('mailto:')
                 is_javascript = url.lower().startswith('javascript:')
                 is_ftp = url.lower().startswith('ftp:')
                 is_tel = url.lower().startswith('tel:')
                 is_local = url.lower().startswith(('file:', 'localhost', '127.0.0.1')) # Avoid local paths
                 is_valid_http = url.startswith(('http://', 'https://'))

                 if is_valid_http and not any([is_file, is_mailto, is_javascript, is_ftp, is_tel, is_local]):
                      urls_to_scrape_list.append(url)

            yield from send_progress(f"Selected {len(urls_to_scrape_list)} URLs for scraping (limit: {config.MAX_TOTAL_URLS_TO_SCRAPE}).")

            if not urls_to_scrape_list:
                 yield from send_error_event("No suitable URLs found to scrape after search and filtering.", is_fatal=True)
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
                    except Exception as exc:
                        yield from send_progress(f"    -> Scrape Error for {url[:60]}...: {exc}") # Log specific future error

                    # Send progress update periodically
                    if processed_scrape_count % 5 == 0 or processed_scrape_count == len(urls_to_scrape_list):
                          progress_perc = (processed_scrape_count * 100) // len(urls_to_scrape_list)
                          yield from send_progress(f"  -> Scraping: {processed_scrape_count}/{len(urls_to_scrape_list)} ({progress_perc}%). Success: {len(scraped_sources_list)}")

            scrape_duration = time.time() - start_scrape_time
            yield from send_progress(f"Scraping finished in {scrape_duration:.2f}s. Successfully scraped {len(scraped_sources_list)}/{len(urls_to_scrape_list)} URLs.")

            if not scraped_sources_list:
                yield from send_error_event("Failed to scrape any content successfully.", is_fatal=True)
                return

            # Ensure order matches original scrape list if needed (useful for consistent bibliography)
            scraped_url_map = {item['url']: item for item in scraped_sources_list}
            ordered_scraped_list = [scraped_url_map[url] for url in urls_to_scrape_list if url in scraped_url_map]
            scraped_sources_list = ordered_scraped_list


            # === Step 3: Generate Bibliography Map ===
            url_to_index_map, bibliography_prompt_list = generate_bibliography_map(scraped_sources_list)
            yield from send_progress(f"Generated bibliography map for {len(url_to_index_map)} sources.")


            # === Step 4: Synthesize Information (Streaming) ===
            yield from send_progress(f"Synthesizing information using {config.GOOGLE_MODEL_NAME}...")
            yield from send_event({'type': 'stream_start', 'target': 'synthesis'}) # Signal UI to start displaying synthesis stream

            # Prepare context, respecting estimated token limits
            context_for_llm_structured = []
            current_chars = 0
            sources_included_count = 0
            for source in scraped_sources_list:
                 # Estimate size: URL length + content length + some overhead
                 source_len = len(source.get('url', '')) + len(source.get('content', '')) + 50
                 if current_chars + source_len <= config.MAX_CONTEXT_CHARS:
                     context_for_llm_structured.append(source)
                     current_chars += source_len
                     sources_included_count += 1
                 else:
                     yield from send_progress(f"  -> Warning: Context limit reached. Using first {sources_included_count}/{len(scraped_sources_list)} sources for synthesis.")
                     break

            estimated_tokens = current_chars / config.CHARS_PER_TOKEN_ESTIMATE
            yield from send_progress(f"  -> Synthesizing based on {len(context_for_llm_structured)} sources (~{current_chars // 1000}k chars / ~{estimated_tokens // 1000}k tokens).")

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
            2. Find relevant information for each step across ALL sources.
            3. Synthesize findings concisely.
            4. **MANDATORY CITATION**: Immediately after ANY information from a source, cite it using the exact format: `[Source URL: <full_url_here>]`. Cite every distinct piece of info.
            5. If no info found for a step, state: "No specific information found for this step."
            6. Use Markdown for structure (`### Step X: ...`).
            7. Output ONLY the synthesis, structured by plan steps. NO intro/conclusion/summary/bibliography here. Separate steps with `---`.
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
                        # Send error, but don't make it fatal unless it's API key/quota
                        is_fatal_synthesis_err = "API key" in synthesis_stream_error or "quota" in synthesis_stream_error.lower()
                        yield from send_error_event(f"LLM stream error during synthesis: {synthesis_stream_error}", is_fatal=is_fatal_synthesis_err)
                        if is_fatal_synthesis_err: return # Stop if fatal API key/quota error
                        break # Break loop on non-fatal stream error, report generation will proceed
                    elif result['type'] == 'stream_warning':
                         yield from send_progress(f"LLM Stream Warning (Synthesis): {result['message']}") # Log warnings
                    elif result['type'] == 'stream_end':
                         break # Normal stream end
            except Exception as e:
                 yield from send_error_event(f"Fatal error processing LLM synthesis stream: {e}", is_fatal=True)
                 traceback.print_exc()
                 return # Stop processing

            yield from send_progress("Synthesis generation finished.")
            if not accumulated_synthesis.strip() and not synthesis_stream_error:
                 yield from send_progress("Warning: Synthesis resulted in empty content.")


            # === Step 5: Generate Final Report (Streaming) ===
            yield from send_progress(f"Generating final report using {config.GOOGLE_MODEL_NAME}...")
            yield from send_event({'type': 'stream_start', 'target': 'report'}) # Signal UI for report stream

            # Prepare inputs, handle potential truncation if synthesis is very large
            report_prompt_components_base_size = len(topic) + len(json.dumps(research_plan)) + len(bibliography_prompt_list)
            available_chars_for_synthesis = config.MAX_CONTEXT_CHARS - report_prompt_components_base_size - 2000 # Safety buffer

            if len(accumulated_synthesis) > available_chars_for_synthesis:
                yield from send_progress(f"  -> Warning: Synthesis text potentially too large for report prompt, truncating.")
                truncated_synthesis = accumulated_synthesis[:available_chars_for_synthesis] + "\n\n... [Synthesis truncated due to context limits]"
            else:
                truncated_synthesis = accumulated_synthesis

            report_prompt = f"""
            Create a comprehensive Markdown research report on "{topic}".

            Inputs:
            1. Research Plan:
               ```json
               {json.dumps(research_plan, indent=2)}
               ```
            2. Synthesized Information (with raw URL citations):
               ```markdown
               {truncated_synthesis if truncated_synthesis.strip() else "No synthesized information was generated or provided."}
               ```
            3. Bibliography Map (URL -> Reference Number):
               ```
               {bibliography_prompt_list if bibliography_prompt_list else "No sources available for bibliography."}
               ```

            Instructions:
            1. Write a final report in Markdown format with sections: `# Research Report: {topic}`, `## Introduction`, `## Findings`, `## Conclusion`, `## Bibliography`.
            2. **Introduction**: Introduce "{topic}", state the report's purpose, and briefly outline the research plan steps.
            3. **Findings**: Organize by plan step (`### Step X: <Description>`). Integrate the synthesized information for each step. If synthesis was empty or missing for a step, state that clearly.
            4. **CRITICAL CITATION REPLACEMENT**: Find EVERY occurrence of `[Source URL: <full_url_here>]` in the synthesized text. Replace it with the corresponding Markdown footnote `[^N]`, where N is the number associated with `<full_url_here>` in the Bibliography Map. If a URL in a citation tag is NOT found in the Bibliography Map, OMIT the citation marker entirely for that instance.
            5. **Conclusion**: Summarize key findings (or lack thereof). Mention limitations (e.g., number of sources, potential scraping gaps, empty synthesis). Suggest further research if appropriate.
            6. **Bibliography**: List all sources from the Bibliography Map numerically using Markdown footnote definition format: `[^N]: <full_url_here>`. If the map is empty, state "No sources cited."
            7. Output ONLY the complete Markdown report. No extra text before or after.
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
                        is_fatal_report_err = "API key" in report_stream_error or "quota" in report_stream_error.lower()
                        yield from send_error_event(f"LLM stream error during report generation: {report_stream_error}", is_fatal=is_fatal_report_err)
                        if is_fatal_report_err: return
                        break # Break on non-fatal stream error
                    elif result['type'] == 'stream_warning':
                         yield from send_progress(f"LLM Stream Warning (Report): {result['message']}")
                    elif result['type'] == 'stream_end':
                         break # Normal end
            except Exception as e:
                 yield from send_error_event(f"Fatal error processing LLM report stream: {e}", is_fatal=True)
                 traceback.print_exc()
                 return

            yield from send_progress("Report generation finished.")
            if not final_report_markdown.strip():
                 yield from send_progress("Warning: Final report content is empty.")
                 # Create a placeholder if completely empty
                 final_report_markdown = f"# Research Report: {topic}\n\n*Report generation failed or produced no content.*"


            # === Step 6: Final Processing and Completion ===
            yield from send_progress("Processing final report for display...")

            # Convert final Markdown to HTML for display
            report_html = convert_markdown_to_html(final_report_markdown)

            # Prepare raw data preview (limited size)
            preview_limit_chars = 3000
            raw_data_preview_list = []
            current_preview_len = 0
            for src in scraped_sources_list:
                # Use compact JSON for preview to save space
                src_dump = json.dumps(src, separators=(',', ':'), ensure_ascii=False)
                if current_preview_len + len(src_dump) < preview_limit_chars:
                    raw_data_preview_list.append(src_dump)
                    current_preview_len += len(src_dump)
                else:
                    break # Stop adding if preview limit exceeded
            raw_data_preview = "[\n  " + ",\n  ".join(raw_data_preview_list) + "\n]"
            if len(scraped_sources_list) > len(raw_data_preview_list):
                 raw_data_preview += f"\n... ({len(raw_data_preview_list)}/{len(scraped_sources_list)} sources shown due to preview size limit)"
            elif not scraped_sources_list:
                 raw_data_preview = "[] (No sources scraped)"

            # Send final data package to the client
            final_data = {
                'type': 'complete',
                'report_html': report_html,
                'report_markdown': final_report_markdown,
                'raw_scraped_data_preview': raw_data_preview,
                'docx_available': DOCX_CONVERSION_AVAILABLE # Use the flag set at startup
            }
            yield from send_event(final_data)

            end_time_total = time.time()
            yield from send_progress(f"Research process completed successfully in {end_time_total - start_time_total:.2f} seconds.")

        except Exception as e:
            # Catch any unexpected errors in the main workflow
            print(f"FATAL: An unexpected error occurred during stream generation:")
            traceback.print_exc() # Print full traceback to server logs
            error_msg = f"Unexpected server error: {type(e).__name__} - {escape(str(e))}"
            yield from send_error_event(error_msg, is_fatal=True)
        finally:
             # Signal that the stream is definitively finished, regardless of success/error
             yield from send_event({'type': 'stream_terminated'})


    # Set headers for Server-Sent Events
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no', # Important for Nginx buffering
        'Connection': 'keep-alive'
    }
    return Response(stream_with_context(generate_updates()), headers=headers)


@app.route('/download_docx', methods=['POST'])
def download_docx():
    """Converts the received Markdown report to DOCX and sends it as a download."""
    if not DOCX_CONVERSION_AVAILABLE:
        print("Error [/download_docx]: Attempted download when DOCX conversion is unavailable.")
        return jsonify({"success": False, "message": "DOCX download failed: 'html2docx' library not installed or loaded."}), 400

    markdown_content = request.form.get('markdown_report')
    topic = request.form.get('topic', 'Research_Report') # Get topic for filename

    if not markdown_content:
        print("Error [/download_docx]: No Markdown content received.")
        return jsonify({"success": False, "message": "Error: No report content received for DOCX conversion."}), 400

    try:
        # 1. Convert Markdown to HTML (required by html2docx)
        report_html = convert_markdown_to_html(markdown_content)

        # 2. Convert HTML to DOCX in memory buffer using the utility function
        docx_buffer = convert_html_to_docx(report_html)

        # 3. Prepare filename
        # Sanitize topic for filename: remove invalid chars, replace spaces
        safe_filename_topic = re.sub(r'[^\w\s-]', '', topic).strip()
        safe_filename_topic = re.sub(r'[-\s]+', '_', safe_filename_topic)
        # Truncate if too long
        filename_base = f"{safe_filename_topic}_Research_Report"
        filename = f"{filename_base[:config.DOWNLOAD_FILENAME_MAX_LENGTH]}.docx"

        # 4. Send the buffer as a file download
        return send_file(
            docx_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

    except ImportError: # Should be caught by initial check, but as fallback
         print("Error [/download_docx]: html2docx library missing during conversion attempt.")
         return jsonify({"success": False, "message": "Internal Server Error: DOCX conversion library missing."}), 500
    except Exception as e:
        print(f"ERROR [/download_docx]: Error converting report to DOCX:")
        traceback.print_exc()
        msg = f"An error occurred during DOCX conversion: {escape(str(e))}"
        return jsonify({"success": False, "message": msg}), 500


# --- Run the App ---
if __name__ == '__main__':
    # Check if GOOGLE_API_KEY was loaded (config already checked, this is redundant but informative)
    if not config.GOOGLE_API_KEY:
        print("FATAL ERROR: GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")
        sys.exit(1) # Exit if key is missing

    print(f"INFO: Starting Flask server...")
    # Use threaded=True for handling multiple requests concurrently during SSE
    # debug=True enables auto-reloading and provides detailed error pages (disable in production)
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
    # For production, use a proper WSGI server like Gunicorn or Waitress:
    # Example: gunicorn --workers 4 --bind 0.0.0.0:5001 app:app
    # Example: waitress-serve --host 0.0.0.0 --port 5001 app:app