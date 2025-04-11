# app.py
import sys
import os
import json
import traceback
import logging
from urllib.parse import quote, unquote
from typing import Dict, Any, Generator, List, Optional

from flask import (
    Flask, render_template, request, redirect, url_for, jsonify,
    Response, stream_with_context, current_app
)
import google.generativeai as genai

# Import configuration and outsourced modules
import config as config
from research_orchestrator import run_research_process # Core logic moved
from utils import convert_markdown_to_html # Keep utility

# --- Initialize Flask App ---
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')
app.secret_key = os.urandom(24)

# --- Configure Logging ---
# Basic config (level set in config.py)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
# Get Flask's logger
flask_logger = logging.getLogger('flask.app')

# --- Initialize Google Generative AI Client ---
if not config.GOOGLE_API_KEY:
     flask_logger.critical("FATAL: GOOGLE_API_KEY environment variable not set.")
     raise ValueError("FATAL: GOOGLE_API_KEY environment variable not set.")
if not config.GOOGLE_MODEL_NAME:
     flask_logger.critical("FATAL: GOOGLE_MODEL_NAME environment variable not set.")
     raise ValueError("FATAL: GOOGLE_MODEL_NAME environment variable not set.")

try:
     genai.configure(api_key=config.GOOGLE_API_KEY)
     flask_logger.info(f"Google Generative AI configured with model: {config.GOOGLE_MODEL_NAME}")
except Exception as e:
     flask_logger.critical(f"FATAL: Failed to configure Google Generative AI: {e}", exc_info=True)
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
        flask_logger.warning("Research request received with empty topic.")
        return redirect(url_for('index'))
    encoded_topic = quote(topic)
    flask_logger.info(f"Redirecting to results page for topic: '{topic}'")
    return render_template('results.html',
                           topic=topic,
                           encoded_topic=encoded_topic,
                           pico_css=config.PICO_CSS_CDN)

@app.route('/stream')
def stream():
    """The main SSE route that performs research and streams progress by iterating over the orchestrator."""
    encoded_topic = request.args.get('topic', '')
    topic = unquote(encoded_topic) # Decode topic from URL
    if not topic:
        topic = "Default Topic - No Topic Provided" # Fallback
        flask_logger.warning("SSE stream requested with no topic parameter, using default.")

    flask_logger.info(f"SSE stream started for topic: '{topic}'")

    # This inner function generates the actual SSE stream content
    def generate_updates() -> Generator[str, None, None]:
        temp_files_to_clean: List[str] = [] # Track files locally

        # --- SSE Formatting Helper functions ---
        def format_sse_event(data: Dict[str, Any]) -> str:
            """Formats data as an SSE 'data:' message string."""
            try:
                payload = json.dumps(data)
                return f"data: {payload}\n\n"
            except TypeError as e:
                # Log error, create safe fallback event
                current_app.logger.error(f"Error serializing data for SSE: {e}. Data: {data}", exc_info=False)
                safe_data = {'type': data.get('type', 'error'), 'message': f"Serialization Error: {e}"}
                try:
                    payload = json.dumps(safe_data)
                    return f"data: {payload}\n\n"
                except Exception as inner_e:
                    current_app.logger.error(f"Internal server error during fallback SSE event serialization: {inner_e}")
                    return "data: {\"type\": \"error\", \"message\": \"Internal server error during SSE event serialization.\"}\n\n"

        # --- Main Research Workflow ---
        try:
            # Directly iterate over the generator returned by the orchestrator
            orchestrator_generator = run_research_process(topic=topic)

            for event_data in orchestrator_generator:
                event_type = event_data.get('type')

                # Process different event types yielded by the orchestrator
                if event_type == 'progress':
                    message = event_data.get('message', 'Progress update')
                    is_error = event_data.get('is_error', False)
                    is_fatal = event_data.get('is_fatal', False)
                    log_func = current_app.logger.error if is_error else current_app.logger.info
                    log_prefix = "SSE FATAL ERROR:" if is_fatal else "SSE ERROR:" if is_error else "SSE Progress:"
                    log_func(f"{log_prefix} {message}")
                    yield format_sse_event({'type': 'error' if is_error else 'progress', 'message': message, 'fatal': is_fatal})

                elif event_type == 'scrape_success':
                    metadata = event_data.get('metadata')
                    if metadata and metadata.get('temp_filepath'):
                        temp_files_to_clean.append(metadata['temp_filepath'])
                        current_app.logger.debug(f"Tracking temp file for cleanup: {metadata['temp_filepath']}")
                    # Optionally send a progress update about the successful scrape? No, orchestrator handles this.

                elif event_type == 'event': # Generic event wrapper
                     inner_event_data = event_data.get('data', {})
                     yield format_sse_event(inner_event_data) # Pass inner data directly

                else:
                    current_app.logger.warning(f"Received unknown event type from orchestrator: {event_type}")
                    # Optionally yield an error or just ignore
                    # yield format_sse_event({'type': 'error', 'message': f'Unknown server event type: {event_type}', 'fatal': False})


            # If the loop finishes without a fatal error handled inside:
            current_app.logger.info("Orchestrator generator finished.")
            # Normal termination is handled by the 'complete' event from orchestrator

        except Exception as e:
            # Catch unexpected errors *during the orchestration execution*
            current_app.logger.error(f"FATAL: Unexpected error during stream generation for topic '{topic}': {e}", exc_info=True)
            error_msg = f"Unexpected server error during research: {type(e).__name__} - {str(e)}"
            # Send fatal error via SSE
            yield format_sse_event({'type': 'error', 'message': error_msg, 'fatal': True})

        finally:
            # --- Clean up temporary files ---
            if temp_files_to_clean:
                 current_app.logger.info(f"Cleaning up {len(temp_files_to_clean)} temporary scrape files...")
                 cleaned_count = 0
                 failed_count = 0
                 for fpath in temp_files_to_clean:
                     try:
                         if fpath and os.path.exists(fpath):
                              os.remove(fpath)
                              cleaned_count += 1
                     except OSError as e:
                         current_app.logger.warning(f"Failed to remove temp file {os.path.basename(fpath)}: {e}")
                         failed_count += 1
                     except Exception as e:
                         current_app.logger.error(f"Unexpected error removing temp file {fpath}: {e}", exc_info=False)
                         failed_count += 1
                 current_app.logger.info(f"Temp file cleanup complete. Removed: {cleaned_count}, Failed: {failed_count}")
            else:
                 current_app.logger.info("No temporary files tracked for cleanup.")

            # Signal that the stream is definitively finished
            current_app.logger.info(f"SSE stream processing finished for topic: '{topic}'")
            yield format_sse_event({'type': 'stream_terminated'})


    # Set headers for Server-Sent Events
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        'Connection': 'keep-alive'
    }
    # Use stream_with_context to ensure context is available during generation
    return Response(stream_with_context(generate_updates()), headers=headers)


# --- Run the App ---
if __name__ == '__main__':
    print(f"INFO: Starting Flask development server...")
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)