import sys
import os
import json
import traceback
import logging
from urllib.parse import quote, unquote
from typing import Dict, Any, Generator, Callable

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# You might want to configure Flask's logger specifically for production
# E.g., RotatingFileHandler, sending logs to stderr for container environments, etc.
# app.logger.addHandler(...)

# --- Initialize Google Generative AI Client ---
if not config.GOOGLE_API_KEY:
     app.logger.critical("FATAL: GOOGLE_API_KEY environment variable not set.")
     raise ValueError("FATAL: GOOGLE_API_KEY environment variable not set.")
if not config.GOOGLE_MODEL_NAME:
     app.logger.critical("FATAL: GOOGLE_MODEL_NAME environment variable not set.")
     raise ValueError("FATAL: GOOGLE_MODEL_NAME environment variable not set.")

try:
     genai.configure(api_key=config.GOOGLE_API_KEY)
     app.logger.info(f"Google Generative AI configured with model: {config.GOOGLE_MODEL_NAME}")
except Exception as e:
     app.logger.critical(f"FATAL: Failed to configure Google Generative AI: {e}", exc_info=True)
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
        app.logger.warning("Research request received with empty topic.")
        return redirect(url_for('index'))
    encoded_topic = quote(topic)
    app.logger.info(f"Redirecting to results page for topic: '{topic}'")
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
        app.logger.warning("SSE stream requested with no topic parameter, using default.")

    app.logger.info(f"SSE stream started for topic: '{topic}'")

    # This inner function generates the actual SSE stream content
    def generate_updates() -> Generator[str, None, None]:
        temp_files_to_clean = [] # Keep track of files to clean up

        # --- SSE Helper functions (defined inside to close over flask context if needed) ---
        def send_event(data: Dict[str, Any]) -> str:
            """Formats data as an SSE message string."""
            try:
                payload = json.dumps(data)
                return f"data: {payload}\n\n"
            except TypeError as e:
                current_app.logger.error(f"Error serializing data for SSE: {e}. Data: {data}", exc_info=True)
                try:
                    safe_data = {'type': data.get('type', 'error'), 'message': f"Serialization Error: {e}"}
                    payload = json.dumps(safe_data)
                    return f"data: {payload}\n\n"
                except Exception:
                    current_app.logger.error("Internal server error during fallback SSE event serialization.")
                    return "data: {\"type\": \"error\", \"message\": \"Internal server error during SSE event serialization.\"}\n\n"

        def send_progress_update(message: str, is_error: bool = False, is_fatal: bool = False) -> str:
            """Sends a progress or error update event."""
            event_type = 'error' if is_error else 'progress'
            event_data = {'type': event_type, 'message': message}
            log_func = current_app.logger.error if is_error else current_app.logger.info

            if is_error:
                log_func(f"SSE {'FATAL ' if is_fatal else ''}ERROR: {message}")
                event_data['fatal'] = is_fatal
            else:
                log_func(f"SSE Progress: {message}")

            return send_event(event_data)

        # --- Callback functions to pass to the orchestrator ---
        # These callbacks will yield the SSE formatted string directly
        def yield_event_callback(data: Dict[str, Any]):
            nonlocal generator_instance # Need to yield from the outer generator
            generator_instance.send(send_event(data))

        def yield_progress_callback(message: str, is_error: bool = False, is_fatal: bool = False):
            nonlocal generator_instance
            generator_instance.send(send_progress_update(message, is_error, is_fatal))


        # --- Main Research Workflow (now delegated) ---
        generator_instance = None # Will hold the generator instance
        try:
            # The run_research_process function will now call the callbacks directly
            # instead of yielding. We wrap its execution in a generator that yields
            # the results from the callbacks.
            def _inner_run():
                nonlocal temp_files_to_clean
                # The orchestrator returns the list of temp files it created
                temp_files_to_clean = run_research_process(
                    topic=topic,
                    send_event_callback=yield_event_callback,
                    send_progress_callback=yield_progress_callback
                )
                # Signal normal completion if orchestrator finishes without fatal error
                yield send_progress_update("Orchestrator finished processing.")

            # Create the generator and assign it
            generator_instance = _inner_run()

            # Iterate through the generator, which effectively runs the orchestrator
            # and yields the SSE messages produced by the callbacks.
            for sse_message in generator_instance:
                yield sse_message

        except Exception as e:
            # Catch unexpected errors *during the orchestration setup or iteration*
            current_app.logger.error(f"FATAL: Unexpected error during stream generation orchestration for topic '{topic}': {e}", exc_info=True)
            # traceback.print_exc() # Log full traceback to server logs
            error_msg = f"Unexpected server error during research orchestration: {type(e).__name__} - {str(e)}"
            yield send_progress_update(error_msg, is_error=True, is_fatal=True) # Send fatal error via SSE

        finally:
            # --- Clean up temporary files ---
            if temp_files_to_clean:
                 current_app.logger.info(f"Cleaning up {len(temp_files_to_clean)} temporary scrape files...")
                 cleaned_count = 0
                 failed_count = 0
                 for fpath in temp_files_to_clean:
                     try:
                         if fpath and os.path.exists(fpath): # Check if path is not None
                              os.remove(fpath)
                              cleaned_count += 1
                     except OSError as e:
                         current_app.logger.warning(f"Failed to remove temp file {os.path.basename(fpath)}: {e}")
                         failed_count += 1
                     except Exception as e:
                         current_app.logger.error(f"Unexpected error removing temp file {fpath}: {e}")
                         failed_count += 1
                 current_app.logger.info(f"Temp file cleanup complete. Removed: {cleaned_count}, Failed: {failed_count}")
            else:
                 current_app.logger.info("No temporary files to clean up.")

            # Signal that the stream is definitively finished
            current_app.logger.info(f"SSE stream terminated for topic: '{topic}'")
            yield send_event({'type': 'stream_terminated'})


    # Set headers for Server-Sent Events
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no', # Important for Nginx/Caddy buffering issues
        'Connection': 'keep-alive'
    }
    # Use stream_with_context to ensure context is available during generation
    return Response(stream_with_context(generate_updates()), headers=headers)


# --- Run the App ---
if __name__ == '__main__':
    # Checks for keys are done at the top level now
    print(f"INFO: Starting Flask development server...") # Use print for initial startup message
    # Use Flask's built-in server for development ONLY. Use Gunicorn/Waitress for production.
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
    # Note: `threaded=True` is suitable for development with concurrent requests,
    # but a proper WSGI server (Gunicorn) handles concurrency better in production.
