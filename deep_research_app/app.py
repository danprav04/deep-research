# app.py
import sys
import os
import json
import traceback
import logging
import logging.handlers
from urllib.parse import quote, unquote
from typing import Dict, Any, Generator, List, Optional

from flask import (
    Flask, render_template, request, redirect, url_for, jsonify,
    Response, stream_with_context, current_app, make_response,
    send_from_directory # Added for favicon
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import google.generativeai as genai

# Import configuration and outsourced modules
import config as config
from research_orchestrator import run_research_process # Core logic moved
from utils import convert_markdown_to_html # Keep utility

# --- Initialize Flask App ---
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')
# Use FLASK_SECRET_KEY from env if available, otherwise use a default (less secure for dev only)
# For production, *always* set a strong FLASK_SECRET_KEY environment variable.
app.secret_key = config.FLASK_SECRET_KEY or os.urandom(24)
if not config.FLASK_SECRET_KEY:
     print("WARNING: FLASK_SECRET_KEY environment variable not set. Using insecure default for session management.", file=sys.stderr)


# --- Configure Logging ---
def setup_logging():
    log_level_str = config.LOG_LEVEL
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    log_file = config.LOG_FILE_PATH
    log_dir = os.path.dirname(log_file)

    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            print(f"ERROR: Could not create log directory {log_dir}: {e}", file=sys.stderr)
            # Fallback to stderr logging if file path fails
            logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] %(message)s')
            logging.getLogger().error(f"Logging to file failed, falling back to stderr.")
            return

    # Use TimedRotatingFileHandler for automatic rotation
    # Ensure logs directory exists before setting up handler
    if log_dir: # Check again in case creation failed silently
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when='midnight', interval=1, backupCount=config.LOG_ROTATION_DAYS, encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] %(message)s')
        file_handler.setFormatter(formatter)
        handlers = [file_handler]
    else:
        handlers = [logging.StreamHandler(sys.stderr)] # Log to stderr if file logging failed

    # Configure root logger
    logging.basicConfig(level=log_level, handlers=handlers, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] %(message)s')

    # Get Flask's logger and add our handler to it
    # Also configure Werkzeug logger to use our handler
    flask_logger = logging.getLogger('flask.app')
    werkzeug_logger = logging.getLogger('werkzeug')
    for handler in handlers:
        flask_logger.addHandler(handler)
        werkzeug_logger.addHandler(handler)
    flask_logger.setLevel(log_level)
    werkzeug_logger.setLevel(logging.WARNING) # Set Werkzeug to WARNING for less noise in prod logs

    # Prevent Flask's default handlers if we added ours
    flask_logger.propagate = False
    # If logging to stderr, we don't need Werkzeug's default handler either
    if any(isinstance(h, logging.StreamHandler) for h in handlers):
        werkzeug_logger.propagate = False


    logging.getLogger().info(f"Logging configured. Level: {log_level_str}, Target: {'File ('+log_file+')' if log_dir else 'stderr'}, Rotation: {config.LOG_ROTATION_DAYS} days.")

setup_logging()
logger = logging.getLogger(__name__) # Use module-specific logger after setup

# --- Configure Rate Limiting ---
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[config.DEFAULT_RATE_LIMIT],
    storage_uri="memory://",  # Use memory storage for simplicity, consider Redis for multi-process setups
    strategy="fixed-window" # Or "moving-window"
)
logger.info(f"Rate limiting configured with default: {config.DEFAULT_RATE_LIMIT}")


# --- Initialize Google Generative AI Client ---
if not config.GOOGLE_API_KEY:
     logger.critical("FATAL: GOOGLE_API_KEY environment variable not set.")
     raise ValueError("FATAL: GOOGLE_API_KEY environment variable not set.")
if not config.GOOGLE_MODEL_NAME:
     logger.critical("FATAL: GOOGLE_MODEL_NAME environment variable not set.")
     raise ValueError("FATAL: GOOGLE_MODEL_NAME environment variable not set.")

try:
     genai.configure(api_key=config.GOOGLE_API_KEY)
     logger.info(f"Google Generative AI configured with model: {config.GOOGLE_MODEL_NAME}")
except Exception as e:
     logger.critical(f"FATAL: Failed to configure Google Generative AI: {e}", exc_info=True)
     raise RuntimeError(f"FATAL: Failed to configure Google Generative AI: {e}")


# --- Flask Routes ---

@app.route('/')
def index():
    """Displays the main input form."""
    return render_template('index.html', pico_css=config.PICO_CSS_CDN)

@app.route('/research', methods=['POST'])
@limiter.limit(config.DEFAULT_RATE_LIMIT) # Apply rate limit
def research_start():
    """Validates topic and redirects to the results page."""
    topic = request.form.get('topic', '').strip()
    # Basic validation (can be expanded)
    if not topic:
        logger.warning("Research request received with empty topic.")
        # Consider flashing a message to the user here
        return redirect(url_for('index'))
    if len(topic) > 300: # Arbitrary length limit
        logger.warning(f"Research request received with excessively long topic (>{len(topic)} chars). Truncating.")
        topic = topic[:300] # Truncate or return error

    # Encode safely for URL
    encoded_topic = quote(topic, safe='')
    logger.info(f"Redirecting to results page for topic: '{topic}' (Encoded: {encoded_topic})")
    return render_template('results.html',
                           topic=topic,
                           encoded_topic=encoded_topic,
                           pico_css=config.PICO_CSS_CDN)


@app.route('/stream')
@limiter.limit(config.DEFAULT_RATE_LIMIT) # Apply rate limit
def stream():
    """The main SSE route that performs research and streams progress."""
    encoded_topic = request.args.get('topic', '')
    topic = ""
    try:
        # Decode topic from URL, handle potential errors
        topic = unquote(encoded_topic)
    except Exception as e:
         logger.error(f"Error decoding topic from URL parameter '{encoded_topic}': {e}")
         # Return an error response immediately? Or proceed with a default?
         # For now, proceed with default, but log clearly.
         topic = "Default Topic - Error Decoding URL Parameter"

    if not topic or topic == "Default Topic - Error Decoding URL Parameter":
        logger.warning(f"SSE stream requested with missing or invalid topic parameter ('{encoded_topic}'). Using default.")
        # Use a default topic or return an error? For robustness, let's try a default.
        topic = "Default Topic - No Valid Topic Provided"

    logger.info(f"SSE stream started for topic: '{topic}'")

    def generate_updates() -> Generator[str, None, None]:
        temp_files_to_clean: List[str] = [] # Track files locally

        def format_sse_event(data: Dict[str, Any]) -> str:
            """Formats data as a valid SSE 'data:' message string."""
            try:
                # Ensure complex objects are handled (though basic types are expected)
                payload = json.dumps(data)
                return f"data: {payload}\n\n"
            except TypeError as e:
                current_app.logger.error(f"Error serializing data for SSE: {e}. Data: {data}", exc_info=False) # Avoid logging potentially large data
                safe_data = {'type': data.get('type', 'error'), 'message': f"Serialization Error: Could not format server event."}
                try:
                    payload = json.dumps(safe_data)
                    return f"data: {payload}\n\n"
                except Exception as inner_e:
                    current_app.logger.error(f"Internal server error during fallback SSE event serialization: {inner_e}")
                    return "data: {\"type\": \"error\", \"message\": \"Internal server error during SSE event processing.\"}\n\n"

        try:
            orchestrator_generator = run_research_process(topic=topic)

            for event_data in orchestrator_generator:
                event_type = event_data.get('type')

                if event_type == 'progress':
                    message = event_data.get('message', 'Progress update')
                    is_error = event_data.get('is_error', False)
                    is_fatal = event_data.get('is_fatal', False)
                    log_func = current_app.logger.error if is_error or is_fatal else current_app.logger.info
                    log_prefix = "FATAL SSE:" if is_fatal else "ERROR SSE:" if is_error else "PROGRESS SSE:"
                    log_func(f"{log_prefix} {message[:500]}{'...' if len(message) > 500 else ''}")
                    yield format_sse_event({'type': 'error' if is_error or is_fatal else 'progress', 'message': message, 'fatal': is_fatal})

                elif event_type == 'scrape_success':
                    metadata = event_data.get('metadata')
                    if metadata and metadata.get('temp_filepath'):
                        filepath = metadata['temp_filepath']
                        temp_files_to_clean.append(filepath)
                        current_app.logger.debug(f"Tracking temp file for cleanup: {os.path.basename(filepath)}")

                elif event_type == 'event': # Generic event wrapper for things like 'complete'
                     inner_event_data = event_data.get('data', {})
                     yield format_sse_event(inner_event_data)

                else:
                    current_app.logger.warning(f"Received unknown event type from orchestrator: {event_type}")
                    yield format_sse_event({'type': 'error', 'message': f'Unknown server event type received: {event_type}', 'fatal': False})

            current_app.logger.info(f"Orchestrator generator finished normally for topic: '{topic}'.")

        except Exception as e:
            current_app.logger.error(f"FATAL: Unexpected error during stream generation for topic '{topic}': {e}", exc_info=True)
            error_msg = "An unexpected server error occurred during the research process. Please check server logs."
            try:
                yield format_sse_event({'type': 'error', 'message': error_msg, 'fatal': True})
            except Exception as yield_err:
                 current_app.logger.error(f"Failed to yield final fatal error message over SSE: {yield_err}")

        finally:
            # --- Clean up temporary files ---
            if temp_files_to_clean:
                 current_app.logger.info(f"Cleaning up {len(temp_files_to_clean)} temporary scrape files for topic '{topic}'...")
                 cleaned_count = 0
                 failed_count = 0
                 for fpath in temp_files_to_clean:
                     try:
                         if fpath and isinstance(fpath, str) and os.path.exists(fpath):
                              os.remove(fpath)
                              cleaned_count += 1
                              current_app.logger.debug(f"Removed temp file: {os.path.basename(fpath)}")
                         elif not os.path.exists(fpath):
                              current_app.logger.debug(f"Temp file already removed or path invalid: {fpath}")
                              cleaned_count +=1
                     except OSError as e:
                         current_app.logger.warning(f"Failed to remove temp file {os.path.basename(fpath)}: {e}")
                         failed_count += 1
                     except Exception as e:
                         current_app.logger.error(f"Unexpected error removing temp file {fpath}: {e}", exc_info=False)
                         failed_count += 1
                 current_app.logger.info(f"Temp file cleanup for '{topic}' complete. Removed: {cleaned_count}, Failed: {failed_count}. Total tracked: {len(temp_files_to_clean)}")
            else:
                 current_app.logger.info(f"No temporary files tracked for cleanup for topic '{topic}'.")

            current_app.logger.info(f"SSE stream processing definitively finished for topic: '{topic}'")
            try:
                yield format_sse_event({'type': 'stream_terminated'})
            except Exception as yield_err:
                 current_app.logger.error(f"Failed to yield final stream_terminated message: {yield_err}")

    # --- Set headers for Server-Sent Events (SSE) ---
    # DO NOT set 'Connection: keep-alive' here - it violates WSGI spec (PEP 3333)
    # The WSGI server (Gunicorn/Waitress) handles connection persistence.
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        # 'X-Accel-Buffering': 'no', # Primarily for Nginx, uncomment if using Nginx & experiencing buffering
    }
    # Use stream_with_context to ensure context is available during generation
    response = Response(stream_with_context(generate_updates()), mimetype='text/event-stream', headers=headers)
    # Optional: Add X-Accel-Buffering header specifically if needed for Nginx proxy
    # response.headers['X-Accel-Buffering'] = 'no'
    return response


@app.route('/health')
def health_check():
    """Basic health check endpoint."""
    # Could add checks here (e.g., LLM connectivity, DB status if used)
    # Example: Check LLM model availability
    llm_ok = False
    try:
        # Perform a lightweight check, e.g., list models (if API allows)
        # Or just assume configured is enough for basic check
        if config.GOOGLE_API_KEY and config.GOOGLE_MODEL_NAME:
            # genai.get_model(f'models/{config.GOOGLE_MODEL_NAME}') # This might be too slow/costly for health check
            llm_ok = True # Basic check: Assume OK if configured
    except Exception as e:
        logger.warning(f"Health check: LLM configuration check failed: {e}", exc_info=False)

    status = {"status": "ok", "llm_configured": llm_ok}
    http_status = 200 if llm_ok else 503 # Service Unavailable if LLM isn't ready

    return jsonify(status), http_status

# --- Favicon Route (Optional but Recommended) ---
@app.route('/favicon.ico')
def favicon():
    # Serve a favicon file if you have one in static/
    # return send_from_directory(os.path.join(app.root_path, 'static'),
    #                           'favicon.ico', mimetype='image/vnd.microsoft.icon')
    # Or simply return 204 No Content if you don't have one
    return '', 204

# --- Error Handlers ---
@app.errorhandler(429)
def ratelimit_handler(e):
    """Handles rate limit errors."""
    logger.warning(f"Rate limit exceeded: {e.description} from {get_remote_address()}")
    # Simple JSON response for API-like errors, or render template
    # return make_response(jsonify(error=f"Rate limit exceeded: {e.description}"), 429)
    # Render a simple error page instead
    return render_template('error.html', error_code=429, error_message=f"Too many requests: {e.description}. Please try again later."), 429

@app.errorhandler(404)
def not_found_error(error):
    """Handles 404 Not Found errors."""
    logger.warning(f"404 Not Found: {request.path} ({error})")
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handles 500 Internal Server errors."""
    # Log the error with stack trace
    logger.error(f"Internal Server Error (500): {error}", exc_info=True)
    # Return a generic error page/message
    return render_template('500.html'), 500

# Catch-all for other Werkzeug/HTTP exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    """Handles unexpected exceptions."""
    # Handle specific HTTP exceptions if needed, otherwise treat as 500
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        # Use the exception's default response if available
        return e
    # Log the full exception details for non-HTTP errors
    logger.error(f"Unhandled Exception: {e}", exc_info=True)
    # Return the generic 500 error page
    return render_template('500.html'), 500

# Note: Removed the `if __name__ == '__main__':` block.
# Use a WSGI server like Gunicorn or Waitress to run the app.
# Example command: waitress-serve --host 0.0.0.0 --port 8000 deep_research_app.app:app
# Or: gunicorn --bind 0.0.0.0:8000 deep_research_app.app:app