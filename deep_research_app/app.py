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
    send_from_directory
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import google.generativeai as genai

# Import configuration and outsourced modules
import config as config # Imports updated config with absolute paths
from research_orchestrator import run_research_process # Core logic moved
from utils import convert_markdown_to_html # Keep utility

# --- Configure Logging ---
# Moved setup_logging call to *before* Flask app creation to ensure
# logging is ready early, especially for Gunicorn workers.

def setup_logging():
    """Configures logging for the application."""
    log_level_str = config.LOG_LEVEL
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_file = config.LOG_FILE_PATH # Now guaranteed to be absolute
    log_dir = os.path.dirname(log_file)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] %(message)s')
    log_target_description = "stderr" # Default target description

    # Attempt to create log directory if specified and doesn't exist
    if log_dir:
        try:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                print(f"INFO: Created log directory: {log_dir}")
            # Check write permissions (important for Linux/Gunicorn)
            if not os.access(log_dir, os.W_OK):
                 raise OSError(f"Log directory '{log_dir}' exists but is not writable.")

            # Create file handler if directory is ready
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_file, when='midnight', interval=1, backupCount=config.LOG_ROTATION_DAYS, encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(log_formatter)
            handlers = [file_handler]
            log_target_description = f"File ({log_file})"

        except (OSError, Exception) as e:
            print(f"ERROR: Could not create/access log directory or file '{log_file}'. Logging to stderr. Error: {e}", file=sys.stderr)
            # Fallback to stderr logging
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setLevel(log_level)
            stream_handler.setFormatter(log_formatter)
            handlers = [stream_handler]
            log_target_description = "stderr (fallback due to error)"
    else:
        # No log file specified, use stderr
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(log_formatter)
        handlers = [stream_handler]

    # --- Configure Root Logger ---
    root_logger = logging.getLogger()
    # Remove existing handlers attached to the root logger (important for Gunicorn workers)
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close() # Close handler to release file resources if any
    # Add the configured handler(s)
    for handler in handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # --- Configure Flask and Werkzeug Loggers ---
    # Get Flask's logger and Werkzeug logger
    flask_logger = logging.getLogger('flask.app')
    werkzeug_logger = logging.getLogger('werkzeug')

    # Clear existing handlers and add our configured ones
    for logger_instance in [flask_logger, werkzeug_logger]:
        if logger_instance.hasHandlers():
            for handler in logger_instance.handlers[:]:
                logger_instance.removeHandler(handler)
                # No need to close here, handlers are shared and closed above if root had them
        for handler in handlers:
            logger_instance.addHandler(handler)

    # Set specific levels for Flask/Werkzeug
    flask_logger.setLevel(log_level)
    werkzeug_logger.setLevel(logging.WARNING) # Keep Werkzeug less verbose in production

    # Prevent double logging by stopping propagation to the root logger's handlers
    flask_logger.propagate = False
    werkzeug_logger.propagate = False

    # Use the root logger for the initial confirmation message
    logging.getLogger().info(f"Logging configured. Level: {log_level_str}, Target: {log_target_description}, Rotation: {config.LOG_ROTATION_DAYS} days.")

# Call logging setup *before* Flask app instance is created
setup_logging()
logger = logging.getLogger(__name__) # Module-specific logger

# --- Initialize Flask App ---
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

# --- App Configuration ---
app.secret_key = config.FLASK_SECRET_KEY or os.urandom(24)
if not config.FLASK_SECRET_KEY:
     logger.warning("FLASK_SECRET_KEY environment variable not set. Using insecure default for session management.")

# --- Configure Rate Limiting ---
try:
    limiter = Limiter(
        get_remote_address,
        app=app,
        # default_limits=[config.DEFAULT_RATE_LIMIT], # <-- REMOVED THIS LINE
        storage_uri="memory://",
        strategy="fixed-window",
        headers_enabled=True
    )
    # Update log message to reflect no default limit
    logger.info(f"Rate limiting configured. Limits applied via route decorators (value: {config.DEFAULT_RATE_LIMIT}).")
except Exception as e:
    logger.error(f"Failed to initialize Flask-Limiter: {e}", exc_info=True)
    # Consider if the app should stop or continue without rate limiting
    # For now, log the error and continue


# --- Initialize Google Generative AI Client ---
# Validation happens in config.py, critical errors raise ValueError there
try:
     genai.configure(api_key=config.GOOGLE_API_KEY)
     logger.info(f"Google Generative AI configured with model: {config.GOOGLE_MODEL_NAME}")
except Exception as e:
     # Log critical error and raise runtime error to prevent app start
     logger.critical(f"FATAL: Failed to configure Google Generative AI: {e}", exc_info=True)
     raise RuntimeError(f"FATAL: Failed to configure Google Generative AI: {e}")


# --- Flask Routes ---

@app.route('/')
# No @limiter decorator here, so it won't be limited anymore
def index():
    """Displays the main input form."""
    return render_template('index.html', pico_css=config.PICO_CSS_CDN)

@app.route('/research', methods=['POST'])
@limiter.limit(config.DEFAULT_RATE_LIMIT) # Apply specific limit ("1 per minute")
def research_start():
    """Validates topic and redirects to the results page."""
    topic = request.form.get('topic', '').strip()

    if not topic:
        logger.warning("Research request received with empty topic.")
        # TODO: Add flash message for user feedback
        return redirect(url_for('index'))

    MAX_TOPIC_LEN = 300 # Define locally or move to config
    if len(topic) > MAX_TOPIC_LEN:
        logger.warning(f"Research request received with excessively long topic (>{len(topic)} chars). Truncating.")
        topic = topic[:MAX_TOPIC_LEN].strip() # Truncate
        # TODO: Add flash message informing user of truncation

    # Encode safely for URL
    encoded_topic = quote(topic, safe='')
    logger.info(f"Redirecting to results page for topic: '{topic}' (Encoded: {encoded_topic})")
    # Render the results page directly, passing the topic for display
    return render_template('results.html',
                           topic=topic,
                           encoded_topic=encoded_topic,
                           pico_css=config.PICO_CSS_CDN)


@app.route('/stream')
@limiter.limit(config.DEFAULT_RATE_LIMIT) # Apply specific limit ("1 per minute")
def stream():
    """The main SSE route that performs research and streams progress."""
    encoded_topic = request.args.get('topic', '')
    topic = ""
    try:
        topic = unquote(encoded_topic)
        if not topic.strip():
             raise ValueError("Decoded topic is empty or whitespace.")
    except Exception as e:
         logger.error(f"Error decoding topic from URL parameter '{encoded_topic}': {e}")
         topic = "Invalid Topic Received" # Set a placeholder

    logger.info(f"SSE stream starting for topic: '{topic}'")

    def generate_updates() -> Generator[str, None, None]:
        # Use current_app's logger within the generator context
        app_logger = current_app.logger
        temp_files_to_clean: List[str] = [] # Track files locally

        def format_sse_event(data: Dict[str, Any]) -> str:
            """Formats data as a valid SSE 'data:' message string."""
            try:
                payload = json.dumps(data)
                return f"data: {payload}\n\n"
            except TypeError as e:
                app_logger.error(f"Error serializing data for SSE: {e}. Data type: {type(data)}", exc_info=False)
                safe_data = {'type': data.get('type', 'error'), 'message': "Serialization Error: Could not format server event."}
                try:
                    payload = json.dumps(safe_data)
                    return f"data: {payload}\n\n"
                except Exception as inner_e:
                    app_logger.error(f"Internal server error during fallback SSE event serialization: {inner_e}")
                    return "data: {\"type\": \"error\", \"message\": \"Internal server error during SSE event processing.\"}\n\n"

        # --- Start Orchestration ---
        try:
            orchestrator_generator = run_research_process(topic=topic)

            for event_data in orchestrator_generator:
                try:
                    event_type = event_data.get('type')

                    if event_type == 'progress':
                        message = event_data.get('message', 'Progress update')
                        is_error = event_data.get('is_error', False)
                        is_fatal = event_data.get('is_fatal', False)
                        log_func = app_logger.error if is_error or is_fatal else app_logger.info
                        log_prefix = "FATAL SSE:" if is_fatal else "ERROR SSE:" if is_error else "PROGRESS SSE:"
                        log_func(f"{log_prefix} {message[:500]}{'...' if len(message) > 500 else ''}") # Log progress
                        yield format_sse_event({'type': 'error' if is_error or is_fatal else 'progress', 'message': message, 'fatal': is_fatal})
                        if is_fatal:
                             app_logger.warning(f"Orchestrator yielded fatal error for topic '{topic}'. Stopping stream generation.")
                             break # Stop processing if orchestrator signals fatal error

                    elif event_type == 'scrape_success':
                        metadata = event_data.get('metadata')
                        if metadata and metadata.get('temp_filepath'):
                            filepath = metadata['temp_filepath']
                            if isinstance(filepath, str) and os.path.isabs(filepath) and config.TEMP_FILE_DIR in filepath: # Basic validation
                                temp_files_to_clean.append(filepath)
                                app_logger.debug(f"Tracking temp file for cleanup: {os.path.basename(filepath)}")
                            else:
                                app_logger.warning(f"Received potentially invalid temp_filepath in scrape_success: {filepath}")

                    elif event_type == 'event': # Generic event wrapper
                         inner_event_data = event_data.get('data', {})
                         yield format_sse_event(inner_event_data)

                    else:
                        app_logger.warning(f"Received unknown event type from orchestrator: {event_type}")
                        yield format_sse_event({'type': 'error', 'message': f'Unknown server event type received: {event_type}', 'fatal': False})

                except Exception as e:
                     app_logger.error(f"Error processing event from orchestrator (Topic: '{topic}'): {e}", exc_info=True)
                     # Yield an error to the client about this specific event processing failure
                     try:
                         yield format_sse_event({'type': 'error', 'message': 'Error processing server event.', 'fatal': False})
                     except Exception as yield_err:
                         app_logger.error(f"Failed to yield event processing error message over SSE: {yield_err}")


            app_logger.info(f"Orchestrator generator finished normally for topic: '{topic}'.")

        except Exception as e:
            # Catch errors during the *creation* or *iteration* of the orchestrator generator itself
            app_logger.error(f"FATAL: Unhandled error during stream generation for topic '{topic}': {e}", exc_info=True)
            error_msg = "An unexpected server error occurred during the research process. Please check server logs."
            try:
                # Yield a fatal error message to the client
                yield format_sse_event({'type': 'error', 'message': error_msg, 'fatal': True})
            except Exception as yield_err:
                 app_logger.error(f"Failed to yield final fatal error message over SSE: {yield_err}")

        finally:
            # --- Clean up temporary files ---
            if temp_files_to_clean:
                 app_logger.info(f"Cleaning up {len(temp_files_to_clean)} temporary scrape files for topic '{topic}'...")
                 cleaned_count = 0
                 failed_count = 0
                 for fpath in temp_files_to_clean:
                     try:
                         # Double-check path validity and existence before removing
                         if fpath and isinstance(fpath, str) and os.path.isabs(fpath) and os.path.exists(fpath) and config.TEMP_FILE_DIR in fpath:
                              os.remove(fpath)
                              cleaned_count += 1
                              app_logger.debug(f"Removed temp file: {os.path.basename(fpath)}")
                         elif fpath and not os.path.exists(fpath):
                              app_logger.debug(f"Temp file already removed or path invalid (existence check): {fpath}")
                              # Consider counting this as cleaned or skipped? Let's count as cleaned.
                              cleaned_count += 1
                         else:
                             app_logger.warning(f"Skipping cleanup of potentially invalid temp file path: {fpath}")
                             failed_count += 1
                     except OSError as e:
                         app_logger.warning(f"Failed to remove temp file {os.path.basename(fpath)}: {e}")
                         failed_count += 1
                     except Exception as e:
                         app_logger.error(f"Unexpected error removing temp file {fpath}: {e}", exc_info=False)
                         failed_count += 1
                 app_logger.info(f"Temp file cleanup for '{topic}' complete. Removed: {cleaned_count}, Failed/Skipped: {failed_count}. Total tracked: {len(temp_files_to_clean)}")
            else:
                 app_logger.info(f"No temporary files tracked for cleanup for topic '{topic}'.")

            app_logger.info(f"SSE stream processing definitively finished for topic: '{topic}'")
            # Signal stream termination to the client
            try:
                yield format_sse_event({'type': 'stream_terminated'})
            except Exception as yield_err:
                 app_logger.error(f"Failed to yield final stream_terminated message: {yield_err}")

    # --- Set headers for Server-Sent Events (SSE) ---
    # DO NOT set hop-by-hop headers like 'Connection' here. The WSGI server handles it.
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        # 'X-Accel-Buffering': 'no', # For Nginx proxy buffering issues (uncomment if needed)
    }
    # Use stream_with_context to ensure app context is available during generation
    response = Response(stream_with_context(generate_updates()), mimetype='text/event-stream', headers=headers)
    return response


@app.route('/health')
# No @limiter decorator here, so it won't be limited anymore
def health_check():
    """Basic health check endpoint."""
    llm_ok = False
    llm_model_name = config.GOOGLE_MODEL_NAME or "Not Configured"
    try:
        if config.GOOGLE_API_KEY and config.GOOGLE_MODEL_NAME:
            # Optional: Perform a lightweight check like listing models or checking model existence
            # genai.get_model(f'models/{config.GOOGLE_MODEL_NAME}') # Potentially slow/costly
            llm_ok = True # Basic check: Assume OK if configured
    except Exception as e:
        logger.warning(f"Health check: LLM configuration check failed for model '{llm_model_name}': {e}", exc_info=False)

    status = {
        "status": "ok",
        "llm_configured": llm_ok,
        "llm_model": llm_model_name if llm_ok else "Configuration Error"
    }
    http_status = 200 if llm_ok else 503 # Service Unavailable if critical component (LLM) isn't ready

    return jsonify(status), http_status


# --- Favicon Route ---
@app.route('/favicon.ico')
# No @limiter decorator here, so it won't be limited anymore
def favicon():
    # Serve from static folder
    # Use safe join and check existence
    favicon_path = os.path.join(app.root_path, 'static')
    if os.path.exists(os.path.join(favicon_path, 'favicon.ico')):
        return send_from_directory(favicon_path, 'favicon.ico', mimetype='image/vnd.microsoft.icon')
    else:
        # Return No Content if favicon doesn't exist
        return '', 204


# --- Error Handlers ---
@app.errorhandler(429)
def ratelimit_handler(e):
    """Handles rate limit errors."""
    logger.warning(f"Rate limit exceeded: {e.description} from {get_remote_address()}")
    # Render a user-friendly error page
    return render_template('error.html',
                           error_code=429,
                           error_message=f"Too many requests received from your IP address ({e.description}). Please wait a moment and try again."), 429

@app.errorhandler(404)
def not_found_error(error):
    """Handles 404 Not Found errors."""
    logger.warning(f"404 Not Found: {request.path} (Error: {error}) from {get_remote_address()}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handles 500 Internal Server errors."""
    # Log the detailed error with stack trace
    logger.error(f"Internal Server Error (500): Path: {request.path}, Error: {error}", exc_info=True)
    # Return a generic error page/message
    return render_template('500.html'), 500

# Catch-all for other Werkzeug/HTTP exceptions and general Python exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    """Handles unexpected exceptions."""
    from werkzeug.exceptions import HTTPException

    # Log the full exception details
    # Distinguish between HTTP exceptions (like 400 Bad Request) and unexpected server errors
    if isinstance(e, HTTPException):
        # For HTTP exceptions, log as warning unless it's 5xx
        log_level = logging.ERROR if e.code >= 500 else logging.WARNING
        logger.log(log_level, f"HTTP Exception Caught: {e.code} {e.name} for {request.path}. Desc: {e.description}", exc_info=False) # Don't need full stack for standard HTTP errors
        # Use the exception's default response rendering
        return e
    else:
        # For non-HTTP exceptions, log as critical error with full traceback
        logger.error(f"Unhandled Server Exception: {e}", exc_info=True)
        # Return the generic 500 error page
        return render_template('500.html'), 500

# Note: No `if __name__ == '__main__':` block. Use Gunicorn or Waitress.