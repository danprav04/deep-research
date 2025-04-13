# --- File: deep_research_app/app.py ---
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
from werkzeug.middleware.proxy_fix import ProxyFix # <-- IMPORT ProxyFix
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
            os.makedirs(log_dir, exist_ok=True) # Use exist_ok=True
            print(f"INFO: Ensured log directory exists: {log_dir}")
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
    werkzeug_logger.setLevel(logging.WARNING if log_level <= logging.INFO else log_level) # Keep Werkzeug less verbose unless debugging

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
app.secret_key = config.FLASK_SECRET_KEY # Will raise error from config if not set
# IMPORTANT: Apply ProxyFix *before* initializing extensions that use client address
# Assumes Caddy is setting X-Forwarded-For=1, X-Forwarded-Proto=1
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
logger.info("ProxyFix middleware applied (x_for=1, x_proto=1).")

# --- Configure Rate Limiting ---
try:
    limiter = Limiter(
        # Uses the client IP identified by ProxyFix
        key_func=get_remote_address,
        app=app,
        # No default_limits - limits are applied explicitly via decorators
        storage_uri="memory://", # Consider "redis://" for multi-process prod setups
        strategy="fixed-window",
        headers_enabled=True # Useful for clients to see limit status
    )
    # Use the actual configured limit value in the log message
    logger.info(f"Rate limiting configured. Limits applied via route decorators (value: '{config.DEFAULT_RATE_LIMIT}'). Storage: memory://")
except Exception as e:
    logger.error(f"Failed to initialize Flask-Limiter: {e}", exc_info=True)
    # For production, consider raising an error or having a clear fallback
    # For now, log the error and continue (limiter might not function)


# --- Initialize Google Generative AI Client ---
# Validation happens in config.py, critical errors raise ValueError there
try:
     genai.configure(api_key=config.GOOGLE_API_KEY)
     logger.info(f"Google Generative AI configured with model: {config.GOOGLE_MODEL_NAME}")
except Exception as e:
     # Log critical error and raise runtime error to prevent app start
     logger.critical(f"FATAL: Failed to configure Google Generative AI: {e}", exc_info=True)
     # No need to raise here, config.py already did
     # raise RuntimeError(f"FATAL: Failed to configure Google Generative AI: {e}")


# --- Flask Routes ---

@app.route('/')
# No rate limit applied to the index page
def index():
    """Displays the main input form."""
    return render_template('index.html', pico_css=config.PICO_CSS_CDN)

@app.route('/research', methods=['POST'])
@limiter.limit(config.DEFAULT_RATE_LIMIT) # Apply specific limit from config
def research_start():
    """Validates topic and redirects to the results page."""
    topic = request.form.get('topic', '').strip()

    if not topic:
        logger.warning("Research request received with empty topic.")
        # Consider adding flash messages for better UX
        return redirect(url_for('index'))

    MAX_TOPIC_LEN = 300 # Keep reasonable topic length limit
    if len(topic) > MAX_TOPIC_LEN:
        logger.warning(f"Research request received with long topic ({len(topic)} chars). Truncating.")
        topic = topic[:MAX_TOPIC_LEN].strip() # Truncate

    # Encode safely for URL
    encoded_topic = quote(topic, safe='')
    logger.info(f"Redirecting to results page for topic: '{topic}' (Encoded: {encoded_topic})")
    # Render the results page directly, passing the topic for display
    return render_template('results.html',
                           topic=topic,
                           encoded_topic=encoded_topic,
                           pico_css=config.PICO_CSS_CDN)


@app.route('/stream')
@limiter.limit(config.DEFAULT_RATE_LIMIT) # Apply specific limit from config
def stream():
    """The main SSE route that performs research and streams progress."""
    encoded_topic = request.args.get('topic', '')
    topic = ""
    try:
        topic = unquote(encoded_topic)
        if not topic.strip():
             raise ValueError("Decoded topic is empty or whitespace.")
        # Add basic length check here too?
        if len(topic) > 300: # Match the POST limit
             logger.warning(f"Stream requested with long topic ({len(topic)} chars). Using truncated topic.")
             topic = topic[:300].strip()
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
                            # Ensure filepath is a string and looks like an absolute path within the configured temp dir
                            if (isinstance(filepath, str) and
                                os.path.isabs(filepath) and
                                os.path.normpath(filepath).startswith(os.path.normpath(config.TEMP_FILE_DIR))):
                                temp_files_to_clean.append(filepath)
                                app_logger.debug(f"Tracking temp file for cleanup: {os.path.basename(filepath)}")
                            else:
                                app_logger.warning(f"Received potentially invalid or unsafe temp_filepath in scrape_success: {filepath}")

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
                         # Re-validate path safety before removing
                         if (fpath and isinstance(fpath, str) and os.path.isabs(fpath) and
                             os.path.normpath(fpath).startswith(os.path.normpath(config.TEMP_FILE_DIR)) and
                             os.path.exists(fpath)):
                              os.remove(fpath)
                              cleaned_count += 1
                              app_logger.debug(f"Removed temp file: {os.path.basename(fpath)}")
                         elif fpath and not os.path.exists(fpath):
                              app_logger.debug(f"Temp file already removed or path invalid (existence check): {fpath}")
                              cleaned_count += 1 # Count as cleaned if it's already gone
                         else:
                             app_logger.warning(f"Skipping cleanup of potentially invalid/unsafe temp file path: {fpath}")
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
        # For Nginx proxy buffering issues, ensure proxy_buffering off; in Caddyfile:
        # reverse_proxy localhost:8000 {
        #    flush_interval -1
        # }
        # 'X-Accel-Buffering': 'no', # Uncomment if explicitly needed for Nginx
    }
    # Use stream_with_context to ensure app context is available during generation
    response = Response(stream_with_context(generate_updates()), mimetype='text/event-stream', headers=headers)
    return response


@app.route('/health')
@limiter.exempt # Exempt health check from rate limiting
def health_check():
    """Basic health check endpoint."""
    llm_ok = False
    llm_model_name = config.GOOGLE_MODEL_NAME or "Not Configured"
    try:
        # A more robust check could involve a small, cheap API call if feasible,
        # but for now, configuration check is sufficient.
        if config.GOOGLE_API_KEY and config.GOOGLE_MODEL_NAME:
            # Basic check: Assume OK if configured; avoids cost/quota usage.
            llm_ok = True
            # Potential future check (use cautiously):
            # genai.list_models() # Example: Simple check if API key is valid
    except Exception as e:
        logger.warning(f"Health check: LLM configuration or basic check failed for model '{llm_model_name}': {e}", exc_info=False)

    status = {
        "status": "ok",
        "llm_configured": llm_ok,
        "llm_model": llm_model_name if llm_ok else "Configuration Error or Check Failed"
    }
    http_status = 200 if llm_ok else 503 # Service Unavailable if critical component isn't ready

    return jsonify(status), http_status


# --- Favicon Route ---
@app.route('/favicon.ico')
@limiter.exempt # Exempt favicon from rate limiting
def favicon():
    # Serve from static folder using safe_join and checking existence
    try:
        favicon_path = os.path.join(app.root_path, 'static')
        # Use send_from_directory for security (prevents path traversal)
        return send_from_directory(favicon_path, 'favicon.ico', mimetype='image/vnd.microsoft.icon')
    except FileNotFoundError:
        # Return No Content if favicon doesn't exist
        return '', 204


# --- Error Handlers ---
@app.errorhandler(429)
def ratelimit_handler(e):
    """Handles rate limit errors."""
    remote_addr = get_remote_address()
    logger.warning(f"Rate limit exceeded: {e.description} from {remote_addr}")
    return render_template('error.html',
                           error_code=429,
                           error_message=f"Too many requests received from your IP address ({e.description}). Please wait a moment and try again.",
                           pico_css=config.PICO_CSS_CDN), 429

@app.errorhandler(404)
def not_found_error(error):
    """Handles 404 Not Found errors."""
    remote_addr = get_remote_address() # Get IP via key_func
    logger.warning(f"404 Not Found: {request.path} (Error: {error}) from {remote_addr}")
    return render_template('404.html', pico_css=config.PICO_CSS_CDN), 404

@app.errorhandler(500)
def internal_error(error):
    """Handles 500 Internal Server errors."""
    remote_addr = get_remote_address()
    logger.error(f"Internal Server Error (500): Path: {request.path}, Error: {error} from {remote_addr}", exc_info=True)
    return render_template('500.html', pico_css=config.PICO_CSS_CDN), 500

# Catch-all for other Werkzeug/HTTP exceptions and general Python exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    """Handles unexpected exceptions."""
    from werkzeug.exceptions import HTTPException

    remote_addr = get_remote_address()
    if isinstance(e, HTTPException):
        log_level = logging.ERROR if e.code >= 500 else logging.WARNING
        logger.log(log_level, f"HTTP Exception Caught: {e.code} {e.name} for {request.path}. Desc: {e.description}. From: {remote_addr}", exc_info=False)
        # Render specific templates for common HTTP errors
        if e.code == 404:
             return render_template('404.html', pico_css=config.PICO_CSS_CDN), 404
        elif e.code == 429:
             return render_template('error.html',
                                   error_code=429,
                                   error_message=f"Too many requests received from your IP address ({e.description}). Please wait.",
                                   pico_css=config.PICO_CSS_CDN), 429
        elif e.code >= 500:
             return render_template('500.html', pico_css=config.PICO_CSS_CDN), e.code
        else:
            # For other client errors (4xx) render the generic error template
             return render_template('error.html',
                                   error_code=e.code,
                                   error_message=e.description or e.name or "An error occurred processing your request.",
                                   pico_css=config.PICO_CSS_CDN), e.code
        # return e # Or fallback to Werkzeug's default response if preferred
    else:
        # For non-HTTP exceptions, log as critical error with full traceback
        logger.error(f"Unhandled Server Exception: {e} for {request.path}. From: {remote_addr}", exc_info=True)
        # Return the generic 500 error page
        return render_template('500.html', pico_css=config.PICO_CSS_CDN), 500


# --- Production Deployment Note ---
# Use a production WSGI server like Gunicorn or Waitress.
# Example Gunicorn command (adjust workers based on CPU cores):
# gunicorn --workers 3 --bind 0.0.0.0:8000 --log-level info --access-logfile - --error-logfile - deep_research_app.app:app
# Example Waitress command:
# waitress-serve --host 0.0.0.0 --port 8000 --call deep_research_app.app:app

# Add this block for direct execution (e.g., python app.py for development)
if __name__ == '__main__':
    # Use waitress for simple development serving
    # Note: For true production, use Gunicorn/Waitress via command line as shown above
    try:
        from waitress import serve
        host = os.getenv("FLASK_HOST", "127.0.0.1")
        port = int(os.getenv("FLASK_PORT", 8000))
        dev_log_level = os.getenv("LOG_LEVEL", "DEBUG").upper() # Use DEBUG for dev server
        print(f"--- Development Server ---")
        print(f"Starting Waitress server on http://{host}:{port}")
        print(f"Log Level: {dev_log_level}")
        print(f"Note: For production, run using Gunicorn or Waitress directly.")
        print(f"--------------------------")
        # Set Flask app logger level for dev
        logging.getLogger('flask.app').setLevel(dev_log_level)
        serve(app, host=host, port=port)
    except ImportError:
        print("Waitress not found. Running with Flask's development server (not recommended for production).")
        app.run(debug=(dev_log_level == 'DEBUG'), host=host, port=port)
    except Exception as run_err:
        logger.critical(f"Failed to start development server: {run_err}", exc_info=True)
        sys.exit(1)