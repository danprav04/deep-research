# Deep Research Agent 🤖

[![Project Status: Maintained](https://img.shields.io/badge/status-maintained-brightgreen.svg)](https://github.com/m-a-mo/deep-research/)

An autonomous AI agent that conducts in-depth research on any given topic, generating a comprehensive, well-structured report with citations.

---

### Key Features ✨

*   **🧠 AI-Powered Planning:** Automatically generates a multi-step research plan using Google's Gemini model.
*   **🌐 Automated Web Search:** Conducts targeted web searches based on keywords from the research plan.
*   **✂️ Concurrent Scraping & Sanitization:** Efficiently scrapes content from multiple URLs in parallel and sanitizes the HTML to extract meaningful text.
*   **✍️ AI-Driven Synthesis:** Synthesizes the gathered information, structuring it according to the research plan and citing every piece of data to its source.
*   **📜 Comprehensive Report Generation:** Compiles a final report in Markdown, complete with an introduction, conclusion, and a formatted bibliography with footnote-style citations.
*   **⚡ Real-Time Streaming:** Streams the entire research process to the user in real-time using Server-Sent Events (SSE), from planning to the final report generation.
*   **🎨 Modern Frontend:** Features a clean, responsive user interface built with Pico.css, including light and dark theme support.

---

## How It Works ⚙️

The Deep Research Agent follows a systematic, multi-step process to transform a simple topic into a detailed report.

1.  **User Input & Planning:**
    *   A user provides a research topic (e.g., "The Impact of AI on Journalism").
    *   The **Orchestrator** sends a request to the Gemini LLM to create a logical, step-by-step research plan with relevant search keywords for each step.

2.  **Web Search & Filtering:**
    *   For each step in the plan, the agent uses the generated keywords to perform web searches via the DuckDuckGo Search API.
    *   It collects and de-duplicates all found URLs, filtering out irrelevant domains (like social media) and file types (like PDFs, images).

3.  **Scraping & Sanitization:**
    *   The application uses a concurrent `ThreadPoolExecutor` to scrape multiple URLs at once, maximizing efficiency.
    *   The raw HTML from each page is sanitized using the `Bleach` library to remove scripts, styles, and unnecessary tags, leaving only clean, structured text content. This sanitized content is saved to temporary files.

4.  **Information Synthesis:**
    *   The sanitized content from all sources is compiled into a large context, balanced across the different research steps to ensure comprehensive coverage.
    *   This context, along with the original research plan, is sent back to the Gemini LLM. The LLM is prompted to analyze the information and write a detailed synthesis for each plan step, **critically adding a raw URL citation** to every sentence or data point it extracts.

5.  **Final Report Generation:**
    *   The Markdown synthesis from the previous step is combined with the topic, plan, and a newly generated bibliography map (e.g., `[1]: http://example.com`).
    *   The LLM receives this final package and is prompted to write a complete report, including:
        *   An introduction and conclusion.
        *   Re-writing the synthesized findings into a polished narrative.
        *   Replacing the raw `[Source URL: ...]` citations with clean Markdown footnote links (`[^1]`, `[^2]`, etc.).
        *   A formatted `## Bibliography` section.

6.  **Streaming & Display:**
    *   Throughout this entire process, status updates and live LLM-generated text (for synthesis and the final report) are streamed directly to the browser and displayed in real-time.
    *   The final, server-rendered HTML report is securely sanitized with `DOMPurify.js` on the client-side before being displayed, with interactive tooltips for citations.

---

## 🚀 Getting Started

Follow these instructions to get a local copy of the Deep Research Agent up and running.

### Prerequisites

*   Python 3.8+
*   Pip package manager

### Installation & Setup

1.  **Clone the Repository:**
    ```sh
    git clone https://github.com/m-a-mo/deep-research.git
    cd deep-research
    ```

2.  **Create and Activate a Virtual Environment:**
    *   **Windows:**
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **macOS / Linux:**
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    ```sh
    pip install -r deep_research_app/requirements.txt
    ```

4.  **Configure Environment Variables:**
    *   Create a `.env` file in the project root directory (`/deep-research/.env`).
    *   Copy the contents from the example below and add your API keys.

    > **Important:** The `.env` file is used to store secrets and configuration. It is listed in `.gitignore` and should not be committed to source control.

    **`.env` file contents:**
    ```env
    # --- REQUIRED ---
    # Get your API key from Google AI Studio: https://aistudio.google.com/app/apikey
    GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
    
    # A strong, random secret key for Flask session security
    FLASK_SECRET_KEY="YOUR_SUPER_SECRET_FLASK_KEY"

    # --- OPTIONAL (Defaults are provided in config.py) ---
    # Specify the model to use. 'gemini-1.5-flash-latest' is a good default.
    GOOGLE_MODEL_NAME="gemini-1.5-flash-latest"

    # Set the logging level (DEBUG, INFO, WARNING, ERROR)
    LOG_LEVEL="INFO"

    # Configure the rate limit for research requests
    DEFAULT_RATE_LIMIT="1 per minute"
    ```

5.  **Run the Application:**
    *   The project is configured to run with a production-ready WSGI server like `Waitress`.
    *   From the project root directory, run:
        ```sh
        python deep_research_app/app.py
        ```
    *   The application will be available at `http://127.0.0.1:8000`.

---

## 🔧 Configuration

You can customize the agent's behavior by setting environment variables in the `.env` file. The most important ones are listed below. See `deep_research_app/config.py` for all available options.

| Variable                          | Description                                                                 | Default Value                    |
| --------------------------------- | --------------------------------------------------------------------------- | -------------------------------- |
| `GOOGLE_API_KEY`                  | **Required.** Your Google Gemini API Key.                                   | `None`                           |
| `FLASK_SECRET_KEY`                | **Required.** A secret for signing session cookies.                         | `None`                           |
| `GOOGLE_MODEL_NAME`               | The specific Gemini model to use for all tasks.                             | `gemini-1.5-flash-latest`        |
| `LOG_LEVEL`                       | The logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).                  | `INFO`                           |
| `DEFAULT_RATE_LIMIT`              | Rate limit for starting new research jobs per IP address.                     | `1 per minute`                   |
| `MAX_TOTAL_URLS_TO_SCRAPE`        | The absolute maximum number of unique URLs to scrape for a single research.   | `100`                            |
| `MAX_WORKERS`                     | Number of concurrent threads for scraping URLs.                             | `3`                              |
| `TARGET_MAX_CONTEXT_TOKENS`       | The target token limit for the context sent to the LLM for synthesis.         | `700000`                         |
| `LLM_TEMPERATURE`                 | Controls the creativity of the LLM (0.0=deterministic, 1.0=creative).         | `0.5`                            |

---

## 📂 Project Structure

```
/
├── deep_research_app/
│   ├── static/               # CSS, JavaScript, and other static assets
│   │   ├── custom.css
│   │   └── results.js
│   ├── templates/            # Flask HTML templates
│   │   ├── 404.html
│   │   ├── 500.html
│   │   ├── error.html
│   │   ├── index.html
│   │   └── results.html
│   ├── app.py                # Main Flask application, routes, and SSE handler
│   ├── config.py             # Application configuration from environment variables
│   ├── llm_interface.py      # Handles all communication with the Gemini LLM
│   ├── requirements.txt      # Python dependencies
│   ├── research_orchestrator.py # Core logic for the entire research process
│   ├── utils.py              # Helper functions (parsing, sanitizing, etc.)
│   └── web_research.py       # Functions for web searching and scraping
├── tests/                    # Unit and integration tests
│   ├── test_app.py
│   └── ...
├── .gitignore
├── create_context.py         # Utility script to generate project context
└── README.md                 # This file
```

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

1.  Fork the Project.
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the Branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Acknowledgments

*   [**Pico.css**](https://picocss.com/) for the class-less CSS framework.
*   [**Google Gemini**](https://deepmind.google/technologies/gemini/) for the powerful generative AI capabilities.
*   [**DOMPurify**](https://github.com/cure53/DOMPurify) for robust client-side HTML sanitization.
