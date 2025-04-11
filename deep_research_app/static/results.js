// results.js - Frontend logic for the research results page

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const progressList = document.getElementById('progress-list');
    const loader = document.getElementById('loader');
    const reportContainer = document.getElementById('report-container'); // Contains both stream outputs
    const synthesisOutputContainer = document.getElementById('synthesis-output-container');
    const reportOutputContainer = document.getElementById('report-output-container');
    const synthesisOutput = document.getElementById('synthesis-output');
    const reportOutput = document.getElementById('report-output');
    const finalReportDisplay = document.getElementById('final-report-display');
    const reportDisplayDiv = document.getElementById('report-display');

    // --- State ---
    let eventSource = null;
    // Read encoded topic from a data attribute on the body or a specific element
    const encodedTopic = document.body.dataset.encodedTopic || "";

    // --- Helper Functions ---

    function addProgress(message, isError = false, isFatal = false) {
        if (!progressList) return;
        const li = document.createElement('li');
        // Basic escaping to prevent HTML injection in progress messages
        li.textContent = message;
        if (isFatal) {
            li.classList.add('fatal-error');
        } else if (isError) {
            li.classList.add('error');
        }
        progressList.appendChild(li);
        // Auto-scroll to the bottom
        progressList.scrollTop = progressList.scrollHeight;
    }

    // Citation Tooltip Functionality using PicoCSS data-tooltip
    function addCitationTooltips() {
        // Add a small delay to ensure DOM is fully updated after HTML injection
        setTimeout(() => {
            if (!reportDisplayDiv) return;
            // Select links within the specific report display area
            const citationLinks = reportDisplayDiv.querySelectorAll('sup a[href^="#fn:"]');
            let appliedCount = 0;
            citationLinks.forEach(link => {
                try {
                    // Decode URI component in case footnote IDs have special characters
                    const footnoteId = decodeURIComponent(link.getAttribute('href')?.substring(1));
                    if (!footnoteId) return;

                    // Find the footnote element using the decoded ID
                    const footnote = document.getElementById(footnoteId);
                    if (footnote) {
                        // Clone to avoid modifying the original bibliography
                        const clone = footnote.cloneNode(true);
                        // Remove the backlink (↩) from the tooltip content
                        const backlink = clone.querySelector('a[href^="#fnref:"]');
                        if (backlink) { backlink.remove(); }

                        // Get text content, clean up whitespace, trim
                        let footnoteContent = clone.textContent?.replace(/\s+/g, ' ').trim() || '';

                        // Limit tooltip length
                        const maxTooltipLength = 250;
                        if (footnoteContent.length > maxTooltipLength) {
                             footnoteContent = footnoteContent.substring(0, maxTooltipLength) + '...';
                        }

                        // Set PicoCSS tooltip attribute and ARIA label
                        const tooltipText = footnoteContent || 'Citation details unavailable.';
                        link.setAttribute('data-tooltip', tooltipText);
                        link.setAttribute('aria-label', `Citation: ${tooltipText}`);
                        appliedCount++;
                    } else {
                        // Footnote target not found
                         link.setAttribute('data-tooltip', 'Citation details missing.');
                         link.setAttribute('aria-label', 'Citation details missing.');
                         console.warn(`Footnote target not found for ID: ${footnoteId}`);
                    }
                } catch (e) {
                    console.error(`Error processing citation link ${link.getAttribute('href')}:`, e);
                    link.setAttribute('data-tooltip', 'Error processing citation.');
                    link.setAttribute('aria-label', 'Error processing citation.');
                }
            });
            console.log(`Citation tooltips applied to ${appliedCount} links within report.`);
        }, 150); // Delay of 150ms
    }

    function updateStreamOutput(target, content) {
        const element = target === 'synthesis' ? synthesisOutput : reportOutput;
        if (element) {
            element.textContent += content;
            // Auto-scroll stream output divs
            element.scrollTop = element.scrollHeight;
        }
    }

    function clearStreamOutputs() {
         if (synthesisOutput) synthesisOutput.textContent = "";
         if (reportOutput) reportOutput.textContent = "";
         if (synthesisOutputContainer) synthesisOutputContainer.classList.remove('hidden');
         if (reportOutputContainer) reportOutputContainer.classList.remove('hidden');
         if (reportContainer) reportContainer.classList.remove('hidden'); // Show the main container
    }

    function hideStreamOutputs() {
        if (reportContainer) reportContainer.classList.add('hidden');
        // Can hide individual containers too if needed
        // if (synthesisOutputContainer) synthesisOutputContainer.classList.add('hidden');
        // if (reportOutputContainer) reportOutputContainer.classList.add('hidden');
    }

    // --- MODIFIED FUNCTION ---
    function showFinalReport(reportContent) { // Renamed parameter for clarity
        hideStreamOutputs();
        if (finalReportDisplay) finalReportDisplay.classList.remove('hidden');
        if (reportDisplayDiv) {
            try {
                let finalHtml = "<article><p><em>Error: Report content is empty or invalid.</em></p></article>"; // Default error message

                if (reportContent && typeof reportContent === 'string') {
                    // Check if marked library is loaded
                    if (typeof marked === 'function') {
                         // --- Convert Markdown to HTML using marked.js ---
                         const rawHtml = marked.parse(reportContent);

                         // --- Sanitize the generated HTML using DOMPurify ---
                         // Configure DOMPurify to allow elements and attributes needed by PicoCSS tooltips and footnotes
                         finalHtml = DOMPurify.sanitize(rawHtml, {
                             USE_PROFILES: { html: true }, // Allow standard HTML elements
                             ADD_ATTR: ['data-tooltip', 'aria-label', 'role'], // Allow tooltip/ARIA attributes
                             ADD_TAGS: ['sup', 'section'], // Allow sup for citations, section for footnotes if needed
                             // Ensure IDs are allowed for footnotes/references
                             ALLOW_DATA_ATTR: true, // Allow data-* attributes generally
                             FORCE_BODY: true // Ensure the output is wrapped in a body tag for safety if needed
                         });

                    } else {
                        console.error("Marked.js library not loaded. Cannot render Markdown.");
                        addProgress("Error: Markdown library not loaded. Cannot display final report correctly.", true);
                        // Display raw content as a fallback, but warn the user
                        finalHtml = `<article><p><em>Error: Markdown renderer is missing. Displaying raw content:</em></p><pre><code>${escapeHtml(reportContent)}</code></pre></article>`;
                    }
                }

                reportDisplayDiv.innerHTML = finalHtml;
                addCitationTooltips(); // Apply tooltips AFTER content is injected and sanitized

            } catch (e) {
                 console.error("Error rendering final report:", e);
                 addProgress(`Error displaying final report: ${e.message}`, true);
                 reportDisplayDiv.innerHTML = "<article><p><em>An error occurred while rendering the report. Please check the console.</em></p></article>";
            }

        } else {
            console.error("Final report display area not found.");
            addProgress("Error: Cannot find the area to display the final report.", true);
        }
    }

    // Simple HTML escaping function for fallback display
    function escapeHtml(unsafe) {
        if (!unsafe) return '';
        return unsafe
             .replace(/&/g, "&")
             .replace(/</g, "<")
             .replace(/>/g, ">")
             .replace(/"/g, '"')
             .replace(/'/g, "'");
     }

    function handleCompletion(eventData) {
        addProgress("Research complete. Processing final results...");
        // Pass the content assumed to be Markdown to showFinalReport
        // **IMPORTANT**: If the server *does* send HTML, you'd revert this.
        // If the field name changes on the server (e.g., to 'report_markdown'), update 'report_html' here.
        showFinalReport(eventData.report_html);
        addProgress("Final report displayed. Process finished.");
        if (loader) loader.classList.add('hidden');
        if (eventSource) {
            eventSource.close();
            console.log("SSE connection closed by client after completion.");
        }
    }

    function handleStreamTermination() {
        addProgress("Server stream terminated.", false);
        if (loader) loader.classList.add('hidden');
        if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
            eventSource.close();
            console.log("SSE connection closed by client after termination signal.");
        }
        // Check if the report was displayed; if not, indicate potential incompletion.
        if (finalReportDisplay && finalReportDisplay.classList.contains('hidden')) {
            addProgress("Process stopped or terminated before final report was generated. Check logs for details.", true, true);
            // Ensure report area shows an error if it's empty
             if (reportDisplayDiv && !reportDisplayDiv.innerHTML.trim()) {
                 reportDisplayDiv.innerHTML = "<article><p><em>Report generation was interrupted or failed.</em></p></article>";
                 if (finalReportDisplay) finalReportDisplay.classList.remove('hidden'); // Show the error message area
             }
        }
    }

    // --- SSE Connection ---
    function connectSSE() {
        if (eventSource) {
            eventSource.close(); // Close existing connection if any
        }

        if (!encodedTopic) {
            addProgress("Error: Cannot connect to stream. Topic is missing.", true, true);
            if (loader) loader.classList.add('hidden');
            return;
        }

        const streamUrl = `/stream?topic=${encodedTopic}`;
        console.log(`Connecting to SSE stream at ${streamUrl}`);
        eventSource = new EventSource(streamUrl);

        eventSource.onopen = function() {
            console.log("SSE connection opened.");
            if (progressList) progressList.innerHTML = ''; // Clear previous logs
            addProgress("Connection established. Starting research process...");
            clearStreamOutputs();
            if (finalReportDisplay) finalReportDisplay.classList.add('hidden'); // Hide final report area
            if (reportDisplayDiv) reportDisplayDiv.innerHTML = '<p><em>Loading final report...</em></p>'; // Reset report area
            if (loader) loader.classList.remove('hidden'); // Show loader
        };

        eventSource.onmessage = function(event) {
            try {
                const eventData = JSON.parse(event.data);

                switch (eventData.type) {
                    case 'progress':
                        addProgress(eventData.message);
                        break;
                    case 'error':
                        addProgress(eventData.message, true, eventData.fatal || false);
                        if (eventData.fatal) {
                            addProgress("Research stopped due to fatal error.", true, true);
                            if (loader) loader.classList.add('hidden');
                            if (eventSource) {
                                eventSource.close();
                                console.log("SSE connection closed by client due to fatal server error.");
                            }
                             // Show error in report area if process fails fatally
                            if (reportDisplayDiv && finalReportDisplay && finalReportDisplay.classList.contains('hidden')) {
                                reportDisplayDiv.innerHTML = `<article><p><em>Report generation failed due to a fatal error: ${escapeHtml(eventData.message)}</em></p></article>`;
                                finalReportDisplay.classList.remove('hidden');
                                hideStreamOutputs();
                            }
                        }
                        break;
                    case 'stream_start':
                        // Optional: Add status message above the specific stream output
                        const targetStart = eventData.target === 'synthesis' ? synthesisOutput : reportOutput;
                        const streamStatus = eventData.target === 'synthesis'
                            ? synthesisOutputContainer?.querySelector('.stream-status')
                            : reportOutputContainer?.querySelector('.stream-status');
                        if (streamStatus) streamStatus.innerHTML = `<em>Streaming ${eventData.target}... (Content will appear below)</em>`;
                        if(targetStart) targetStart.textContent = ''; // Clear previous content
                        addProgress(`Starting LLM ${eventData.target} stream...`);
                        break;
                    case 'llm_chunk':
                        updateStreamOutput(eventData.target, eventData.content);
                        break;
                    case 'complete':
                        handleCompletion(eventData);
                        break;
                    case 'stream_terminated':
                        handleStreamTermination();
                        break;
                    default:
                        console.warn("Received unknown SSE event type:", eventData.type, eventData);
                }
            } catch (e) {
                console.error("Error parsing SSE message or processing event:", e, "Data:", event.data);
                addProgress(`Client-side error processing update: ${e.message}`, true);
            }
        };

        eventSource.onerror = function(error) {
            console.error("SSE connection error:", error);
            // Avoid adding duplicate fatal errors if one was already logged
            const lastProgress = progressList?.lastElementChild?.textContent || "";
            if (!lastProgress.includes("fatal") && !lastProgress.includes("Error connecting")) {
                addProgress("Error connecting to the research stream. Server might be down, busy, or restarting. Please check connection or try again later.", true, true);
            }
            if (loader) loader.classList.add('hidden');
            hideStreamOutputs(); // Hide stream outputs on connection error

            // Indicate failure if report wasn't shown and connection truly failed
             if (finalReportDisplay && finalReportDisplay.classList.contains('hidden')) {
                 addProgress("Could not complete research due to connection failure.", true, true);
                 if (reportDisplayDiv) {
                    reportDisplayDiv.innerHTML = "<article><p><em>Failed to connect to the research service. Please ensure it's running and accessible.</em></p></article>";
                    finalReportDisplay.classList.remove('hidden'); // Show the error
                 }
            }
            // Note: The browser might attempt to reconnect automatically depending on the error.
            // We don't explicitly close the eventSource here on *every* error.
        };
    }

    // --- Initial Setup ---
    connectSSE();

    // Optional: Add event listener for page unload to explicitly close SSE
    window.addEventListener('beforeunload', () => {
        if (eventSource) {
            eventSource.close();
            console.log("SSE connection closed on page unload.");
        }
    });

});