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
            const citationLinks = reportDisplayDiv.querySelectorAll('sup a[href^="#fn:"]');
            let appliedCount = 0;
            citationLinks.forEach(link => {
                const footnoteId = link.getAttribute('href')?.substring(1); // Use optional chaining
                if (!footnoteId) return;

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
            });
            console.log(`Citation tooltips applied to ${appliedCount} links.`);
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

    function showFinalReport(htmlContent) {
        hideStreamOutputs();
        if (finalReportDisplay) finalReportDisplay.classList.remove('hidden');
        if (reportDisplayDiv) {
             // Use DOMPurify here if you were allowing more complex HTML from Markdown
             // For now, assuming trusted Markdown conversion on the server
             reportDisplayDiv.innerHTML = htmlContent || "<article><p><em>Error: Received empty report content.</em></p></article>";
             addCitationTooltips(); // Apply tooltips AFTER content is injected
        }
    }

    function handleCompletion(eventData) {
        addProgress("Research complete. Processing final results...");
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
                        }
                        break;
                    case 'stream_start':
                        // Optional: Add status message above the specific stream output
                        const targetStart = eventData.target === 'synthesis' ? synthesisOutput : reportOutput;
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
            addProgress("Error connecting to the research stream. Server might be down, busy, or restarting. Please try again later.", true, true);
            if (loader) loader.classList.add('hidden');
            hideStreamOutputs(); // Hide stream outputs on connection error
            if (eventSource) {
                 // Don't automatically close on error, browser might retry
                 // eventSource.close();
            }
             // Indicate failure if report wasn't shown
            if (finalReportDisplay && finalReportDisplay.classList.contains('hidden')) {
                 addProgress("Could not complete research due to connection failure.", true, true);
            }
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