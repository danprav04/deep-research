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
    const encodedTopic = document.body.dataset.encodedTopic || "";

    // --- <<< DEBUGGING LOG >>> ---
    // Check if marked and DOMPurify are available when the script runs
    console.log('results.js executing. Checking libraries:');
    console.log('typeof marked:', typeof marked);
    console.log('typeof DOMPurify:', typeof DOMPurify);
    // --- <<< END DEBUGGING LOG >>> ---

    // --- Helper Functions ---

    function addProgress(message, isError = false, isFatal = false) {
        if (!progressList) return;
        const li = document.createElement('li');
        li.textContent = message; // Basic escaping
        if (isFatal) {
            li.classList.add('fatal-error');
        } else if (isError) {
            li.classList.add('error');
        }
        progressList.appendChild(li);
        progressList.scrollTop = progressList.scrollHeight; // Auto-scroll
    }

    function addCitationTooltips() {
        setTimeout(() => {
            if (!reportDisplayDiv) return;
            const citationLinks = reportDisplayDiv.querySelectorAll('sup a[href^="#fn:"]');
            let appliedCount = 0;
            citationLinks.forEach(link => {
                try {
                    const footnoteId = decodeURIComponent(link.getAttribute('href')?.substring(1));
                    if (!footnoteId) return;
                    const footnote = document.getElementById(footnoteId);
                    if (footnote) {
                        const clone = footnote.cloneNode(true);
                        const backlink = clone.querySelector('a[href^="#fnref:"]');
                        if (backlink) { backlink.remove(); }
                        let footnoteContent = clone.textContent?.replace(/\s+/g, ' ').trim() || '';
                        const maxTooltipLength = 250;
                        if (footnoteContent.length > maxTooltipLength) {
                            footnoteContent = footnoteContent.substring(0, maxTooltipLength) + '...';
                        }
                        const tooltipText = footnoteContent || 'Citation details unavailable.';
                        link.setAttribute('data-tooltip', tooltipText);
                        link.setAttribute('aria-label', `Citation: ${tooltipText}`);
                        appliedCount++;
                    } else {
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
        }, 150); // Delay
    }

    function updateStreamOutput(target, content) {
        const element = target === 'synthesis' ? synthesisOutput : reportOutput;
        if (element) {
            element.textContent += content;
            element.scrollTop = element.scrollHeight; // Auto-scroll
        }
    }

    function clearStreamOutputs() {
        if (synthesisOutput) synthesisOutput.textContent = "";
        if (reportOutput) reportOutput.textContent = "";
        if (synthesisOutputContainer) synthesisOutputContainer.classList.remove('hidden');
        if (reportOutputContainer) reportOutputContainer.classList.remove('hidden');
        if (reportContainer) reportContainer.classList.remove('hidden');
    }

    function hideStreamOutputs() {
        if (reportContainer) reportContainer.classList.add('hidden');
    }

    function showFinalReport(reportContent) {
        hideStreamOutputs();
        if (finalReportDisplay) finalReportDisplay.classList.remove('hidden');
        if (reportDisplayDiv) {
            try {
                let finalHtml = "<article><p><em>Error: Report content is empty or invalid.</em></p></article>";

                if (reportContent && typeof reportContent === 'string') {
                    // --- Check if marked library is loaded ---
                    if (typeof marked === 'function' && typeof DOMPurify === 'object') {
                        const rawHtml = marked.parse(reportContent);
                        finalHtml = DOMPurify.sanitize(rawHtml, {
                            USE_PROFILES: { html: true },
                            ADD_ATTR: ['data-tooltip', 'aria-label', 'role', 'id', 'href'], // Allow id/href for footnotes
                            ADD_TAGS: ['sup', 'section', 'div', 'li', 'ol', 'ul', 'a', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'em', 'code', 'pre', 'blockquote'], // Common Markdown elements + footnote structure
                            ALLOW_DATA_ATTR: true,
                            FORCE_BODY: false // Let PicoCSS handle body styles
                        });
                    } else {
                         // --- This is the path that was likely taken before ---
                        console.error("Marked.js or DOMPurify library not loaded correctly. Cannot render Markdown safely.");
                        console.error('Current status - typeof marked:', typeof marked, 'typeof DOMPurify:', typeof DOMPurify);
                        addProgress("Error: Markdown renderer or sanitizer is missing. Cannot display final report correctly.", true);
                        finalHtml = `<article><p><em>Error: Markdown renderer is missing. Displaying raw content:</em></p><pre><code>${escapeHtml(reportContent)}</code></pre></article>`;
                    }
                }

                reportDisplayDiv.innerHTML = finalHtml;
                addCitationTooltips(); // Apply tooltips AFTER content is injected

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
        showFinalReport(eventData.report_html); // Assumes report_html contains Markdown
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
        if (finalReportDisplay && finalReportDisplay.classList.contains('hidden')) {
            addProgress("Process stopped or terminated before final report was generated. Check logs for details.", true, true);
            if (reportDisplayDiv && !reportDisplayDiv.innerHTML.trim()) {
                reportDisplayDiv.innerHTML = "<article><p><em>Report generation was interrupted or failed.</em></p></article>";
                if (finalReportDisplay) finalReportDisplay.classList.remove('hidden');
            }
        }
    }

    function connectSSE() {
        if (eventSource) {
            eventSource.close();
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
            if (progressList) progressList.innerHTML = '';
            addProgress("Connection established. Starting research process...");
            clearStreamOutputs();
            if (finalReportDisplay) finalReportDisplay.classList.add('hidden');
            if (reportDisplayDiv) reportDisplayDiv.innerHTML = '<p><em>Loading final report...</em></p>';
            if (loader) loader.classList.remove('hidden');
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
                            if (eventSource) eventSource.close();
                            if (reportDisplayDiv && finalReportDisplay?.classList.contains('hidden')) {
                                reportDisplayDiv.innerHTML = `<article><p><em>Report generation failed due to a fatal error: ${escapeHtml(eventData.message)}</em></p></article>`;
                                finalReportDisplay.classList.remove('hidden');
                                hideStreamOutputs();
                            }
                        }
                        break;
                    case 'stream_start':
                        const targetStart = eventData.target === 'synthesis' ? synthesisOutput : reportOutput;
                        const streamStatus = eventData.target === 'synthesis'
                            ? synthesisOutputContainer?.querySelector('.stream-status')
                            : reportOutputContainer?.querySelector('.stream-status');
                        if (streamStatus) streamStatus.innerHTML = `<em>Streaming ${eventData.target}... (Content will appear below)</em>`;
                        if(targetStart) targetStart.textContent = '';
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
            const lastProgress = progressList?.lastElementChild?.textContent || "";
            if (!lastProgress.includes("fatal") && !lastProgress.includes("Error connecting")) {
                 addProgress("Error connecting to the research stream. Server might be down or restarting. Check connection or try again.", true, true);
            }
            if (loader) loader.classList.add('hidden');
            hideStreamOutputs();
            if (finalReportDisplay?.classList.contains('hidden')) {
                addProgress("Could not complete research due to connection failure.", true, true);
                if (reportDisplayDiv) {
                    reportDisplayDiv.innerHTML = "<article><p><em>Failed to connect to the research service. Please check logs or try again later.</em></p></article>";
                    finalReportDisplay.classList.remove('hidden');
                }
            }
            // Browser might auto-retry, don't close here unless specific errors warrant it.
        };
    }

    // --- Initial Setup ---
    // Check libraries are loaded *before* connecting
     if (typeof marked !== 'function' || typeof DOMPurify !== 'object') {
         console.error("Required libraries (marked.js or DOMPurify) not found on initial load. Report rendering will likely fail.");
         addProgress("Error: Essential rendering library (marked.js or DOMPurify) failed to load. Please check network connection and browser console.", true, true);
         // Optionally prevent connection if libraries are critical
         // if (loader) loader.classList.add('hidden');
         // return;
     }

    connectSSE(); // Start the SSE connection

    window.addEventListener('beforeunload', () => {
        if (eventSource) {
            eventSource.close();
            console.log("SSE connection closed on page unload.");
        }
    });
});