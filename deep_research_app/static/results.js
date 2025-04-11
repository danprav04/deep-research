// results.js - Frontend logic for the research results page

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const progressList = document.getElementById('progress-list');
    const loader = document.getElementById('loader');
    const reportContainer = document.getElementById('report-container');
    const synthesisOutputContainer = document.getElementById('synthesis-output-container');
    const reportOutputContainer = document.getElementById('report-output-container');
    const synthesisOutput = document.getElementById('synthesis-output');
    const reportOutput = document.getElementById('report-output');
    const finalReportDisplay = document.getElementById('final-report-display');
    const reportDisplayDiv = document.getElementById('report-display');

    // --- State ---
    let eventSource = null;
    const encodedTopic = document.body.dataset.encodedTopic || "";

    // --- DEBUGGING LOG ---
    console.log('results.js executing. Checking libraries:');
    // Marked.js might not be needed anymore if conversion is purely server-side
    // console.log('typeof marked:', typeof marked);
    console.log('typeof DOMPurify:', typeof DOMPurify);
    // --- END DEBUGGING LOG ---

    // --- Helper Functions ---

    function addProgress(message, isError = false, isFatal = false) {
        if (!progressList) return;
        const li = document.createElement('li');
        // Sanitize simple text message before setting textContent (belt-and-suspenders)
        li.textContent = message.toString(); // Ensure it's a string
        if (isFatal) {
            li.classList.add('fatal-error');
        } else if (isError) {
            li.classList.add('error');
        }
        progressList.appendChild(li);
        progressList.scrollTop = progressList.scrollHeight;
    }

    function linkifyBibliographyUrls() {
        if (!reportDisplayDiv) return;
        const bibliographyItems = reportDisplayDiv.querySelectorAll('.footnotes ol li');
        const urlRegex = /(https?:\/\/[^\s<>"']+)/g; // Regex to find URLs

        bibliographyItems.forEach(item => {
            try {
                // Find the paragraph within the list item, which usually contains the URL
                const paragraph = item.querySelector('p');
                const targetElement = paragraph || item; // Fallback to the item itself

                // Get the current HTML, process it, and set it back
                let currentHtml = targetElement.innerHTML;
                let replaced = false;

                // Find URLs and replace them with anchor tags
                currentHtml = currentHtml.replace(urlRegex, (url) => {
                    // Avoid replacing URLs that are already inside an anchor tag's href
                    // This is a simple check, might need refinement for complex cases
                    const surroundingChars = currentHtml.substring(
                        Math.max(0, currentHtml.indexOf(url) - 10),
                        currentHtml.indexOf(url)
                    );
                    if (surroundingChars.includes('href="')) {
                        return url; // Already linked, don't replace
                    }
                    replaced = true;
                    // Create a safe anchor tag string
                    // Basic escaping for the URL in href attribute
                    const escapedUrl = url.replace(/"/g, '"');
                    return `<a href="${escapedUrl}" target="_blank" rel="noopener noreferrer">${url}</a>`;
                });

                if (replaced) {
                    targetElement.innerHTML = currentHtml; // Update the HTML content
                }
            } catch (e) {
                console.error(`Error linkifying URL in bibliography item: ${item.id || 'unknown id'}`, e);
            }
        });
        if (bibliographyItems.length > 0) {
            console.log(`Bibliography URL linkification applied to ${bibliographyItems.length} items.`);
        }
    }


    function addCitationTooltips() {
        // Slight delay to ensure DOM is fully ready after innerHTML update
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
                        // Clone the footnote content to avoid modifying the original
                        const clone = footnote.cloneNode(true);
                        // Remove the back-reference link ('↩') from the cloned content
                        const backlink = clone.querySelector('a[href^="#fnref:"]');
                        if (backlink) { backlink.remove(); }
                        // Get cleaned text content, trim whitespace
                        let footnoteContent = clone.textContent?.replace(/\s+/g, ' ').trim() || '';
                        // Truncate long tooltip content
                        const maxTooltipLength = 250;
                        if (footnoteContent.length > maxTooltipLength) {
                            footnoteContent = footnoteContent.substring(0, maxTooltipLength) + '...';
                        }
                        // Use PicoCSS tooltip attribute
                        const tooltipText = footnoteContent || 'Citation details unavailable.';
                        link.setAttribute('data-tooltip', tooltipText);
                        link.setAttribute('aria-label', `Citation: ${tooltipText}`); // For accessibility
                        appliedCount++;
                    } else {
                         // Handle cases where the footnote target li might be missing
                        link.setAttribute('data-tooltip', 'Citation details missing.');
                        link.setAttribute('aria-label', 'Citation details missing.');
                        console.warn(`Footnote target not found for ID: ${footnoteId}`);
                    }
                } catch (e) {
                    console.error(`Error processing citation link ${link.getAttribute('href')}:`, e);
                    // Add fallback tooltip on error
                    link.setAttribute('data-tooltip', 'Error processing citation.');
                    link.setAttribute('aria-label', 'Error processing citation.');
                }
            });
            if (citationLinks.length > 0) {
                console.log(`Citation tooltips applied to ${appliedCount}/${citationLinks.length} links within report.`);
            }
        }, 150); // Delay to allow DOM updates
    }

    function updateStreamOutput(target, content) {
        const element = target === 'synthesis' ? synthesisOutput : reportOutput;
        if (element) {
            // Append text content directly, assuming server sends plain text chunks
            element.textContent += content;
            // Scroll to the bottom
            element.scrollTop = element.scrollHeight;
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

    // Shows the final report, assuming server sends HTML
    function showFinalReport(reportHtmlContent) {
        hideStreamOutputs();
        if (finalReportDisplay) finalReportDisplay.classList.remove('hidden');
        if (reportDisplayDiv) {
            try {
                let finalHtml = "<article><p><em>Error: Received empty report content.</em></p></article>"; // Default

                if (reportHtmlContent && typeof reportHtmlContent === 'string') {
                    // Check if the server sent the raw markdown fallback message
                    const looksLikeRawFallback = reportHtmlContent.includes("Raw Markdown content:</p><pre><code>");

                    if (looksLikeRawFallback) {
                        console.warn("Server indicated Markdown->HTML conversion failed. Displaying server-provided fallback HTML.");
                        // Sanitize the fallback HTML (which includes escaped raw markdown)
                        finalHtml = DOMPurify.sanitize(reportHtmlContent, { USE_PROFILES: { html: true } });
                    } else {
                        // Assume it's intended HTML from the server (already converted)
                        console.log("Received report content, assuming pre-converted HTML. Sanitizing...");
                        // Sanitize the received HTML before displaying
                        finalHtml = DOMPurify.sanitize(reportHtmlContent, {
                            USE_PROFILES: { html: true }, // Allows standard HTML tags
                            ADD_ATTR: ['data-tooltip', 'aria-label', 'role', 'id', 'href', 'target', 'rel'], // Allow necessary attributes
                            ADD_TAGS: ['sup', 'section', 'article', 'aside', 'nav', 'header', 'footer', 'figure', 'figcaption', 'details', 'summary', 'main'], // Allow common semantic tags
                            ALLOW_DATA_ATTR: true, // Needed for data-tooltip
                            FORCE_BODY: false // Don't wrap in body
                        });
                        console.log("Sanitization complete.");
                    }
                }

                console.log("Setting innerHTML for report-display...");
                reportDisplayDiv.innerHTML = finalHtml; // Inject the sanitized HTML
                console.log("Finished setting innerHTML.");

                // Apply post-rendering enhancements
                addCitationTooltips();
                linkifyBibliographyUrls(); // Make bibliography URLs clickable

            } catch (e) {
                 console.error("Error processing or displaying final report:", e);
                 addProgress(`Error displaying final report: ${e.message}`, true);
                 reportDisplayDiv.innerHTML = "<article><p><em>An error occurred while rendering the report. Please check the console.</em></p></article>";
            }
        } else {
            console.error("Final report display area not found.");
            addProgress("Error: Cannot find the area to display the final report.", true);
        }
    }

    // Basic HTML escaping (kept for potential other uses)
    function escapeHtml(unsafe) {
        if (!unsafe) return '';
        return unsafe
            .replace(/&/g, "&") // Corrected basic escaping
            .replace(/</g, "<")
            .replace(/>/g, ">")
            .replace(/"/g, '"')
            .replace(/'/g, "'");
    }

    function handleCompletion(eventData) {
        addProgress("Research complete. Processing final results...");
        // Pass the report_html content directly
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
        // Check if the final report area is still hidden (meaning process stopped early)
        if (finalReportDisplay && finalReportDisplay.classList.contains('hidden')) {
            addProgress("Process stopped or terminated before final report was generated. Check logs for details.", true, true);
             // Show an error message in the report area if it's empty
            if (reportDisplayDiv && !reportDisplayDiv.innerHTML.trim().includes("<em>Report generation failed")) {
                reportDisplayDiv.innerHTML = "<article><p><em>Report generation was interrupted or failed.</em></p></article>";
                if (finalReportDisplay) finalReportDisplay.classList.remove('hidden'); // Show the error area
            }
        }
    }

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
            if (progressList) progressList.innerHTML = ''; // Clear previous progress
            addProgress("Connection established. Starting research process...");
            clearStreamOutputs(); // Show and clear streaming areas
            if (finalReportDisplay) finalReportDisplay.classList.add('hidden'); // Hide final report area
            if (reportDisplayDiv) reportDisplayDiv.innerHTML = '<p><em>Loading final report...</em></p>'; // Placeholder
            if (loader) loader.classList.remove('hidden'); // Show loader
        };

        eventSource.onmessage = function(event) {
            try {
                const eventData = JSON.parse(event.data);
                // console.log("SSE Received:", eventData.type); // Optional: Log event types

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
                            // Display fatal error in the report area if it's still hidden
                            if (reportDisplayDiv && finalReportDisplay?.classList.contains('hidden')) {
                                reportDisplayDiv.innerHTML = `<article><h2>Fatal Error</h2><p><em>Report generation failed: ${escapeHtml(eventData.message)}</em></p></article>`;
                                finalReportDisplay.classList.remove('hidden');
                                hideStreamOutputs();
                            }
                        }
                        break;
                    case 'stream_start':
                        const targetId = eventData.target === 'synthesis' ? 'synthesis' : 'report';
                        const container = document.getElementById(`${targetId}-output-container`);
                        const streamStatus = container?.querySelector('.stream-status');
                        const outputElement = document.getElementById(`${targetId}-output`);

                        if (streamStatus) streamStatus.innerHTML = `<em>Streaming ${eventData.target}...</em>`;
                        if (outputElement) outputElement.textContent = ''; // Clear previous stream content
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
                        addProgress(`Received unknown event: ${eventData.type}`, true); // Log as a progress error
                }
            } catch (e) {
                console.error("Error parsing SSE message or processing event:", e, "Data:", event.data);
                // Avoid adding overly technical errors to the user log unless necessary
                addProgress(`Client error processing stream update. Check console for details.`, true);
            }
        };

        eventSource.onerror = function(error) {
            console.error("SSE connection error:", error);
            const lastProgressItem = progressList?.lastElementChild?.textContent || "";

            // Avoid duplicate connection error messages
            if (!lastProgressItem.includes("Error connecting")) {
                 addProgress("Error connecting to the research stream. Server might be unavailable. Retrying or check server logs.", true, true);
            }

            if (loader) loader.classList.add('hidden');
            hideStreamOutputs(); // Hide streaming areas on connection error

            // If the report wasn't shown yet, display a connection failure message
            if (finalReportDisplay?.classList.contains('hidden')) {
                 addProgress("Could not complete research due to connection failure.", true, true);
                if (reportDisplayDiv && !reportDisplayDiv.innerHTML.includes("<em>Failed to connect")) {
                    reportDisplayDiv.innerHTML = "<article><h2>Connection Error</h2><p><em>Failed to connect to the research service. Please check the server or try again later.</em></p></article>";
                    finalReportDisplay.classList.remove('hidden'); // Show the error display area
                }
            }
            // Note: EventSource might automatically retry, depending on the error.
            // Consider adding logic here to stop retrying after N failures if needed.
        };
    }

    // --- Initial Setup ---
    connectSSE(); // Start the SSE connection when the page loads

    // Clean up SSE connection when the user navigates away
    window.addEventListener('beforeunload', () => {
        if (eventSource) {
            eventSource.close();
            console.log("SSE connection closed on page unload.");
        }
    });
});