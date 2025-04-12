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
    const reportDisplayDiv = document.getElementById('report-display'); // The div to inject final HTML into

    // --- State ---
    let eventSource = null;
    const encodedTopic = document.body.dataset.encodedTopic || "";
    let currentSynthesisContent = ""; // Accumulate stream content
    let currentReportContent = ""; // Accumulate stream content

    // --- Check Libraries ---
    if (typeof DOMPurify === 'undefined') {
        console.error("FATAL: DOMPurify library not loaded. Cannot safely display report.");
        addProgress("FATAL CLIENT ERROR: Security library (DOMPurify) missing. Cannot proceed.", true, true);
        return; // Stop execution
    }
    // Marked.js is now optional if server renders HTML, but keep check if needed later
    if (typeof marked === 'undefined') {
        console.warn("Marked.js library not loaded. Markdown rendering relies on server.");
    }
    console.log("DOMPurify loaded:", typeof DOMPurify);
    console.log("Marked loaded:", typeof marked);

    // --- Helper Functions ---

    function addProgress(message, isError = false, isFatal = false) {
        if (!progressList) return;
        const li = document.createElement('li');
        // Basic text sanitization before adding to DOM
        li.textContent = message.toString();
        if (isFatal) {
            li.classList.add('fatal-error');
        } else if (isError) {
            li.classList.add('error');
        }
        progressList.appendChild(li);
        // Auto-scroll to the bottom
        progressList.scrollTop = progressList.scrollHeight;
    }

    // Make URLs in the final bibliography section clickable
    function linkifyBibliographyUrls() {
        // Check if the target div exists
        if (!reportDisplayDiv) return;

        // Use querySelectorAll on the display div to find list items within the footnotes section
        // Select direct children 'li' of the 'ol' inside '.footnotes' for specificity
        const bibliographyItems = reportDisplayDiv.querySelectorAll('section.footnotes > ol > li');
        if (bibliographyItems.length === 0) {
             console.log("No bibliography items found to linkify.");
             return;
        }

        // More robust URL regex: handles various characters and ensures it's not already linked
        // It looks for http/https URLs not preceded by `href="` or `src="` and not followed by `"` or `</a>` immediately.
        const urlRegex = /(?<!(?:href|src)=["'])(https?:\/\/[^\s<>"']+)(?![^<]*>|["']|<\/a>)/g;


        let linkifiedCount = 0;
        bibliographyItems.forEach((item, index) => {
            try {
                // Find the primary paragraph or the list item itself
                const paragraph = item.querySelector('p');
                const targetElement = paragraph || item;

                // Process only the text nodes directly within the target element to avoid messing up existing links
                // This is more complex but safer than blanket innerHTML replace
                const walker = document.createTreeWalker(targetElement, NodeFilter.SHOW_TEXT, null);
                let node;
                const nodesToReplace = [];

                while(node = walker.nextNode()) {
                    const text = node.nodeValue || '';
                    if (urlRegex.test(text)) {
                         nodesToReplace.push(node);
                    }
                }

                // Replace text nodes containing URLs with new structure (text + links)
                nodesToReplace.forEach(textNode => {
                    const fragment = document.createDocumentFragment();
                    let lastIndex = 0;
                    let match;
                    const text = textNode.nodeValue || '';
                    urlRegex.lastIndex = 0; // Reset regex state

                    while ((match = urlRegex.exec(text)) !== null) {
                        const url = match[0];
                        const index = match.index;

                        // Add text before the match
                        if (index > lastIndex) {
                            fragment.appendChild(document.createTextNode(text.substring(lastIndex, index)));
                        }

                        // Create and add the link
                        const link = document.createElement('a');
                        link.href = url;
                        link.textContent = url; // Display the URL as the link text
                        link.target = "_blank";
                        link.rel = "noopener noreferrer";
                        fragment.appendChild(link);

                        lastIndex = index + url.length;
                        linkifiedCount++;
                    }

                    // Add any remaining text after the last match
                    if (lastIndex < text.length) {
                        fragment.appendChild(document.createTextNode(text.substring(lastIndex)));
                    }

                    // Replace the original text node with the fragment
                    textNode.parentNode?.replaceChild(fragment, textNode);
                });

            } catch (e) {
                console.error(`Error linkifying URL in bibliography item #${index + 1} (ID: ${item.id || 'N/A'}):`, e);
            }
        });

        if (linkifiedCount > 0) {
            console.log(`Bibliography URL linkification applied to ${linkifiedCount} URLs.`);
        } else {
             console.log("No new URLs found to linkify in bibliography.");
        }
    }


    // Add tooltips to citation markers (like [^1]) showing the bibliography entry
    function addCitationTooltips() {
        if (!reportDisplayDiv) return;

        // Target the superscript links pointing to footnotes within the article body
        const citationLinks = reportDisplayDiv.querySelectorAll('article sup a[href^="#fn:"]');
        let appliedCount = 0;
        if (citationLinks.length === 0) {
            console.log("No citation links found to add tooltips.");
            return;
        }

        citationLinks.forEach(link => {
            try {
                // Get the target footnote ID from the link's href
                const href = link.getAttribute('href');
                if (!href) return;
                const footnoteId = decodeURIComponent(href.substring(1)); // Remove '#' and decode
                if (!footnoteId) return;

                // Find the corresponding footnote list item using the ID
                const footnoteElement = document.getElementById(footnoteId);
                if (footnoteElement) {
                    // Clone the footnote content to avoid modifying the original
                    const clone = footnoteElement.cloneNode(true);

                    // Remove the back-reference link ('↩') from the cloned content for the tooltip
                    const backlink = clone.querySelector('a.footnote-backref');
                    if (backlink) { backlink.remove(); }

                    // Get cleaned text content, trim whitespace
                    let footnoteContent = clone.textContent?.replace(/\s+/g, ' ').trim() || '';

                    // Truncate long tooltip content for better display
                    const maxTooltipLength = 250;
                    if (footnoteContent.length > maxTooltipLength) {
                        footnoteContent = footnoteContent.substring(0, maxTooltipLength) + '...';
                    }

                    // Set PicoCSS tooltip attribute and ARIA label
                    const tooltipText = footnoteContent || 'Citation details unavailable.';
                    link.setAttribute('data-tooltip', tooltipText);
                    link.setAttribute('aria-label', `Citation: ${tooltipText}`); // Accessibility
                    appliedCount++;
                } else {
                    // Handle cases where the footnote target li might be missing
                    link.setAttribute('data-tooltip', 'Citation details missing.');
                    link.setAttribute('aria-label', 'Citation details missing.');
                    console.warn(`Footnote target not found for ID: ${footnoteId}`);
                }
            } catch (e) {
                console.error(`Error processing citation link ${link.getAttribute('href') || 'unknown'}:`, e);
                // Add fallback tooltip on error
                link.setAttribute('data-tooltip', 'Error processing citation.');
                link.setAttribute('aria-label', 'Error processing citation.');
            }
        });

        if (appliedCount > 0) {
            console.log(`Citation tooltips applied to ${appliedCount}/${citationLinks.length} links.`);
        }
    }

    // Update the live stream areas (synthesis or report)
    function updateStreamOutput(target, content) {
        const element = target === 'synthesis' ? synthesisOutput : reportOutput;
        const contentVar = target === 'synthesis' ? currentSynthesisContent : currentReportContent;

        if (element) {
            // Append text content directly. Assumes server sends plain text chunks for these streams.
            const newText = contentVar + content;
            element.textContent = newText; // Update entire content at once for simplicity

            // Update the tracking variable
            if (target === 'synthesis') {
                currentSynthesisContent = newText;
            } else {
                currentReportContent = newText;
            }

            // Scroll to the bottom to show latest content
            element.scrollTop = element.scrollHeight;
        }
    }

    // Clear stream outputs and make visible
    function clearStreamOutputs() {
        currentSynthesisContent = "";
        currentReportContent = "";
        if (synthesisOutput) synthesisOutput.textContent = "";
        if (reportOutput) reportOutput.textContent = "";
        // Show the containers
        if (synthesisOutputContainer) synthesisOutputContainer.classList.remove('hidden');
        if (reportOutputContainer) reportOutputContainer.classList.remove('hidden');
        if (reportContainer) reportContainer.classList.remove('hidden'); // Show parent container too
    }

    // Hide the streaming output areas
    function hideStreamOutputs() {
        if (reportContainer) reportContainer.classList.add('hidden');
    }

    // Display the final report HTML received from the server
    function showFinalReport(reportHtmlContent) {
        hideStreamOutputs(); // Hide the live stream areas
        if (finalReportDisplay) finalReportDisplay.classList.remove('hidden'); // Show final report container
        if (reportDisplayDiv) {
            try {
                let finalHtml = "<article><h2>Error</h2><p><em>Received empty or invalid report content from server.</em></p></article>"; // Default error message

                if (reportHtmlContent && typeof reportHtmlContent === 'string' && reportHtmlContent.trim()) {
                    console.log("Received report HTML content from server. Sanitizing...");

                    // Configure DOMPurify - allow necessary elements/attributes for report structure + footnotes
                    const sanitizeConfig = {
                        USE_PROFILES: { html: true }, // Allows standard HTML tags
                        ADD_TAGS: ['sup', 'section', 'article', 'aside', 'nav', 'header', 'footer', 'figure', 'figcaption', 'details', 'summary', 'main', 'section', 'dl', 'dt', 'dd'], // Allow common semantic and structure tags, including footnote section
                        ADD_ATTR: ['data-tooltip', 'aria-label', 'role', 'id', 'href', 'target', 'rel', 'start', 'type', 'colspan', 'rowspan', 'class', 'disabled', 'checked'], // Allow necessary attributes including footnote IDs, hrefs, tooltips, table spans, list types, task list classes/states
                        ALLOW_DATA_ATTR: true, // Needed for data-tooltip
                        ALLOW_ARIA_ATTR: true, // Allow ARIA attributes
                        FORCE_BODY: false, // Don't wrap in body
                        ALLOW_UNKNOWN_PROTOCOLS: false // Disallow protocols like javascript:
                    };

                    // Sanitize the HTML received from the server
                    finalHtml = DOMPurify.sanitize(reportHtmlContent, sanitizeConfig);
                    console.log("Sanitization complete.");

                    // Check if sanitization resulted in empty content
                     if (!finalHtml || !finalHtml.trim()) {
                        console.error("DOMPurify sanitization resulted in empty content.");
                        finalHtml = "<article><h2>Error</h2><p><em>Report content could not be displayed securely after sanitization.</em></p></article>";
                        addProgress("Client Error: Report content blocked by security filter after sanitization.", true);
                    }

                } else {
                     console.warn("Received empty or invalid report HTML content from server.");
                     // Keep the default error message defined above
                }

                console.log("Setting innerHTML for report-display...");
                reportDisplayDiv.innerHTML = finalHtml; // Inject the sanitized HTML
                console.log("Finished setting innerHTML.");

                // Apply post-rendering enhancements *after* HTML is injected
                // Use setTimeout to ensure DOM is fully updated
                setTimeout(() => {
                    console.log("Applying post-render enhancements (tooltips, linkify)...");
                    addCitationTooltips();
                    linkifyBibliographyUrls();
                    console.log("Post-render enhancements applied.");
                }, 100); // Small delay (100ms)

            } catch (e) {
                 console.error("Error processing or displaying final report:", e);
                 addProgress(`Client Error displaying final report: ${e.message}`, true);
                 reportDisplayDiv.innerHTML = "<article><h2>Error</h2><p><em>An error occurred on the client while rendering the report. Please check the console.</em></p></article>";
            }
        } else {
            console.error("Final report display area ('report-display' div) not found.");
            addProgress("Client Error: Cannot find the area to display the final report.", true);
        }
    }

    // Handle the 'complete' event from the server
    function handleCompletion(eventData) {
        addProgress("Research process completed by server. Processing final results...");
        if (eventData && eventData.report_html) {
            showFinalReport(eventData.report_html);
            addProgress("Final report displayed.");
        } else {
             addProgress("Error: Completion event received, but final report content was missing.", true);
             showFinalReport(null); // Show error message
        }
        if (loader) loader.classList.add('hidden');
        if (eventSource) {
            eventSource.close();
            console.log("SSE connection closed by client after completion event.");
        }
    }

    // Handle the 'stream_terminated' event from the server
    function handleStreamTermination() {
        addProgress("Server confirmed stream termination.", false);
        if (loader) loader.classList.add('hidden');
        if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
            eventSource.close();
            console.log("SSE connection closed by client after termination signal.");
        }
        // Check if the final report area is still hidden (meaning process stopped early or failed before 'complete')
        if (finalReportDisplay && finalReportDisplay.classList.contains('hidden')) {
            addProgress("Process stopped or terminated before final report was generated. Check logs for details.", true, false); // Mark as error but not necessarily fatal to the whole page
             // Show an error message in the report area if it's empty and doesn't already show an error
            if (reportDisplayDiv && !reportDisplayDiv.innerHTML.includes("<h2>Error</h2>")) {
                reportDisplayDiv.innerHTML = "<article><h2>Process Interrupted</h2><p><em>Report generation did not complete successfully. The process may have been interrupted or encountered an error before finishing.</em></p></article>";
                finalReportDisplay.classList.remove('hidden'); // Show the error area
                hideStreamOutputs(); // Ensure stream areas are hidden
            }
        } else {
             addProgress("Process finished."); // Add a final confirmation if report was shown
        }
    }

    // Establish the Server-Sent Events connection
    function connectSSE() {
        if (eventSource) {
            eventSource.close(); // Close existing connection if any
        }
        if (!encodedTopic) {
            addProgress("Client Error: Cannot connect to stream. Topic parameter is missing.", true, true);
            if (loader) loader.classList.add('hidden');
            return;
        }

        // Construct the stream URL
        const streamUrl = `/stream?topic=${encodedTopic}`;
        console.log(`Connecting to SSE stream at ${streamUrl}`);
        addProgress("Connecting to research stream...");

        try {
            eventSource = new EventSource(streamUrl);
        } catch (e) {
             console.error("Failed to create EventSource:", e);
             addProgress(`Client Error: Failed to initialize connection to ${streamUrl}. Check browser compatibility or network settings.`, true, true);
             if (loader) loader.classList.add('hidden');
             return;
        }


        eventSource.onopen = function() {
            console.log("SSE connection opened.");
            if (progressList) progressList.innerHTML = ''; // Clear previous progress messages
            addProgress("Connection established. Starting research process...");
            clearStreamOutputs(); // Show and clear streaming areas
            if (finalReportDisplay) finalReportDisplay.classList.add('hidden'); // Hide final report area initially
            if (reportDisplayDiv) reportDisplayDiv.innerHTML = '<article><p><em>Loading final report...</em></p></article>'; // Placeholder
            if (loader) loader.classList.remove('hidden'); // Show loader
        };

        eventSource.onmessage = function(event) {
            try {
                const eventData = JSON.parse(event.data);
                // console.debug("SSE Received:", event.data); // Log raw data if needed

                switch (eventData.type) {
                    case 'progress':
                        addProgress(eventData.message);
                        break;
                    case 'error': // This now covers 'error' and 'fatal' progress messages from server
                        const isFatal = eventData.fatal || false;
                        addProgress(eventData.message, true, isFatal);
                        if (isFatal) {
                            addProgress("Research stopped due to fatal server error.", true, true);
                            if (loader) loader.classList.add('hidden');
                            if (eventSource) eventSource.close();
                            // Display fatal error in the report area if it's still hidden
                            if (reportDisplayDiv && finalReportDisplay?.classList.contains('hidden')) {
                                const safeErrorMessage = DOMPurify.sanitize(eventData.message || "Unknown fatal error", { USE_PROFILES: { html: false }}); // Sanitize error message text
                                reportDisplayDiv.innerHTML = `<article><h2>Fatal Error</h2><p><em>Report generation failed: ${safeErrorMessage}</em></p></article>`;
                                finalReportDisplay.classList.remove('hidden');
                                hideStreamOutputs();
                            }
                        }
                        break;
                    case 'stream_start': // Fired by orchestrator when LLM stream begins
                        const targetId = eventData.target === 'synthesis' ? 'synthesis' : 'report';
                        const container = document.getElementById(`${targetId}-output-container`);
                        const streamStatus = container?.querySelector('.stream-status');
                        const outputElement = document.getElementById(`${targetId}-output`);

                        if (streamStatus) streamStatus.innerHTML = `<em>Streaming ${eventData.target}... (Content appears below)</em>`;
                        if (outputElement) outputElement.textContent = ''; // Clear previous stream content
                        // Update the appropriate content variable
                         if (targetId === 'synthesis') currentSynthesisContent = ""; else currentReportContent = "";
                        addProgress(`Starting LLM ${eventData.target} stream...`);
                        break;
                    case 'llm_chunk': // Chunk of text from LLM stream
                        updateStreamOutput(eventData.target, eventData.content);
                        break;
                    case 'complete': // Research process finished, report generated
                        handleCompletion(eventData);
                        break;
                    case 'stream_terminated': // Server signals stream is fully closed (finally block)
                        handleStreamTermination();
                        break;
                    default:
                        console.warn("Received unknown SSE event type:", eventData.type, eventData);
                        addProgress(`Received unknown event from server: ${eventData.type}`, true); // Log as a progress error
                }
            } catch (e) {
                console.error("Error parsing SSE message or processing event:", e);
                console.error("Received problematic data:", event.data);
                // Avoid adding overly technical errors to the user log unless necessary
                addProgress(`Client error processing stream update. Check console for details.`, true);
            }
        };

        eventSource.onerror = function(error) {
            console.error("SSE connection error:", error);
            const errorMsg = "Connection Error: Lost connection to the research stream. The server might be unavailable or restarting. Please try again later.";
            // Add error only if not already shown
             if (!progressList || !progressList.lastElementChild?.textContent?.includes("Connection Error")) {
                 addProgress(errorMsg, true, true); // Treat connection loss as fatal for the current process
             }

            if (loader) loader.classList.add('hidden');
            hideStreamOutputs(); // Hide streaming areas on connection error

            // If the report wasn't shown yet, display a connection failure message
            if (finalReportDisplay?.classList.contains('hidden')) {
                if (reportDisplayDiv && !reportDisplayDiv.innerHTML.includes("<h2>Connection Error</h2>")) {
                    reportDisplayDiv.innerHTML = "<article><h2>Connection Error</h2><p><em>Failed to maintain connection to the research service. The process could not be completed.</em></p></article>";
                    finalReportDisplay.classList.remove('hidden'); // Show the error display area
                }
            }

            // Close the connection attempt if an error occurs
            if (eventSource) {
                eventSource.close();
                console.log("SSE connection closed due to error.");
            }
        };
    }

    // --- Initial Setup ---
    connectSSE(); // Start the SSE connection when the page loads

    // Clean up SSE connection when the user navigates away or closes the tab
    window.addEventListener('beforeunload', () => {
        if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
            eventSource.close();
            console.log("SSE connection closed on page unload.");
        }
    });
});