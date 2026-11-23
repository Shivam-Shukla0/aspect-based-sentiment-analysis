"use strict";

// Talks to the Flask backend: POST /predict with {"review": ..., "model": ..., "aspects": [...]}
// answers {"review": ..., "predictions": [{"aspect": ..., "sentiment": ...}]}.
// Failures answer {"error": "message"} with a 4xx or 5xx status.
// Must match the labels the backend can return; anything else is shown with a neutral style.
const SENTIMENTS = ["positive", "negative", "neutral", "conflict"];
// Same limit as the server (app/server.py), checked here first to give instant feedback.
const MAX_REVIEW_LENGTH = 2000;

// Small helper that builds a DOM node; used everywhere instead of innerHTML for safety.
function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) {
        node.className = className;
    }
    if (text !== undefined) {
        // textContent never parses markup, so model output cannot inject HTML.
        node.textContent = text;
    }
    return node;
}

// Each validate* function returns an error message, or "" when the input is fine.
function validateReview(review) {
    if (!review.trim()) {
        return "Enter a review first.";
    }
    if (review.length > MAX_REVIEW_LENGTH) {
        return "The review is longer than " + MAX_REVIEW_LENGTH + " characters.";
    }
    return "";
}

// "battery life, screen" -> ["battery life", "screen"]: split on commas, trim, drop empties.
function parseAspects(text) {
    return text
        .split(",")
        .map((term) => term.trim())
        .filter((term) => term.length > 0);
}

function validateAspects(review, aspects) {
    if (aspects.length === 0) {
        return "Enter at least one aspect term.";
    }
    // Case-insensitive check, mirroring the server: an aspect must occur in the review text.
    const missing = aspects.filter((term) => !review.toLowerCase().includes(term.toLowerCase()));
    if (missing.length > 0) {
        return "Not found in the review: " + missing.join(", ") + ".";
    }
    return "";
}

function renderPredictions(model, predictions) {
    const card = element("article", "model-card");
    card.appendChild(element("h2", "", model));
    if (predictions.length === 0) {
        card.appendChild(element("p", "", "No aspects found."));
        return card;
    }
    const list = element("ul", "aspect-list");
    for (const item of predictions) {
        // The value becomes part of a CSS class name, so only known labels are allowed through.
        const sentiment = SENTIMENTS.includes(item.sentiment) ? item.sentiment : "neutral";
        const row = element("li");
        row.appendChild(element("span", "aspect", String(item.aspect)));
        row.appendChild(
            element("span", "sentiment sentiment-" + sentiment, String(item.sentiment)),
        );
        list.appendChild(row);
    }
    card.appendChild(list);
    return card;
}

function renderError(model, message) {
    const card = element("article", "model-card");
    card.appendChild(element("h2", "", model));
    card.appendChild(element("p", "model-error", message));
    return card;
}

// Sends one request for one model and returns its predictions, or throws an Error whose
// message is safe to show to the user (network problem, server error text, bad response).
async function predict(review, model, aspects) {
    let response;
    try {
        response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ review: review, model: model, aspects: aspects }),
        });
    } catch (error) {
        throw new Error("Could not reach the server.");
    }
    // The body may not be JSON (e.g. a proxy error page), so parsing is guarded.
    let payload = null;
    try {
        payload = await response.json();
    } catch (error) {
        payload = null;
    }
    if (!response.ok) {
        const detail = payload && payload.error ? payload.error : "HTTP " + response.status;
        throw new Error(detail);
    }
    if (!payload || !Array.isArray(payload.predictions)) {
        throw new Error("The server sent an unexpected response.");
    }
    return payload.predictions;
}

async function analyse(review, models, aspects, results) {
    results.replaceChildren();
    // allSettled (not all): one failing model must not hide the results of the others.
    // The requests run in parallel, one per selected model.
    const settled = await Promise.allSettled(
        models.map((model) => predict(review, model, aspects)),
    );
    settled.forEach((outcome, index) => {
        const model = models[index];
        results.appendChild(
            outcome.status === "fulfilled"
                ? renderPredictions(model, outcome.value)
                : renderError(model, outcome.reason.message),
        );
    });
}

function setStatus(status, message, isError) {
    status.textContent = message;
    status.className = isError ? "error" : "";
}

// Wires the form to the backend; called once when the page loads.
function init() {
    const form = document.getElementById("review-form");
    const status = document.getElementById("status");
    const results = document.getElementById("results");
    const submit = document.getElementById("submit");

    form.addEventListener("submit", async (event) => {
        event.preventDefault(); // handle the submit here instead of a full page reload
        const review = form.elements.review.value;
        const models = Array.from(form.querySelectorAll("input[name=model]:checked")).map(
            (box) => box.value,
        );
        const aspects = parseAspects(form.elements.aspects.value);
        const problem =
            validateReview(review) ||
            validateAspects(review, aspects) ||
            (models.length === 0 ? "Select a model." : "");
        if (problem) {
            setStatus(status, problem, true);
            return;
        }
        setStatus(status, "Analysing...", false);
        submit.disabled = true; // prevent double submits while a request is running
        try {
            await analyse(review, models, aspects, results);
            setStatus(status, "", false);
        } finally {
            // finally: re-enable the button even if something above threw.
            submit.disabled = false;
        }
    });
}

// Browser-only start-up; skipped when the file is loaded by the Node-based tests.
if (typeof document !== "undefined") {
    init();
}

// Export the functions for tests; in a browser `module` does not exist, so this is skipped.
if (typeof module !== "undefined") {
    module.exports = {
        validateReview,
        parseAspects,
        validateAspects,
        renderPredictions,
        renderError,
        predict,
        analyse,
    };
}
