"use strict";

// Talks to the Flask backend: POST /predict with {"review": ..., "model": ..., "aspects": [...]}
// answers {"review": ..., "predictions": [{"aspect": ..., "sentiment": ...}]}.
// Failures answer {"error": "message"} with a 4xx or 5xx status.
const SENTIMENTS = ["positive", "negative", "neutral", "conflict"];
const MAX_REVIEW_LENGTH = 2000;

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

function validateReview(review) {
    if (!review.trim()) {
        return "Enter a review first.";
    }
    if (review.length > MAX_REVIEW_LENGTH) {
        return "The review is longer than " + MAX_REVIEW_LENGTH + " characters.";
    }
    return "";
}

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

function init() {
    const form = document.getElementById("review-form");
    const status = document.getElementById("status");
    const results = document.getElementById("results");
    const submit = document.getElementById("submit");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
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
        submit.disabled = true;
        try {
            await analyse(review, models, aspects, results);
            setStatus(status, "", false);
        } finally {
            submit.disabled = false;
        }
    });
}

if (typeof document !== "undefined") {
    init();
}

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
