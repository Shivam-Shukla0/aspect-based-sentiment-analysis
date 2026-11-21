# Frontend

A small static page: `static/index.html`, `static/style.css` and `static/app.js`. There is no build step and no framework.

## API it expects

The page talks to the Flask backend (issue #43). The contract it relies on:

- `POST /predict` with JSON `{"review": "...", "model": "lstm"}` answers `{"review": "...", "predictions": [{"aspect": "...", "sentiment": "..."}]}`.
- Errors answer a 4xx or 5xx status and `{"error": "message"}`.
- The page sends one request per selected model, so several selected models show side by side.

The backend must serve `app/static/` at the site root. Review text and model output are inserted with `textContent` only, never as markup.

## Manual test checklist

1. Open the page, submit an empty review: the status line asks for a review and nothing is sent.
2. Submit a review with no model ticked: the status line asks to select a model.
3. Submit a review with one model: one card lists each aspect with a coloured sentiment.
4. Tick two or three models: one card per model appears side by side.
5. Stop the backend and submit: each card shows "Could not reach the server."
6. Make one model fail (for example a missing checkpoint): its card shows the error and the other cards still show results.
7. Narrow the window below 30 rem: the layout stays in one column and the button spans the width.
8. Submit `<img src=x onerror=alert(1)>` as a review: no alert, the text is shown literally.
