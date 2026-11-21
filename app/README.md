# Demo application

A Flask backend (`server.py`) and a small static page (`static/index.html`, `style.css`, `app.js`). There is no build step and no framework.

## Run

From the repository root, with the dependencies installed and a trained model available:

```bash
pip install -r requirements.txt
python -m app.server            # http://127.0.0.1:8000, debug mode off
```

| Variable                 | Meaning                                                  | Default                                              |
| ------------------------ | -------------------------------------------------------- | ---------------------------------------------------- |
| `ABSA_HOST`, `ABSA_PORT` | bind address                                             | `127.0.0.1`, `8000`                                  |
| `ABSA_DOMAIN`            | which trained domain to serve (`laptop` or `restaurant`) | `laptop`                                             |
| `ABSA_DEVICE`            | `cpu`, `cuda` or `auto`                                  | `cpu`                                                |
| `ABSA_LSTM_CHECKPOINT`   | path to `lstm_checkpoint.pt`                             | `checkpoint` in `outputs/lstm_<domain>/summary.json` |
| `ABSA_BERT_MODEL_DIR`    | path to the fine-tuned BERT directory                    | `checkpoint` in `outputs/bert_<domain>/summary.json` |

Checkpoints are not in Git. Train them with `scripts/train_lstm.py` and `scripts/train_bert.py` (the Colab notebooks do this) and copy `outputs/` next to the repository, or set the two path variables. Models load on first use and stay in memory. A missing checkpoint answers 503 with a message and the other models keep working.

## API

The models classify aspect terms that the caller supplies: no aspect extractor has been trained, so the request carries the terms, and each term must occur in the review (case-insensitive).

`POST /predict` with JSON

```json
{
  "review": "The battery is weak but the screen is excellent.",
  "model": "lstm",
  "aspects": ["battery", "screen"]
}
```

answers `200` and the shared prediction record

```json
{
  "review": "The battery is weak but the screen is excellent.",
  "predictions": [
    { "aspect": "battery", "sentiment": "negative" },
    { "aspect": "screen", "sentiment": "positive" }
  ]
}
```

Errors answer `{"error": "message"}`:

| Status | Cause                                                                                                |
| ------ | ---------------------------------------------------------------------------------------------------- |
| 400    | body is not a JSON object; `review`, `model` or `aspects` missing or malformed; limits exceeded      |
| 404    | unknown model name, or unknown path                                                                  |
| 405    | wrong method                                                                                         |
| 413    | body larger than 64 KiB                                                                              |
| 422    | an aspect term does not occur in the review, or the model output failed validation                   |
| 500    | the model raised an error while predicting                                                           |
| 503    | the model is not available: checkpoint missing, loader failed, or the `llm` backend (not configured) |

Limits: review up to 2000 characters, up to 20 aspects of up to 100 characters. `GET /models` lists the model names and which are loaded, `GET /health` answers `{"status": "ok"}`, and `/` serves the page.

The `llm` option stays in the page but answers 503: prompt building and validation exist, but no model endpoint or key is configured (no secrets are stored in the app). The page sends one request per selected model, so several models show side by side. Review text and model output are inserted with `textContent` only, never as markup.

## Limitations

- Given-aspect classification only; there is no aspect extractor, so the user types the terms.
- One domain is served per process (`ABSA_DOMAIN`); the models were trained on laptop or restaurant reviews and are not meant for other text.
- Single process, models on CPU by default; this is a demo server, not a production deployment (use a WSGI server and set `ABSA_DEVICE` for more).
- Results come from single-seed training runs; see the technical report.

## Manual test checklist

1. Open the page, submit an empty review: the status line asks for a review and nothing is sent.
2. Submit a review without aspect terms, or with a term that is not in the review: the status line says so and nothing is sent.
3. Submit a review with no model ticked: the status line asks to select a model.
4. Submit a review with one model: one card lists each aspect with a coloured sentiment.
5. Tick two or three models: one card per model appears side by side.
6. Stop the backend and submit: each card shows "Could not reach the server."
7. Make one model fail (for example a missing checkpoint): its card shows the error and the other cards still show results.
8. Narrow the window below 30 rem: the layout stays in one column and the button spans the width.
9. Submit `<img src=x onerror=alert(1)>` as a review: no alert, the text is shown literally.
