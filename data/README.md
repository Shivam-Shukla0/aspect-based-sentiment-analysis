# Data

Raw and processed data are **not committed** (see `.gitignore`). The SemEval-2014 licence terms should be checked before any redistribution.

## Source

SemEval-2014 Task 4: Aspect Based Sentiment Analysis, laptop and restaurant reviews. Task page: <https://alt.qcri.org/semeval2014/task4/> (see its "Data and Tools" page).

Citation: Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., Manandhar, S. (2014). _SemEval-2014 Task 4: Aspect Based Sentiment Analysis._ Proceedings of SemEval 2014.

There are two tiers of download, both from the task organisers:

- **Trial data** (`laptops-trial.xml`, `restaurants-trial.xml`, 100 sentences each): a direct, ungated download from
  `https://alt.qcri.org/semeval2014/task4/data/uploads/{laptops,restaurants}-trial.xml`. Useful for development and format
  verification, but too small to be a benchmark.
- **Full train and test data** (laptop train, restaurant train, test phase A/B — the official benchmark): distributed via the
  ILSP Metashare repository linked from the task's "Data and Tools" page. This requires a free Metashare account and
  accepting its usage licence before download; read that licence yourself when you register, since it was not reachable from
  this environment to summarise here. No account credentials are shared or stored in this repository.

Whoever downloads the full files should keep the original train/test file names, place them in `data/raw/` (git-ignored), and
add their SHA-256 (`sha256sum <file>`) to the table below so `python -m scripts.check_dataset` can verify them.

## Checksums

| File                       | SHA-256                                                            | Source           |
| -------------------------- | ------------------------------------------------------------------ | ---------------- |
| `laptops-trial.xml`        | `0971a30e6add454d344ac011385fb8407db4193db048c8cd8e7a82908ac5172d` | trial (verified) |
| `restaurants-trial.xml`    | `48d758803cd5183e9b8ee8a04520889a4b7ff371a2ff66e399ba193620e5bc41` | trial (verified) |
| laptop train (full)        | _pending download_                                                 | Metashare        |
| restaurant train (full)    | _pending download_                                                 | Metashare        |
| test phase B / gold (full) | _pending download_                                                 | Metashare        |

Run `python -m scripts.check_dataset` after adding files to `data/raw/`; it checks every row above that has a recorded checksum and reports any file that is missing or does not match.

## Splits

- The original train/test separation is preserved. Test data is used only for final evaluation.
- Any validation split is carved out of the training data with `absa.config.RANDOM_SEED` and must be documented in the experiment notebook.

## Processed record schema

One JSON object per (review, aspect) pair:

```json
{
  "review": "The battery is weak but the screen is excellent.",
  "aspect": "battery",
  "sentiment": "negative"
}
```

`sentiment` is one of `positive`, `negative`, `neutral`, `conflict`. `conflict` is kept by default. An experiment that drops it must state so and pass a reduced `allowed_labels` to `absa.data.validate_record`.

## Inference output schema (interface contract, not a model output)

```json
{ "review": "...", "predictions": [{ "aspect": "battery", "sentiment": "negative" }] }
```

The LSTM and BERT classifiers take a supplied aspect. End-to-end evaluation also needs aspect extraction (`src/absa/extraction/`).
