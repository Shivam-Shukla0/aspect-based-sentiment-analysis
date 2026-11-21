# Demo and reproducibility guide

How to reproduce the results, run the demo and answer the questions a reviewer is likely to ask. Every figure quoted in the slides comes from the technical report ([reports/report.tex](../reports/report.tex)) and from the result files the commands below write.

## 1. Set up

On Google Colab (recommended, GPU): open a notebook from the README, choose _Runtime → Change runtime type → GPU_, and run all cells. On your own machine:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch                       # the build that matches your hardware
pip install -r requirements.txt
pip install -e . --no-deps
python -m scripts.verify_environment    # add --require-gpu to insist on CUDA
```

## 2. Data

The official files are distributed through the ILSP Metashare repository (account and licence needed; see [data/README.md](../data/README.md)). Place the four XML files in `data/raw/`, then:

```bash
python -m scripts.check_dataset
python -m scripts.build_processed data/raw/Laptop_Train_v2.xml       --domain laptop     --split train
python -m scripts.build_processed data/raw/Laptops_Test_Gold.xml     --domain laptop     --split test
python -m scripts.build_processed data/raw/Restaurants_Train_v2.xml  --domain restaurant --split train
python -m scripts.build_processed data/raw/Restaurants_Test_Gold.xml --domain restaurant --split test
```

The notebooks download the same four files from a public mirror when the official ones are not in Drive; the report says so, and results should be re-run on the official files before being quoted.

## 3. Train and evaluate (GPU)

Run these for `--domain laptop` and `--domain restaurant`. Each writes `summary.json` and `test_report.json` to `outputs/`.

```bash
python -m scripts.train_lstm      --domain laptop --hidden 128 --lr 0.001 0.003 --dropout 0.0 0.3 0.5 --weighting off --epochs 40
python -m scripts.train_bert      --domain laptop
python -m scripts.train_extractor --domain laptop --classifier-dir outputs/bert_laptop/run_2/model
python -m scripts.benchmark_llm   --domain laptop --model-name Qwen/Qwen2.5-7B-Instruct
python -m scripts.compare_results                    # tables and charts from the saved result files
```

Use the best run's folder for `--classifier-dir` (its `summary.json` lists the runs). The notebooks `01` to `06` run exactly these steps and keep their outputs.

## 4. Run the demo

```bash
python -m app.server                    # http://127.0.0.1:8000
```

The server needs the trained checkpoints in `outputs/` (copy them from Colab or Drive, or set `ABSA_LSTM_CHECKPOINT` and `ABSA_BERT_MODEL_DIR`). Open the page, paste a review, type the aspect terms as they appear in it, tick a model and press _Analyse_. A model without a checkpoint answers with a clear message and the others still work. See [app/README.md](../app/README.md) for the API.

A five-minute demo script:

1. Show the page with `The battery is weak but the screen is excellent.` and the terms `battery, screen`: BERT gives negative and positive.
2. Tick LSTM as well: the two cards appear side by side; point out that the LSTM is the weaker baseline.
3. Type a term that is not in the review: the page refuses before sending anything.
4. Stop one model's checkpoint (or tick `LLM`): the card shows the error and the other cards still work.
5. Open `outputs/bert_laptop/test_report.json` and show the confusion matrix and an error example.

## 5. Questions to be ready for

- **Why is the LSTM so much weaker than BERT?** It starts from random word vectors and learns from 2 to 4 thousand examples; BERT brings pre-trained language knowledge. It is reported as a baseline, not tuned to compete.
- **Why macro F1 and not accuracy?** The classes are imbalanced (about 53% of laptop test aspects are positive); a model that always answers `positive` scores that accuracy. Macro F1 gives each class equal weight.
- **Why drop `conflict`?** It is very rare and the usual SemEval-2014 protocol drops it. The models cannot be fairly scored on a class with a handful of examples.
- **How do you know the test set was not used for tuning?** Every choice uses validation macro F1 on a split carved from the training file; the test file is read once, after the setting is fixed (see the evaluation protocol in the README).
- **Why can the demo not find aspects itself?** The classifiers take a supplied aspect. A BERT extractor exists and is evaluated, but it is not wired into the demo; the app takes the terms from the user.
- **How was the language model run without an API key?** An open-weights instruction model ran on the Colab GPU with greedy decoding; every raw answer is saved, so the scores can be recomputed without the model.
- **What are the limits?** One seed per setting, a mirror copy of the dataset, three-class sentiment only, no overlapping aspects in the extractor.
