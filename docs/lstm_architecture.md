# LSTM-with-attention architecture

Given-aspect classification only (the model is told the aspect; it does not extract one). Pipeline, left to right:

```
review tokens  -> embedding -> LstmEncoder -> review_states (batch, time, H), review_mask
aspect tokens  -> embedding -> LstmEncoder -> aspect_states (batch, atime, H), aspect_mask
                                                      |
                                      AspectAttention (review_states, review_mask,
                                                        aspect_states, aspect_mask)
                                                      |
                                        (logits, attention_weights)
```

`embedding` (`absa.models.build_embedding`) and `LstmEncoder` (`absa.models.LstmEncoder`) are shared: the same weights encode both the review and the aspect, so both end up in the same `H`-dimensional space (`H = hidden_size * 2` if bidirectional). That is this project's reading of the issue's "encode the aspect, for example the mean of its embeddings or its own LSTM" choice: encoding the aspect with the review's own encoder is simpler than adding a second LSTM, keeps both representations in a space the attention mechanism can score directly without an extra projection, and is a standard choice in aspect-attention LSTM designs.

## Attention (`absa.models.AspectAttention`)

1. **Aspect representation**: the masked mean (`absa.models.masked_mean`) of the aspect's encoded states over its real tokens — `absa.models.masked_mean` divides by the real-token count, not the padded length, so it is correct regardless of padding.
2. **Scoring**: Luong-style "general" attention. The aspect representation is passed through a learned linear map (`project_aspect`, no bias), then dotted with every review token's state: `score_t = review_states_t . (W . aspect_repr)`. This is cheaper than a full bilinear form (`nn.Bilinear`) for the same expressiveness, since the aspect-side projection only needs to be computed once per example rather than once per (review-position, aspect) pair.
3. **Masking**: scores at padding positions are set to `-inf` before the softmax, so `exp(-inf) == 0.0` exactly and padding receives exactly zero weight; the weights over real tokens sum to exactly 1.
4. **Context and classification**: the context vector is the attention-weighted sum of `review_states`, and a linear layer (`classifier`) maps it to one logit per label in `absa.config.SENTIMENT_LABELS`.

`AspectAttention.forward` returns `(logits, attention_weights)`; the weights are exposed (not discarded) so a caller can inspect which review tokens drove a prediction, as the issue's scope asks for.

## Training, best checkpoint and class imbalance (`absa.models.lstm_training`)

- **Best epoch.** `train` keeps a copy of the weights whenever validation macro F1 improves (strictly, so the earliest epoch wins a tie) and loads them back before returning. The checkpoint written by `run_training` is therefore the best epoch, and `best_epoch` in the run log names it.
- **Class imbalance.** `LstmConfig.class_weighting` is off by default, which means plain cross-entropy. When it is on, each class in the *training* loss is weighted by `total / (classes_present * class_count)`, so rare classes count more; a class with no training example gets weight 0. The validation loss is never weighted, so it stays comparable between runs. The setting is stored in the run log with the rest of the config.
- **Noisy selection.** Choosing the best epoch from validation macro F1 is only as reliable as the validation split. On the 100-sentence trial files that split has fewer than a dozen records, so the choice of epoch (and any comparison between settings) is noise there and should not be read as a result.

## What is not yet wired up

Evaluation on the official test split (issue #25) and the numbers in `reports/lstm_results.md` need to be redone with the best-epoch checkpoint on the full files; they were produced from the last epoch.
