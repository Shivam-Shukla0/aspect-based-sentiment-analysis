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

## What is not yet wired up

Training, checkpointing and evaluation (issues #24 and #25) are not part of this module; this is the forward pass only, used directly in tests with random weights and verified with real tokenized data.
