# Agent review questions

When conducting a design or implementation review, ask the applicable question below exactly as written. Include its source and code snippet, then provide the corresponding answer. Keep the supplied answer's technical caveats intact.

## Category: design

**Source:** `src/absa/data/splits.py`, lines 32–36 — `split_train_val`

**Question:** What motivates sorting normalized review groups before a seeded shuffle, and how does choosing the validation size by group count affect the actual fraction of aspect records held out?

**Snippet:**

```python
groups = sorted({normalise_review(str(record["review"])) for record in records})
rng = random.Random(seed)  # noqa: S311
rng.shuffle(groups)
cut = max(1, round(len(groups) * val_fraction))
val_groups = set(groups[:cut])
```

**Answer:** The set comprehension creates one group key per distinct normalized review. Sorting gives `rng.shuffle(groups)` a stable starting order, so the same groups and seed produce the same validation selection. A fixed random seed alone cannot ensure this if the initial set iteration order changes between processes.

Grouping by review text keeps aspect records from the same review together, preventing shared review text from leaking across the training and validation splits. However, `len(groups)` counts review groups rather than aspect records, so `cut` selects approximately `val_fraction` of the groups.

If selected groups contain many aspects, the validation fraction of aspect records can exceed the requested fraction; if they contain few aspects, it can be smaller. This prioritizes reproducibility and leakage prevention over an exact record-level fraction, and the random selection does not enforce sentiment-class balance.

## Category: why

**Source:** `src/absa/models/lstm_attention.py`, lines 113–115 — `AspectAttention.forward`

**Question:** Why represent an aspect by the masked mean of its encoded states before projected dot-product attention, and what information might this pooling lose for multiword aspects?

**Snippet:**

```python
aspect_repr = masked_mean(aspect_states, aspect_mask)
projected_aspect = self.project_aspect(aspect_repr)
scores = torch.einsum("bth,bh->bt", review_states, projected_aspect)
```

**Answer:** `masked_mean(aspect_states, aspect_mask)` combines the non-padding encoded aspect tokens into one fixed-size `aspect_repr`. It incorporates information from the whole aspect phrase while preventing padding from diluting the average. `self.project_aspect` learns a transformation that makes this representation useful for comparison with review-token states.

The expression `"bth,bh->bt"` computes a dot product over the hidden dimension `h`, producing one score for every review position `t` in each batch item `b`. These scores depend on the target aspect, allowing different aspects of the same review to assign different relevance to its tokens.

The tradeoff is a compact, efficient representation that gives every non-padding aspect state equal weight during pooling. For multiword aspects, averaging can blur distinctions between important and less informative tokens and remove their separate positional identities. Because the encoded states may already contain contextual and order information, this does not mean that all word-order information is lost.

## Category: tradeoff

**Source:** `src/absa/models/bert_classifier.py`, lines 95–102 — nested function `collate`

**Question:** What tradeoff does truncating only the review make in this review–aspect sentence pair, especially when sentiment evidence about the aspect appears near the end of a long review?

**Snippet:**

```python
encoded = tokenizer(  # type: ignore[operator]
    [str(record["review"]) for record in batch],
    [str(record["aspect"]) for record in batch],
    padding=True,
    truncation="only_first",
    max_length=config.max_length,
    return_tensors="pt",
)
```

**Answer:** The tokenizer receives reviews as the first sequence and aspects as the second sequence. `truncation="only_first"` shortens only the review to satisfy `config.max_length`, preserving the aspect that identifies the sentiment target. The length budget includes both sequences and the required special tokens.

This bounds input size and associated computation and memory, but sacrifices review context. With right-side truncation, sentiment evidence near the end of a long review may be removed, leaving the target aspect present without the evidence needed to classify it correctly.

The snippet does not explicitly set the truncation side, so the loss of late evidence depends on that tokenizer setting. The strategy also requires the aspect and special tokens to fit within the length budget; an unusually long aspect can make only-first truncation fail.

## Category: tradeoff

**Source:** `src/absa/extraction/bio.py`, line 81 — `align_bio_tags`

**Question:** What motivates treating any character overlap as sufficient to include a whole token in an aspect, and how does this choice affect boundary accuracy when an annotation cuts through a token?

**Snippet:**

```python
covered = [i for i, t in enumerate(tokens) if t.end > span_start and t.start < span_end]
```

**Answer:** `covered` stores the indices of tokens whose character intervals overlap the aspect interval. The condition `t.end > span_start and t.start < span_end` detects a nonempty intersection between half-open intervals. A token that merely touches the aspect boundary is excluded.

Including partially overlapping tokens allows token-level BIO tagging to retain an aspect when its annotation starts or ends inside a token. Requiring full containment could omit such a token and lose part or all of the annotated aspect.

The tradeoff is loss of boundaries inside tokens. For example, a token covering `[0, 3)` is included for an aspect covering `[0, 2)`, but reconstructing the aspect from the whole token would extend its end to 3. The relevant token is identified, yet the widened prediction can fail exact character-span matching.

## Category: why

**Source:** `src/absa/models/llm.py`, lines 143–148 — `_find_json_array`

**Question:** Why accept the first decoded array of objects, including an empty array, and what ambiguity does this immediate return create when an LLM response contains multiple JSON arrays?

**Snippet:**

```python
try:
    value, _ = decoder.raw_decode(text, start)
except json.JSONDecodeError:
    continue
if isinstance(value, list) and all(isinstance(item, dict) for item in value):
    return value
```

**Answer:** `decoder.raw_decode(text, start)` attempts to parse a JSON value at a candidate position without requiring the remaining response to be JSON. Ignoring the returned end position permits trailing prose or a closing code fence, while `continue` skips candidates that cannot be decoded. This makes extraction tolerant of surrounding model-generated text.

The condition accepts a list whose elements are dictionaries. An empty list also passes because `all(...)` returns `True` when there are no elements, allowing an empty result to represent no detected aspects.

`return value` stops processing at the first qualifying array. If an earlier array is an example, an unrelated list of objects, or an empty list followed by actual predictions, later arrays are ignored. This provides a simple selection rule but cannot determine which array the model intended as its final answer. The condition checks container types only; it does not itself validate the required aspect and sentiment fields.
