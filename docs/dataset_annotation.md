# SemEval-2014 Task 4 annotation format

Findings below come from running the trial files in `data/raw/` through a one-off inspection script, not from the task papers. The trial files (100 sentences per domain) are the only SemEval-2014 files reachable from this environment; see [`data/README.md`](../data/README.md) for why, and for who should download the full train/test files. Counts here describe the trial sample, not the full benchmark; whoever adds the full files should rerun the same checks and update the numbers in issue #19 (splitting and EDA).

## Elements

- `<sentence id="...">`: one review sentence. The `id` is an opaque integer, unique within a file; it is not a running index (ids in the trial files are not sequential).
- `<text>`: the sentence text, exactly as the offsets below index into.
- `<aspectTerms>`: present only on sentences that have at least one explicitly mentioned aspect term. Absent (not empty) on sentences with no aspect terms — both files have zero `<aspectTerms>` elements with zero children, only a present-or-absent element.
  - `<aspectTerm term="..." polarity="..." from="..." to="..."/>`: one mention. `term` is the literal substring; `polarity` is one of `positive`, `negative`, `neutral`, `conflict`; `from`/`to` are character offsets into `<text>`.
- `<aspectCategories>` (restaurant file only): coarse-grained categories the sentence discusses, independent of whether a term was extracted. Every restaurant sentence has this element; the laptop file has none.
  - `<aspectCategory category="..." polarity="..."/>`: no offsets, since a category is not a span in the text. Restaurant categories seen in the trial file: `food`, `service`, `price`, `ambience`, `anecdotes/miscellaneous`.

## Offset convention

`text[from:to] == term`, i.e. `from`/`to` are zero-based Python-style character offsets (end-exclusive), not inclusive or 1-based. Verified programmatically on every aspect term in both trial files (145 terms total: 49 laptop, 96 restaurant) — zero mismatches. A parser can therefore slice directly; no adjustment is needed. Two examples:

```
id=2128: "I liked the aluminum body."
  aspectTerm term="aluminum body" polarity=positive from=12 to=25
  text[12:25] == "aluminum body"

id=353: "From the build quality to the performance, everything about it has been sub-par from what I would have expected from Apple."
  aspectTerm term="build quality" polarity=negative from=9 to=22
  aspectTerm term="performance" polarity=negative from=30 to=41
```

## Sentences without aspect terms

| File                    | Sentences | No `<aspectTerms>` element | `>1` aspect term |
| ----------------------- | --------- | -------------------------- | ---------------- |
| `laptops-trial.xml`     | 100       | 60                         | 8                |
| `restaurants-trial.xml` | 100       | 46                         | 23               |

Nearly half to most sentences carry no aspect-term annotation at all in this sample (a sentence can still be a valid, on-topic review sentence without an explicit aspect mention). #17 (normalization) must write these to a separate file instead of dropping them, per its acceptance criteria.

## `conflict` polarity

No aspect **term** in either trial file is labelled `conflict`; term-level polarities seen are only `positive`, `negative`, `neutral`. `conflict` does appear once, at the aspect-**category** level, in sentence id `2882`:

```
"The sweet lassi was excellent as was the lamb chettinad and the garlic naan but the rasamalai was forgettable."
  aspectTerm term="sweet lassi"     polarity=positive
  aspectTerm term="lamb chettinad"  polarity=positive
  aspectTerm term="garlic naan"     polarity=positive
  aspectTerm term="rasamalai"       polarity=negative
  aspectCategory category="food"   polarity=conflict
```

This shows what `conflict` means in practice: the sentence mixes positive and negative opinions about the same coarse category, so the aggregate label is neither simply positive nor negative. Because this trial sample happens not to exercise `conflict` at the aspect-term level, parser and schema code must not assume it is rare or absent — `absa.config.SENTIMENT_LABELS` and `absa.data.validate_record` already accept it by default, and this must stay true once the full dataset (which does contain term-level `conflict` examples) is parsed.

## Term-level polarity counts (trial sample)

| File                    | positive | negative | neutral |
| ----------------------- | -------- | -------- | ------- |
| `laptops-trial.xml`     | 29       | 16       | 4       |
| `restaurants-trial.xml` | 68       | 18       | 10      |

## Differences between the laptop and restaurant files

- Only the restaurant file has `<aspectCategories>`; the laptop file has none. A shared parser must not assume both elements exist, and must keep term-level and category-level annotations distinct, per #16's acceptance criteria.
- The restaurant file has more multi-aspect sentences (23 vs. 8) and a higher ratio of sentences that do carry at least one aspect term (54 vs. 40).
- Both files use the same `<sentence>`/`<aspectTerms>`/`<aspectTerm>` structure and the same offset convention, so the term-level parsing logic can be shared between domains.
