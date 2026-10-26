# Task 1: Training Skip-gram Word Embeddings

## Setup

Use the supplied data and root environment. The framework creates the training
vocabulary, converts words to IDs, samples training pairs and negative words, and
performs SGD. The two embedding matrices and their initialization are provided.

## TODO

| TODO | What to implement                                      |
| ---- | ------------------------------------------------------ |
| 1    | Center/context pairs within a radius-2 window          |
| 2    | Lookup center/context vectors and compute dot products |
| 3    | Negative-sampling loss                                 |
| 4    | Cosine similarity scores                               |

For TODO[1], process ONE document at a time and return a list of `(center, context)`
ID pairs. A word is not its own context position. Ignore pairs containing PAD=0
or UNK=1, while keeping original positions when locating neighbors.

For TODO[2], input IDs have shapes `[B]`, `[B]`, `[B,K]`. Centers use the input table;
contexts/negatives use the output table. Return positive scores `[B]` and negative
scores `[B,K]`, with each score the corresponding vectors' dot product.

For TODO[3], use the lecture's per-pair objective:

```text
-log(sigmoid(positive_score)) - sum_k log(sigmoid(-negative_score[k]))
```

Sum over negative samples, then average over the batch. `F.logsigmoid` computes
the log-sigmoid safely; no derivative implementation is required.

For TODO[4], cosine is `dot(u,v)/(norm(u)*norm(v))`. Vector norms, excluding the
query/special tokens, and selecting the top five words are supplied.

## Run and Expected Result

```shell
python hw1_embedding/main.py
```

Defaults: 64-dimensional vectors, radius=2, K=5, 10 epochs, batch=512, SGD lr=10.0.
The supplied sampler uses up to 200,000 positive pairs and uniformly sampled
negative words. Sampling is framework code, not a student implementation task.

Expected result: finite training losses, a saved word-vector table and its curve,
and printed example neighbors. Similar words are not required to have an exact
prescribed ranking on this small corpus.
