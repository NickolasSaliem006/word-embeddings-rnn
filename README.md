# ECE449 Homework 1: Word Embeddings and RNNs

## Complete one end-to-end workflow from the two lectures:

```text
News text -> word/context pairs -> Skip-gram word vectors
          -> RNN hidden states -> mean sentence vector -> news category
```

### Task 1: Skip-gram Word Embeddings (40%)

Complete **four TODOs** in [hw1_embedding/main.py](hw1_embedding/main.py).

### Task 2: RNN News Classification (60%)

Complete **four TODOs** in [hw1_rnn/main.py](hw1_rnn/main.py).

## Setup

```shell
python3 -m venv .venv 
. .venv/bin/activate
python -m pip install -r requirements.txt # create, activate and setup venv

python verify.py  # after completing the TODOs
python main.py # run both 
python playground.py # See section Playground
```

You may run each task separately with `python hw1_embedding/main.py` and then
`python hw1_rnn/main.py`. `python verify.py` runs a few small, provided checks.

Text processing, word IDs, batching, negative sampling, parameter initialization,
SGD updates, evaluation, and saving are already implemented. Modify only TODO
blocks. 

The accuracy of a correct implementation should be greater than 0.85.

## Data

The fixed AG News subset contains 20,000 training and 2,000 validation documents.
Labels: World=0, Sports=1, Business=2, Sci/Tech=3. Use training texts to learn word
vectors and train the classifier; evaluate classification on validation texts.
The files are supplied locally, so no dataset download or tokenizer installation is needed.

## Files to Submit

```text
hw1_embedding/
  main.py
  outputs/
    embedding.pt
    loss_curve.png
hw1_rnn/
  main.py
  outputs/
    rnn.pt
    loss_curve.png
    results.json
```

Compress only these files as `<intl_id>.zip`. Do not include the dataset, virtual
environment, cache, or instructor materials. No separate comparison report is required.

## Grading Policy

| Item                                                     | Weight in the whole homework |
| -------------------------------------------------------- | ---------------------------: |
| Task 1: four TODOs                                       |                      Each 8% |
| Task 1: trained vectors and loss curve                   |                           8% |
| Task 2: four TODOs                                       |                     Each 12% |
| Task 2: saved model, loss curve, correct Accuracy result |                          12% |

The model must load and its reported Accuracy must match its actual predictions.
Real-word neighbors need not match a fixed semantic word list. Hyperparameter
tuning, multiple seeds, comparison experiments and extra metrics are not required.

## Optional CLI Playground (provided; not graded)

After completing the TODOs and training, inspect the model interactively:

```shell
python playground.py
python playground.py --text "The team won the football championship."
python playground.py --text "The company reported higher profits." --json
```

Enter one text per line; `/quit`, Ctrl-D (Windows: Ctrl-Z then Enter), or Ctrl-C
exits. A different model can be selected with `--checkpoint /path/to/rnn.pt`.
The default checkpoint is located relative to the script, regardless of the
current working directory. The playground uses CPU and adds no dependencies.

## Playground Examples

These illustrative inputs are for experimentation; predictions may be incorrect.

- World: "The government held diplomatic talks with neighboring countries."
- Sports: "The team won the football championship after scoring in extra time."
- Business: "The company reported higher profits and increased quarterly revenue."
- Sci/Tech: "Researchers developed a new computer chip for artificial intelligence."

The model always selects one of the four training categories, including for texts
outside those categories.
