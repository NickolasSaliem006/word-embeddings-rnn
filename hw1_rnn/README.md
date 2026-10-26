# Task 2: News Classification with an RNN

## Setup

Finish Task 1 first. This task loads its word-vector table and vocabulary.
Text processing, padding, affine-layer initialization, training, evaluation, and
model saving are provided. 

## Model

```text
Word IDs -> word vectors -> one tanh RNN layer -> mean of real hidden states
         -> Linear classifier -> four class scores
```

The provided input and hidden Linear layers implement `W_e*x+b` and `W_h*h`.
At each time step, use `h_t = tanh(W_e*x_t + W_h*h_previous + b)`.
Start with zero `h_0`; the same layers are reused at every time step.

## TODO

| TODO | What to implement                                         |
| ---- | --------------------------------------------------------- |
| 1    | One recurrent update                                      |
| 2    | Repeat the update over time and collect hidden states     |
| 3    | Average a document's real hidden states                   |
| 4    | Connect embedding lookup, RNN, pooling and classification |

TODO[2] receives `[B,T,D]` and returns `[B,T,H]`. Use `inputs.new_zeros` for the
initial state and `torch.stack(..., dim=1)` to collect states. Do not detach states;
PyTorch autograd performs backward propagation for you.

TODO[3] receives ONE document's real `[L,H]` states; simply average over its sequence
dimension. The supplied `pool_batch` removes PAD positions before calling it.

TODO[4] returns raw `[B,4]` scores. The supplied CrossEntropyLoss handles softmax;
do not add another softmax. Use the supplied Linear layers rather than a library
RNN/RNNCell/GRU/LSTM, because implementing recurrent computation is the exercise.

## Run and Expected Result

```shell
python hw1_rnn/main.py
```

Defaults: embedding dimension=64, hidden size=64, max length=64, 20 epochs,
batch=128, SGD lr=0.5. Training updates the word vectors as well as the RNN.

Expected result: train successfully, save a model and curve, and report validation
Accuracy. The framework computes `correct / total` over the complete validation
set. The accuracy of a correct implementation should be greater than 0.85.
