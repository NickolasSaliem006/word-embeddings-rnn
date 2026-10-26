"""Optional small checks of the lecture computations; complete the TODOs first."""

import importlib.util
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


def load_module(name, task):
    spec = importlib.util.spec_from_file_location(name, ROOT / task / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    torch.manual_seed(42)
    embedding = load_module("embedding_task", "hw1_embedding")
    assert embedding.generate_pairs([3, 4, 5], window=1) == [(3, 4), (4, 3), (4, 5), (5, 4)]
    assert embedding.generate_pairs([3, 1, 4], window=1) == []
    assert embedding.generate_pairs([3, 1, 4], window=2) == [(3, 4), (4, 3)]
    model = embedding.SkipGram(8, dimension=3)
    with torch.no_grad():
        model.input_embedding.copy_(torch.arange(24).reshape(8, 3) / 10)
        model.output_embedding.copy_(torch.arange(24).reshape(8, 3).flip(0) / 10)
    pos, neg = model(torch.tensor([3]), torch.tensor([4]), torch.tensor([[5, 6]]))
    torch.testing.assert_close(
        pos, (model.input_embedding[3] @ model.output_embedding[4]).reshape(1)
    )
    expected_neg = torch.stack(
        [model.input_embedding[3] @ model.output_embedding[i] for i in (5, 6)]
    )
    torch.testing.assert_close(neg, expected_neg.reshape(1, 2))
    torch.testing.assert_close(
        model.loss(pos, neg), (F.softplus(-pos) + F.softplus(neg).sum(1)).mean()
    )
    vectors = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]]
    )
    assert embedding.nearest_neighbors(vectors, 3, k=1) == [4]
    print("Task 1 checks passed.")
    rnn = load_module("rnn_task", "hw1_rnn")
    recurrent = rnn.ManualRNN(3, hidden_size=4)
    inputs = torch.randn(2, 3, 3)
    expected, hidden = [], torch.zeros(2, 4)
    for time in range(3):
        hidden = torch.tanh(recurrent.input_layer(inputs[:, time]) + recurrent.hidden_layer(hidden))
        expected.append(hidden)
    torch.testing.assert_close(recurrent(inputs), torch.stack(expected, dim=1))
    torch.testing.assert_close(
        rnn.mean_pool(torch.tensor([[1.0, 3.0], [3.0, 5.0]])), torch.tensor([2.0, 4.0])
    )
    classifier = rnn.RNNClassifier(torch.randn(8, 3), hidden_size=4)
    output = classifier(torch.tensor([[3, 4], [5, 0]]), torch.tensor([2, 1]))
    assert output.shape == (2, 4)
    single = classifier(torch.tensor([[3, 4]]), torch.tensor([2]))
    padded = classifier(torch.tensor([[3, 4, 0, 0]]), torch.tensor([2]))
    torch.testing.assert_close(single, padded)
    recurrent = rnn.ManualRNN(3, hidden_size=4).double()
    inputs = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    recurrent(inputs)[:, -1].sum().backward()
    assert inputs.grad[:, 0].abs().sum() > 0, "Gradient must reach earlier time steps."
    print("Task 2 checks passed.")


if __name__ == "__main__":
    main()
