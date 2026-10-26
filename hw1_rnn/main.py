import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.utils import (  # noqa: E402
    DATA_DIR,
    collate_batch,
    prepare_examples,
    read_jsonl,
    save_loss_curve,
)

EPOCH, BATCH_SIZE, LR = 20, 128, 0.5  # You may tune these three values.
HIDDEN_SIZE, NUM_CLASSES = 64, 4
OUT_DIR = Path(__file__).resolve().parent / "outputs"


class ManualRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        # Provided affine maps: input_layer includes b; hidden_layer has no second bias.
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def step(self, x_t, previous):
        """x_t [B,D], previous [B,H] -> new state [B,H]."""
        # === TODO[1] BEGIN: recurrent update
        # h_t = tanh(W_e*x_t + W_h*h_prev + b).
        # Use the two supplied Linear maps; do not call nn.RNN/RNNCell/GRU/LSTM.
        return torch.tanh(self.input_layer(x_t) + self.hidden_layer(previous))
        # === TODO[1] END

    def forward(self, inputs):
        """inputs [B,T,D] -> all hidden states [B,T,H]."""
        # === TODO[2] BEGIN: repeat the shared step over time
        # Begin with zero h_0. For each t: update h via step, append h to a list.
        # torch.stack(..., dim=1) returns all states. Do not detach intermediate h.
        all_h = []
        h = torch.zeros(inputs.shape[0], self.hidden_size, dtype=inputs.dtype, device=inputs.device)
        for i in range(inputs.shape[1]):
            h=self.step(inputs[:, i], h)
            all_h.append(h)
        return torch.stack(all_h, dim=1)
        # === TODO[2] END


def mean_pool(valid_states):
    """One document's REAL states [L,H] -> a sentence vector [H]; no PAD handling needed."""
    # === TODO[3] BEGIN: average hidden states
    # Average over the sequence dimension, not the hidden-feature dimension.
    return valid_states.mean(dim=0)
    # === TODO[3] END


def pool_batch(states, lengths):
    """Provided batching utility: remove padding before calling the student's mean_pool."""
    return torch.stack(
        [mean_pool(states[index, :length]) for index, length in enumerate(lengths.tolist())]
    )


class RNNClassifier(nn.Module):
    def __init__(self, vectors, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.embedding = nn.Parameter(vectors.detach().clone())
        self.rnn = ManualRNN(vectors.shape[1], hidden_size)
        self.classifier = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, ids, lengths):
        """IDs [B,T] -> four raw class scores [B,4]."""
        # === TODO[4] BEGIN: complete the classification model
        # Lookup self.embedding[ids] -> self.rnn -> pool_batch -> self.classifier.
        # Return raw scores; the supplied CrossEntropyLoss handles softmax internally.
        vectors = self.embedding[ids]            # [B, T]    -> [B, T, D]
        states = self.rnn(vectors)               # [B, T, D] -> [B, T, H]
        pooled = pool_batch(states, lengths)     # [B, T, H] -> [B, H]
        return self.classifier(pooled)           # [B, H]    -> [B, 4]
        # === TODO[4] END


@torch.no_grad()
def evaluate(model, loader):
    """Provided evaluation: count correct predictions over the whole split."""
    model.eval()
    correct, total = 0, 0
    for ids, lengths, labels in loader:
        predictions = model(ids, lengths).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += len(labels)
    return {"accuracy": correct / total, "correct": correct, "total": total}


def main() -> None:
    if EPOCH < 1 or BATCH_SIZE < 1 or LR <= 0:
        raise ValueError("EPOCH, BATCH_SIZE and LR must be positive.")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    embedding_file = ROOT / "hw1_embedding" / "outputs" / "embedding.pt"
    if not embedding_file.is_file():
        raise FileNotFoundError("Run Task 1 first to create outputs/embedding.pt.")
    checkpoint = torch.load(embedding_file, map_location="cpu", weights_only=True)
    vectors, vocab = checkpoint["embedding"], checkpoint["vocab"]
    train = prepare_examples(read_jsonl(DATA_DIR / "train.jsonl"), vocab)
    valid = prepare_examples(read_jsonl(DATA_DIR / "valid.jsonl"), vocab)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    model = RNNClassifier(vectors)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    loss_function = nn.CrossEntropyLoss()
    losses = []
    print(f"TRAIN_SIZE: {len(train)} | VALID_SIZE: {len(valid)}")
    for epoch in range(1, EPOCH + 1):
        model.train()
        total_loss = 0.0
        for ids, lengths, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(ids, lengths), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss is not finite.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        losses.append(total_loss / len(train))
        print(f"Epoch: {epoch} | train loss: {losses[-1]:.4f}")
    result = {**evaluate(model, valid_loader), "train_loss": losses}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "config": {"embedding_dim": vectors.shape[1], "hidden_size": HIDDEN_SIZE},
        },
        OUT_DIR / "rnn.pt",
    )
    save_loss_curve(OUT_DIR / "loss_curve.png", losses, "RNN Training Loss")
    (OUT_DIR / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"VALID_ACCURACY: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
    print("SAVED_MODEL:", OUT_DIR / "rnn.pt")
    print("SAVED_LOSS_FIG:", OUT_DIR / "loss_curve.png")


if __name__ == "__main__":
    main()
