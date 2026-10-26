"""Provided data loading, batching, pair sampling, and plotting utilities."""

import json
import random
from pathlib import Path

import torch

from common.preprocess import PAD_ID, UNK_ID, encode_tokens, tokenize

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sample_pairs(documents, pair_function, limit: int = 200_000) -> torch.Tensor:
    """Keep a small, reproducible sample without storing the whole corpus's pairs."""
    rng, pairs, seen = random.Random(42), [], 0
    for document in documents:
        for pair in pair_function(document):
            if seen < limit:
                pairs.append(pair)
            else:
                index = rng.randrange(seen + 1)
                if index < limit:
                    pairs[index] = pair
            seen += 1
    if not pairs:
        raise ValueError("No training pairs; check the context window implementation.")
    return torch.tensor(pairs, dtype=torch.long)


def prepare_examples(rows: list[dict], vocab: dict[str, int]):
    return [(encode_tokens(tokenize(row["text"]), vocab)[:64], row["label"]) for row in rows]


def collate_batch(batch):
    """Provided variable-length batching: real lengths and right-side PAD tokens."""
    sequences = [sequence or [UNK_ID] for sequence, _ in batch]
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    ids = torch.full((len(batch), int(lengths.max())), PAD_ID, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = torch.tensor(sequence)
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    return ids, lengths, labels


def save_loss_curve(path: Path, losses: list[float], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
