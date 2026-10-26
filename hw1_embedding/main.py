# =========================
# Skip-gram Word Embeddings (Student Version)
# Fill ONLY TODO[1-4] blocks. The training/data/saving code is provided.
# Deliverables: outputs/embedding.pt and outputs/loss_curve.png
# =========================
"""Task 1: learn word vectors, then inspect cosine nearest neighbors."""

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F  # noqa: F401 -- supplied for TODO[3]
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.preprocess import PAD_ID, UNK_ID, build_vocab, encode_tokens, tokenize  # noqa: E402
from common.utils import DATA_DIR, read_jsonl, sample_pairs, save_loss_curve  # noqa: E402

EPOCH, BATCH_SIZE, LR = 10, 512, 10.0  # You may tune these three values.
EMBEDDING_DIM, WINDOW, NEGATIVES = 64, 2, 5
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def generate_pairs(document: list[int], window: int = WINDOW) -> list[tuple[int, int]]:
    """Return (center, context) pairs for ONE document, keeping its original positions."""
    # === TODO[1] BEGIN: nearby-word pairs
    # For each position i, visit its neighbors within +/-window; do not pair i with itself.
    # Skip pairs containing PAD_ID or UNK_ID. Return a list of ID pairs.
    pairs = []
    for i in range(len(document)):
        center = document[i]
        if (center in (PAD_ID,UNK_ID)):
            continue
        for j in range(max(0,i-window),min(len(document),i+window+1)):
            context= document[j]
            if j == i :
                continue
            if (context in (PAD_ID,UNK_ID)):
                continue
            pairs.append((center,context))
        
    return pairs
    # === TODO[1] END


class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, dimension: int = EMBEDDING_DIM):
        super().__init__()
        self.input_embedding = nn.Parameter(torch.empty(vocab_size, dimension))
        self.output_embedding = nn.Parameter(torch.zeros(vocab_size, dimension))
        nn.init.uniform_(self.input_embedding, -0.5, 0.5)
        with torch.no_grad():
            self.input_embedding[PAD_ID].zero_()

    def forward(self, center_ids, context_ids, negative_ids):
        """IDs have shapes [B], [B], [B,K]; return scores [B] and [B,K]."""
        # === TODO[2] BEGIN: lookup and dot products
        # Centers use input_embedding; context/negative words use output_embedding.
        # Take row vectors by ID, then compute each center/context dot product.
        centers = self.input_embedding[center_ids]          # [B, D]
        contexts = self.output_embedding[context_ids]       # [B, D]
        negatives = self.output_embedding[negative_ids]     # [B, K, D]
        positive_scores = (centers * contexts).sum(dim=1)   # [B]
        negative_scores = (centers.unsqueeze(1) * negatives).sum(dim=2)  # [B, K]
        return positive_scores, negative_scores
        # === TODO[2] END

    def loss(self, positive_scores, negative_scores):
        """Average the per-pair negative-sampling loss over the batch."""
        # === TODO[3] BEGIN: negative-sampling objective
        # -log(sigmoid(positive)) - sum_k log(sigmoid(-negative_k)).
        # Use the supplied F.logsigmoid; sum negatives over K, then mean over B.
        positive_loss = -F.logsigmoid(positive_scores)                # [B]
        negative_loss = -F.logsigmoid(-negative_scores).sum(dim=1)    # [B]
        return (positive_loss + negative_loss).mean()                 # one number
        # === TODO[3] END


def nearest_neighbors(embedding, query_id: int, k: int = 5) -> list[int]:
    """Compute cosine scores. Filtering and ranking are already provided."""
    norms = embedding.norm(dim=1).clamp_min(1e-8)  # noqa: F841 -- used by TODO[4]
    # === TODO[4] BEGIN: cosine similarity
    # Compute similarities [V] between every row and embedding[query_id].
    # Cosine = dot(u,v) / (norm(u)*norm(v)); use the provided norms.
    dots = (embedding * embedding[query_id]).sum(dim=1)      # [V, D] * [D] -> [V, D] -> [V]
    similarities = dots / (norms * norms[query_id])          # [V] / ([V] * one number) -> [V]
    # === TODO[4] END
    similarities[:3] = float("-inf")
    similarities[query_id] = float("-inf")
    return similarities.argsort(descending=True)[:k].tolist()


def main() -> None:
    if EPOCH < 1 or BATCH_SIZE < 1 or LR <= 0:
        raise ValueError("EPOCH, BATCH_SIZE and LR must be positive.")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    documents = [tokenize(row["text"]) for row in read_jsonl(DATA_DIR / "train.jsonl")]
    vocab = build_vocab(documents)
    encoded = [encode_tokens(document, vocab) for document in documents]
    pairs = sample_pairs(encoded, generate_pairs)
    loader = DataLoader(pairs, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    model = SkipGram(len(vocab))
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    losses = []
    print(f"VOCAB_SIZE: {len(vocab)} | TRAINING_PAIRS: {len(pairs)}")
    for epoch in range(1, EPOCH + 1):
        total_loss = 0.0
        for batch in loader:
            # Provided noise sampling: uniformly choose words, excluding PAD/UNK.
            negative_ids = torch.randint(2, len(vocab), (len(batch), NEGATIVES))
            optimizer.zero_grad()
            loss = model.loss(*model(batch[:, 0], batch[:, 1], negative_ids))
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss is not finite.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        losses.append(total_loss / len(pairs))
        print(f"Epoch: {epoch} | train loss: {losses[-1]:.4f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vectors = model.input_embedding.detach()
    torch.save({"embedding": vectors, "vocab": vocab}, OUT_DIR / "embedding.pt")
    save_loss_curve(OUT_DIR / "loss_curve.png", losses, "Skip-gram Training Loss")
    id_to_word = {index: word for word, index in vocab.items()}
    for word in ("football", "bank", "computer"):
        if word in vocab:
            neighbors = nearest_neighbors(vectors, vocab[word])
            print(f"NEIGHBORS {word}: {[id_to_word[index] for index in neighbors]}")
    print("SAVED_EMBEDDING:", OUT_DIR / "embedding.pt")
    print("SAVED_LOSS_FIG:", OUT_DIR / "loss_curve.png")


if __name__ == "__main__":
    main()
