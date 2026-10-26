"""Provided CPU inference playground; no additional student TODOs or dependencies."""

import argparse
import json
from pathlib import Path

import torch

from common.preprocess import UNK_ID, tokenize
from common.utils import prepare_examples
from hw1_rnn.main import RNNClassifier

ROOT = Path(__file__).resolve().parent
CLASSES = ("World", "Sports", "Business", "Sci/Tech")
DEFAULT_CHECKPOINT = ROOT / "hw1_rnn" / "outputs" / "rnn.pt"


class Playground:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Model not found: {checkpoint}. Run main.py after completing the TODOs, "
                "or pass --checkpoint /path/to/rnn.pt."
            )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.vocab = saved["vocab"]
        self.id_to_word = {index: word for word, index in self.vocab.items()}
        state = saved["model_state_dict"]
        # Load fine-tuned vectors from rnn.pt, not the earlier Skip-gram checkpoint.
        self.model = RNNClassifier(state["embedding"], saved["config"]["hidden_size"])
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text):
        tokens = tokenize(text)
        # Reuse exactly the training pipeline, including 64-token truncation
        # and the one-UNK fallback when tokenization yields an empty document.
        ids, _ = prepare_examples([{"text": text, "label": 0}], self.vocab)[0]
        logits = self.model(torch.tensor([ids]), torch.tensor([len(ids)]))[0]
        logprobs = torch.log_softmax(logits, dim=-1)
        if not torch.isfinite(logprobs).all():
            raise ValueError("Model produced non-finite log probabilities.")
        return {
            "text": text,
            "tokens": tokens,
            "input_tokens": [self.id_to_word[index] for index in ids],
            "input_ids": ids,
            "token_count": len(tokens),
            "input_length": len(ids),
            "truncated_count": max(0, len(tokens) - len(ids)),
            "unk_count": sum(index == UNK_ID for index in ids),
            "empty_fallback": not tokens,
            "prediction": CLASSES[int(logprobs.argmax())],
            "classes": [
                {"id": i, "label": label, "logprob": float(logprobs[i]),
                 "probability": float(logprobs[i].exp())}
                for i, label in enumerate(CLASSES)
            ],
        }


def display(result):
    print("TOKENS:", json.dumps(result["tokens"], ensure_ascii=False))
    print("INPUT_TOKENS:", json.dumps(result["input_tokens"], ensure_ascii=False))
    print("INPUT_IDS:", result["input_ids"])
    print(f"Tokens: {result['token_count']} | Used: {result['input_length']} | "
          f"Truncated: {result['truncated_count']} | UNK: {result['unk_count']}")
    if result["empty_fallback"]:
        print("Empty input: the model receives one <UNK>, matching training preprocessing.")
    print("\nClass         logprob (ln)    probability")
    for row in result["classes"]:
        print(f"{row['label']:<12} {row['logprob']:>12.6f} {row['probability']:>13.4%}")
    print("PREDICTION:", result["prediction"])


def main():
    parser = argparse.ArgumentParser(description="Inspect tokenization and RNN class log probabilities.")
    parser.add_argument("--text", help="Predict once; omit for an interactive session.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--json", action="store_true", help="Output one JSON object per prediction.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        playground = Playground(args.checkpoint)
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"playground: {exc}\n")

    def predict(text):
        try:
            result = playground.predict(text)
        except NotImplementedError:
            parser.exit(2, "playground: Complete the RNN TODOs before running inference.\n")
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            display(result)

    if args.text is not None:
        predict(args.text)
        return
    import sys
    interactive = sys.stdin.isatty()
    if interactive:
        print("Enter news text (English training data). /quit or Ctrl-D exits.")
        print("logprob = ln P(class | text), not a language-model token probability.")
    while True:
        try:
            text = input("text> " if interactive else "")
        except (EOFError, KeyboardInterrupt):
            if interactive:
                print()
            break
        if text.strip() == "/quit":
            break
        predict(text)
        if interactive:
            print()


if __name__ == "__main__":
    main()
