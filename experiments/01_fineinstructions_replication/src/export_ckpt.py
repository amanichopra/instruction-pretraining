"""Export a mid-training checkpoint (runs/<name>/ckpt/model.pt) to an HF model dir
so the evaluation scripts can load it while training continues."""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="runs/<name>")
    p.add_argument("--out", default=None, help="default: <run>/hf_step<N>")
    p.add_argument("--model-id", default=None)
    p.add_argument("--size", default=None)
    a = p.parse_args()

    cfg = json.load(open(os.path.join(a.run, "config.json")))
    model_id = a.model_id or cfg["model_id"]
    size = a.size or cfg["size"]
    ckpt = os.path.join(a.run, "ckpt", "model.pt")
    sd = torch.load(ckpt, map_location="cpu")
    step, tokens = sd.get("step"), sd.get("tokens")

    model, _ = build_model(model_id, size, seq_len=cfg["seq_len"], dtype=torch.bfloat16,
                           verbose=False)
    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    if missing or unexpected:
        print(f"WARNING missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    out = a.out or os.path.join(a.run, f"hf_step{step}")
    model.save_pretrained(out, safe_serialization=True)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out)
    json.dump(dict(step=step, tokens=tokens, run=a.run), open(os.path.join(out, "provenance.json"), "w"), indent=2)
    print(f"exported {a.run} step={step} tokens={tokens} -> {out}")


if __name__ == "__main__":
    main()
