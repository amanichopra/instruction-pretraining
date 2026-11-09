"""Verify the vectorized/chunked continuation scorer against a naive reference."""
import os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model
from src.evaluation.benchmarks import score_continuations


@torch.no_grad()
def reference(model, tok, pairs, device, max_len=2048, autocast=True):
    """One pair at a time, no padding, explicit per-token loop."""
    import contextlib
    out = []
    for ctx, cont in pairs:
        c = tok(ctx, add_special_tokens=False)["input_ids"] if ctx else [tok.eos_token_id]
        k = tok(cont, add_special_tokens=False)["input_ids"] or [tok.eos_token_id]
        seq = (c + k)[-max_len:]
        nk = min(len(k), len(seq) - 1)
        inp = torch.tensor([seq], device=device)
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if autocast
               else contextlib.nullcontext())
        with ctx:
            logits = model(input_ids=inp).logits.float()
        lsm = F.log_softmax(logits, dim=-1)
        lp, greedy = 0.0, True
        for t in range(len(seq) - nk, len(seq)):
            lp += lsm[0, t - 1, seq[t]].item()
            if lsm[0, t - 1].argmax().item() != seq[t]:
                greedy = False
        out.append((lp, nk, greedy))
    return out


def main():
    dev = "cuda"
    torch.manual_seed(0)
    model, _ = build_model(size="tiny", seq_len=1024, dtype=torch.float32, verbose=False)
    model.to(dev).eval()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    # deliberately ragged lengths, to exercise padding and the masking
    pairs = [
        ("Question: What is the capital of France?\nAnswer:", " Paris"),
        ("Question: What is the capital of France?\nAnswer:", " a large city in western Europe known for food"),
        ("The quick brown fox", " jumps over the lazy dog and keeps running for a while"),
        ("", " short"),
        ("A much longer context " * 30, " tail tokens here"),
    ]
    # fp32 path: pure logic check, no precision noise
    ref32 = reference(model, tok, pairs, dev, autocast=False)
    from src.ce_utils import per_token_scores
    import src.evaluation.benchmarks as B
    orig = B.per_token_scores
    B.per_token_scores = lambda m, x, y, **kw: per_token_scores(m, x, y, autocast=False, **kw)
    for bs in (1, 2, 5):
        got = B.score_continuations(model, tok, pairs, dev, batch_size=bs)
        worst = max(abs(g[0] - r[0]) for g, r in zip(got, ref32))
        nok = all(g[1] == r[1] for g, r in zip(got, ref32))
        gok = all(g[2] == r[2] for g, r in zip(got, ref32))
        print(f"  [fp32] batch_size={bs}: max|dlogprob|={worst:.2e} n_tokens_match={nok} "
              f"greedy_match={gok}  {'OK' if worst < 1e-3 and nok and gok else 'FAIL'}")
        assert worst < 1e-3 and nok and gok, "vectorized scorer disagrees with reference"
    B.per_token_scores = orig

    # bf16 path: same logic, looser tolerance for reduced precision
    ref = reference(model, tok, pairs, dev, autocast=True)
    for bs in (1, 2, 5):
        got = score_continuations(model, tok, pairs, dev, batch_size=bs)
        assert len(got) == len(ref)
        worst_lp = max(abs(g[0] - r[0]) for g, r in zip(got, ref))
        ntok_ok = all(g[1] == r[1] for g, r in zip(got, ref))
        greedy_ok = all(g[2] == r[2] for g, r in zip(got, ref))
        print(f"  [bf16] batch_size={bs}: max|dlogprob|={worst_lp:.2e}  n_tokens_match={ntok_ok}  "
              f"greedy_match={greedy_ok}  {'OK' if worst_lp < 0.1 and ntok_ok and greedy_ok else 'FAIL'}")
        assert worst_lp < 0.1 and ntok_ok and greedy_ok
    print("ALL SCORING TESTS PASSED")


if __name__ == "__main__":
    main()
