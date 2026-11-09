"""Verify the chunked/checkpointed loss equals a plain full-batch cross-entropy."""
import os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model
from src.train import loss_fn

def reference(model, x, y, mask=None):
    out = model(input_ids=x)
    logits = out.logits.float()
    ls = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    if mask is None:
        return ls.mean()
    m = mask.reshape(-1).float()
    return (ls * m).sum() / m.sum().clamp(min=1.0)

def main():
    dev = "cuda"
    torch.manual_seed(0)
    model, cfg = build_model(size="tiny", seq_len=256, dtype=torch.float32, verbose=False)
    model.to(dev).train()
    B, L = 3, 128
    x = torch.randint(0, cfg.vocab_size, (B, L), device=dev)
    y = torch.randint(0, cfg.vocab_size, (B, L), device=dev)
    mask = (torch.rand(B, L, device=dev) > 0.4).long()

    for name, mk in [("no_mask", None), ("mask", mask)]:
        for chunk in (0, 37, 128, 4096):
            model.zero_grad(set_to_none=True)
            lo, _, _ = loss_fn(model, x, y, mk, chunk_tokens=chunk)
            lo.backward()
            g_chunk = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
            model.zero_grad(set_to_none=True)
            lr_ = reference(model, x, y, mk)
            lr_.backward()
            g_ref = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
            dl = abs(lo.item() - lr_.item())
            dg = (g_chunk - g_ref).abs().max().item()
            gn = g_ref.abs().max().item()
            ok = dl < 1e-4 and dg < max(1e-4, 1e-3 * gn)
            print(f"  {name:8s} chunk={chunk:5d}  loss {lo.item():.6f} vs {lr_.item():.6f} "
                  f"(d={dl:.2e})  max|dgrad|={dg:.2e} (max|grad|={gn:.2e})  "
                  f"{'OK' if ok else 'FAIL'}")
            assert ok, f"mismatch for {name} chunk={chunk}"
    print("ALL LOSS TESTS PASSED")

if __name__ == "__main__":
    main()
