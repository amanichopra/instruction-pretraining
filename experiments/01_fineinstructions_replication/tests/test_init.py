"""Guard the experiment's controls:
  1. both arms start from bit-identical random weights (same --seed)
  2. a different seed really does change the init
  3. the init carries no pretrained knowledge
"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model, verify_randomized


def main():
    m1, _ = build_model(seed=0, dtype=torch.float32, verbose=False)
    m2, _ = build_model(seed=0, dtype=torch.float32, verbose=False)
    m3, _ = build_model(seed=1, dtype=torch.float32, verbose=False)
    d1, d2, d3 = (dict(m.named_parameters()) for m in (m1, m2, m3))

    same = max((d1[k] - d2[k]).abs().max().item() for k in d1)
    print(f"  seed0 vs seed0: max|delta|={same:.2e}")
    assert same == 0.0, "same seed must give identical init (arms would not be controlled)"

    diff = max((d1[k] - d3[k]).abs().max().item() for k in d1)
    print(f"  seed0 vs seed1: max|delta|={diff:.2e}")
    assert diff > 0, "different seeds must give different init"

    v = verify_randomized(m1, "HuggingFaceTB/SmolLM2-135M")
    if v.get("checked"):
        print(f"  vs pretrained checkpoint: max|cos|="
              f"{v['max_abs_cosine_sim_knowledge_tensors']:.5f} over "
              f"{v['n_knowledge_tensors_checked']} knowledge tensors")
        assert v["is_random_init"], "model appears to retain pretrained weights"
    else:
        print(f"  (skipped pretrained comparison: {v.get('reason')})")
    print("ALL INIT TESTS PASSED")


if __name__ == "__main__":
    main()
