"""Small synthetic sparse-attention check; no model weights or dataset required."""
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vasc_entry", ROOT / "evaluate.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
sys.path.insert(0, str(ROOT / "src"))


def main():
    import torch
    from sparse_vggt.models.attention import predict_attention
    from sparse_vggt.utils.sparse_wrapper import block_sparse_attn_cuda

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this check")
    torch.manual_seed(0)
    cases = 0
    for mode in ("topk", "vasc"):
        for key in list(os.environ):
            if key.startswith("SPARSE_VGGT_"):
                del os.environ[key]
        os.environ.update(entry.method_environment("vggt", mode))
        state = {}
        for layer in range(3):
            q, k, v = [torch.randn(1, 2, 512, 64, device="cuda", dtype=torch.float16)
                       for _ in range(3)]
            score = predict_attention(q, k)
            result = block_sparse_attn_cuda(q, k, v, score, sparse_ratio=.75,
                                           num_patch_tokens=512,
                                           layer_idx=layer, routing_state=state)
            output = result[0] if isinstance(result, tuple) else result
            assert output.shape == q.shape
            assert torch.isfinite(output).all()
            assert output.float().abs().sum() > 0
            cases += 1
    torch.cuda.synchronize()
    print(f"PASS: {cases} CUDA attention calls on {torch.cuda.get_device_name()}")


if __name__ == "__main__":
    main()
