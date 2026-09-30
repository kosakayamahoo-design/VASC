"""CPU checks on the actual selector functions, without CUDA imports or weights."""
import ast
import os
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def load_selector(source=None):
    path = ROOT / "src/sparse_vggt/utils/sparse_wrapper.py"
    wanted = {"_project_conservative_capped_mass", "_reduce_pv_service_to_pair",
              "_reduce_qk_admission_to_pair", "_reduce_qk_indices_to_pair",
              "get_cosa_value_risk_pair_debt_mask", "finalize_cosa_value_risk_pair_debt"}
    nodes = [n for n in ast.parse(source if source is not None else path.read_text(encoding="utf-8")).body
             if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    namespace = dict(torch=torch, F=F, os=os)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20260906)
    for key in list(os.environ):
        if key.startswith("SPARSE_VGGT_"):
            del os.environ[key]
    settings = dict(FUSED_ROUTING="0", CONSERVATIVE_ARRIVAL_PROJECTION="1",
                    CONSERVATIVE_DEBT_PROJECTION="1", INPLACE_ARRIVAL_PROJECTION="0",
                    UNIFORM_CAPACITY_HINT="0", COUNTERFACTUAL_STATS="0",
                    SERVICE_LEDGER_OBSERVER="0", POSTCUT_ORDER_OBSERVER="0",
                    DEBT_SERVICE_DOMAIN="qk_admission", INDEX_SERVICE_REDUCTION="1",
                    SORT_SELECTED_INDICES="1")
    os.environ.update({"SPARSE_VGGT_COSA_" + k: v for k, v in settings.items()})
    ns = load_selector()
    fn = ns["get_cosa_value_risk_pair_debt_mask"]
    cases = memory_active = 0
    for blocks in (1, 2, 7, 8, 31):
        for protect_last in (False, True):
            state = {}
            for layer in range(24):
                scores = torch.randn(1, 2, 3, blocks).softmax(-1)
                values = torch.randn(1, 2, blocks, 13)
                capacity = torch.randint(blocks + 1, (1, 2, 3, 1))
                base = torch.arange(blocks).view(1, 1, 1, -1) < capacity
                protected = torch.zeros_like(base)
                if protect_last:
                    protected[..., -1:] = capacity > 0
                selected = fn(scores, values, base, layer_idx=layer, routing_state=state,
                              protected_mask=protected, fp32_projection=True)
                assert torch.equal(selected.sum(-1, keepdim=True), capacity)
                assert bool(selected[protected].all()), "protected blocks were dropped"
                assert bool(torch.isfinite(state["cosa_value_risk_pair_priority"]).all())
                if bool((state["cosa_value_risk_pair_debt"] > 0).any()):
                    memory_active += 1
                cases += 1
            terminal = ns["finalize_cosa_value_risk_pair_debt"](state)
            assert terminal
            assert ns["finalize_cosa_value_risk_pair_debt"](state) == {}
    assert memory_active > 0
    print(f"PASS: {cases} selector cases; exact capacity, protected blocks, terminal settlement; memory active in {memory_active} cases")


if __name__ == "__main__":
    main()
