"""A-6 头级归因 dump（1.7B 本地 GPU，S1 单次前向）。

理论锚点与判据：docs/protocol/a6-head-rescue-circuit-20260929.md §1/§3
  - 在 sym β=0.20 干预轨迹上逐步 dump 每个头 (l,h) 的直接 logit 归因：
    DLA_{l,h}(c;t) = (W_U · W_O^{l,h} · z^{l,h}_t)[c]，c ∈ {chosen_id, y_true_id}
  - 判读（零 GPU）：analyze_head_loc.py --judge（置乱检验 + 跨 seed Spearman）
红线：T04（l_final 为真实 logits）；与既有 sym 档同协议同 seed 同 β（判读比对用）。
用法：
  python experiments/lin_theory/dump_head_attribution.py \
    --model Qwen/Qwen3-1.7B --layer_early 20 --n_test 300 --seed_test 123 \
    --beta 0.20 --output_dir experiments/outputs/head_attribution
  python experiments/lin_theory/dump_head_attribution.py --selftest
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_sys_parent = Path(__file__).parent.parent
for _p in [
    str(Path(__file__).parent),
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_tldc_per_token import classify_samples, compute_early_exit_logits  # noqa: E402
from common import load_model_and_unembed  # noqa: E402
from src.data_loader import check_correct_exact, load_triviaqa  # noqa: E402


def head_contrib(z, W_O, W_U, b_U, vocab):
    """逐头对全词表的 logit 贡献：z [1,1,nH,dH] → [nH, V]。
    （z @ W_O[l,h]）@ W_U，与真实 logits 同空间（各头贡献之和＋偏置＝残差流 logits）。
    W_O 形状 [nH, dH, d_model]（TransformerLens 口径）。"""
    zf = z.float().squeeze(0).squeeze(0)               # [nH, dH]
    out = torch.einsum("hd,hdm->hm", zf, W_O.float())  # [nH, d_model]
    nH = zf.shape[0]
    out = out @ W_U.float() + b_U.float().view(1, -1) / max(nH, 1)
    return out  # [nH, V]


def selftest():
    ok = 0
    # 小规模手算：d_model=3, V=4, nH=2, dH=2
    z = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])          # [1,1,2,2]
    W_O = torch.tensor([[[1, 0, 0], [0, 1, 0]],            # 头 0 → 第 0 维
                        [[0, 1, 0], [0, 0, 1]]], dtype=torch.float32)  # 头 1 → 第 1 维
    W_U = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32)
    b_U = torch.zeros(4)
    c = head_contrib(z, W_O, W_U, b_U, 4)
    # 头 0：z=[1,0] @ W_O[0]=[1,0,0] → W_U → [1,0,0,0]
    # 头 1：z=[0,1] @ W_O[1]=[[0,1,0],[0,0,1]] → [0,0,1] → W_U → [0,0,1,0]
    assert c.shape == (2, 4)
    assert torch.allclose(c[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(c[1], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    ok += 1
    print(f"SELFTEST PASS: {ok}/1 checks (head_contrib 手算)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--seed_test", type=int, default=123)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--layer_early", type=int, default=20)
    ap.add_argument("--beta", type=float, default=0.20)
    ap.add_argument("--max_new", type=int, default=20)
    ap.add_argument("--output_dir", type=str, default="experiments/outputs/head_attribution")
    ap.add_argument("--ablate_head", type=str, default=None,
                    help="S2 零消融：'L-H' 置零该头（输出到 --output_dir，文件 headablate_*）")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"head-attr dump: model={args.model} β={args.beta} seed={args.seed_test} n={args.n_test}")

    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device, args.model)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    W_O = torch.stack([model.W_O[l] for l in range(n_layers)])  # [L, nH, dH, d_model]

    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)[: args.n_test]
    entries = classify_samples(model, tokenizer, test_samples, device,
                               args.layer_early, 50)
    print(f"  classify: KC={sum(1 for e in entries if e['subset']=='know_correct')}, "
          f"KW={sum(1 for e in entries if e['subset']=='know_wrong')}")

    hook_early = f"blocks.{args.layer_early}.hook_resid_post"
    hook_zs = [f"blocks.{l}.attn.hook_z" for l in range(n_layers)]
    captured = {"h_early": None, "z": {}}
    abl_l = abl_h = None
    if args.ablate_head is not None:
        abl_l, abl_h = (int(x) for x in args.ablate_head.split("-"))
        assert 0 <= abl_l < n_layers and 0 <= abl_h < n_heads

    def _mk_hook_early():
        def _hook(act, hook=None):
            captured["h_early"] = act[:, -1:, :].detach()
            return act
        return _hook

    def _mk_hook_z(l):
        def _hook(z, hook=None):
            z = z.clone()
            if l == abl_l:
                z[..., abl_h, :] = 0.0       # S2 零消融：置零该头
            captured["z"][l] = z.detach()
            return z
        return _hook

    fwd_hooks = [(hook_early, _mk_hook_early())] + \
                [(f"blocks.{l}.attn.hook_z", _mk_hook_z(l)) for l in range(n_layers)]

    samples_out = []
    from tqdm import tqdm
    for e in tqdm(entries, desc="  dump"):
        y_true_id = e["y_true_id"]
        tokens = model.to_tokens(e["prompt"], prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, -1024:]
        steps_log, gids = [], []
        for step in range(args.max_new):
            with torch.no_grad():
                logits_final = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            h_early = captured["h_early"]
            l_early = compute_early_exit_logits(h_early, ln_final, W_U, b_U)
            l_final = logits_final[0, -1:, :].float()
            l_combined = l_final + args.beta * (l_early - l_final)   # sym β
            nid = int(l_combined.argmax(dim=-1).item())
            gids.append(nid)
            zs = torch.stack([captured["z"][l] for l in range(n_layers)])  # [L,1,1,nH,dH]
            contrib = torch.empty(n_layers, n_heads, 2)
            for l in range(n_layers):
                c = head_contrib(zs[l], W_O[l], W_U, b_U, model.cfg.d_vocab_out)  # [nH, V]
                contrib[l, :, 0] = c[:, nid]
                contrib[l, :, 1] = c[:, y_true_id]
            steps_log.append({
                "step": step,
                "chosen_id": nid,
                "attr": contrib.flatten().tolist(),   # [L*nH*2]
            })
            if nid == tokenizer.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        gen_text = tokenizer.decode(gids).strip()
        samples_out.append({
            "sample_id": e["sample_id"], "subset": e["subset"],
            "y_true_id": y_true_id,
            "baseline_correct": e["is_correct"],
            "is_correct": check_correct_exact(gen_text, e["answers"]),
            "gids": gids, "steps": steps_log,
        })

    out = {
        "meta": {
            "model": args.model, "layer_early": args.layer_early, "beta": args.beta,
            "seed_test": args.seed_test, "n_test": args.n_test, "max_new": args.max_new,
            "n_layers": n_layers, "n_heads": n_heads,
            "ablate_head": [abl_l, abl_h],
            "theory_refs": ["docs/protocol/a6-head-rescue-circuit-20260929.md §1"],
            "script": Path(__file__).name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "samples": samples_out,
    }
    if args.ablate_head is None:
        tag = ("b" + f"{args.beta:.2f}").replace(".", "")
        fp = out_dir / f"headattr_sym_{tag}_seed{args.seed_test}.json"
    else:
        fp = out_dir / f"headablate_L{abl_l}_H{abl_h}_seed{args.seed_test}.json"
    with open(fp, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[saved] {fp} ({fp.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
