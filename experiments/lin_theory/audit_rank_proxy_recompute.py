"""秩代理口径复算：artifact 修复 + B1 别名 min-rank + B4 双口径（轻前向，一次/样本）。

背景（2026-09-24 零 GPU 审计，`audit_first_token_rank_proxy.py`）
  - KW 中"可疑首 token"28.7%（功能词 18.0% + 空格 artifact 10.7%）；别名敏感性 52.7%
  - 空格 artifact 是真 bug：BPE 把 " 9" 切成 [' ', '9']，`get_first_answer_token_id` 取到空格
    （id 220），其 rank ≡ 1 ⇒ 数字开头答案被无条件判 know
理论：docs/theory/theory-intervention-failure.md §1.2.1 / §1.2.1-a

**本脚本产出三套口径（同一批样本、同一次前向）**
  A) `old_first`  ：现口径 = 第一个非空别名的**原始首 token**（含空格 artifact）→ 用于复现已发布数字
  B) `fixed_first`：修复口径 = 第一个非空别名的**首个非空白 token**（跳过纯空白 token）
  C) `min_alias`  ：B1 口径 = min over 全部别名（各取首个非空白 token）的 rank
  另附 B4 列：`rank_lens` = ℓ* 早退（lens）logits 上的同一 token rank（final vs lens 是否同侧）

**预注册判据（2026-09-24 跑前写死，跑后不得改）**
  主判量 = KW 计数与 TLDC 救回率（需离线 join 干预档案，本脚本只出标签）在 B/C 口径下的变化
  - 若 |ΔKW| / KW_old >= 10%  或  救回率变化 >= 2pp  ⇒ 论文改用新口径并重算受影响数字
  - 若 < 5% 且 < 1pp                                  ⇒ 保留现口径 + 审计章披露
  - 其余＝灰区 ⇒ 报数不判，交用户/导师定
  ⚠️ 本脚本**不改**已发布数字；只产出复算所需的逐样本标签。

用法（本地 1.7B，分钟级）：
    bash scripts/review_local.sh rank-audit            # seed 123
    bash scripts/review_local.sh rank-audit-456        # seed 456
服务器 8B：
    unset HF_ENDPOINT && HF_HOME=/root/autodl-tmp/huggingface_cache python -u \
      experiments/lin_theory/audit_rank_proxy_recompute.py \
      --model Qwen/Qwen3-8B \
      --seed_test 123
输出：experiments/outputs/rank_audit_proxy/{rank_audit_seed<seed>_<tag>.json, summary_*.md}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from tqdm import tqdm

_sys_parent = Path(__file__).parent.parent
for _p in [str(_sys_parent / "phase2_entropy"), str(_sys_parent / "phase4_generalization"),
           str(_sys_parent / "phase5_cross_task")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data_loader import load_triviaqa, format_prompt, check_correct_exact  # noqa: E402
from common import (  # noqa: E402
    load_model_and_unembed, extract_h_at_layer, greedy_generate, get_first_answer_token_id,
)


# ── 秩与候选 token ────────────────────────────────────────────────────────────
def get_rank(logits_1d: torch.Tensor, token_id: int) -> int:
    """1-indexed rank of token_id in logits (descending). Mirrors validate_s14_tldc.py."""
    sorted_ids = torch.argsort(logits_1d, descending=True)
    hit = (sorted_ids == token_id).nonzero(as_tuple=True)[0]
    return int(hit.item()) + 1 if hit.numel() else -1


def alias_candidates(tokenizer, answers):
    """每个别名 → (首个非空白 token id, 是否触发空格 artifact)。

    修复点：`get_first_answer_token_id` 直接取 encode(" "+alias)[0]，对数字开头答案会取到
    空格 token（id 220，rank≡1）。这里跳过解码后 strip() 为空的 token。
    """
    out = []
    for ans in answers:
        ans_clean = str(ans).strip()
        if not ans_clean:
            continue
        ids = tokenizer.encode(" " + ans_clean, add_special_tokens=False)
        if not ids:
            continue
        artifact = False
        picked = None
        for j, tid in enumerate(ids):
            if tokenizer.decode([tid]).strip() == "":
                artifact = artifact or (j == 0)
                continue
            picked = int(tid)
            break
        if picked is not None:
            out.append({"alias": ans_clean, "tok_id": picked,
                        "tok_str": tokenizer.decode([picked]), "space_artifact": artifact})
    return out


def compute_early_exit_logits(h, ln_final, W_U, b_U):
    """ℓ* lens 早退 logits（仅诊断 rank 用；协议规则 5：lens 不作预测）。"""
    dtype = next(ln_final.parameters()).dtype
    h_norm = ln_final(h.to(dtype=dtype))
    logits = h_norm @ W_U.to(dtype)
    if b_U is not None:
        logits = logits + b_U.to(dtype)
    return logits


def assign(rank: int, is_correct: bool, thr: int) -> str:
    if rank < 0:
        return "invalid"
    if rank <= thr:
        return "know_correct" if is_correct else "know_wrong"
    return "dont_know"


def selftest(model_id: str = "Qwen/Qwen3-1.7B") -> int:
    """零 GPU 纯函数自检（tokenizer 级；跑前门禁）。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    checks = 0

    c = alias_candidates(tok, ["9", "nine"])
    assert c and c[0]["space_artifact"] is True and tok.decode([c[0]["tok_id"]]).strip() == "9", c
    checks += 1  # artifact 修复：数字开头答案取到 "9" 而非空格

    c2 = alias_candidates(tok, ["Roger Bannister", "Magma"])
    assert c2 and c2[0]["space_artifact"] is False and c2[0]["tok_str"].strip() == "Roger"
    checks += 1  # 常规别名与旧口径一致

    c3 = alias_candidates(tok, [" ", "  ", "magma"])
    assert len(c3) == 1 and c3[0]["alias"] == "magma"
    checks += 1  # 纯空白别名跳过

    lg = torch.tensor([0.1, 5.0, 3.0, 2.0])
    assert [get_rank(lg, i) for i in (1, 2, 3, 0)] == [1, 2, 3, 4]
    checks += 1  # rank 1-indexed 降序

    assert assign(50, False, 50) == "know_wrong" and assign(51, True, 50) == "dont_know" \
        and assign(1, True, 50) == "know_correct" and assign(-1, True, 50) == "invalid"
    checks += 1  # 子集划分边界

    print(f"selftest PASS {checks}/{checks}")
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--seed_test", type=int, default=123)
    ap.add_argument("--rank_threshold", type=int, default=50)
    ap.add_argument("--layer_early", type=int, default=20)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--selftest", action="store_true", help="只跑零 GPU 纯函数自检后退出")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.model)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir) if args.output_dir else (_sys_parent / "outputs" / "rank_audit_proxy")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]

    print("=" * 70)
    print(f"秩代理口径复算 | model={args.model} seed={args.seed_test} n={args.n_test} thr={args.rank_threshold}")
    print("=" * 70)

    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device, args.model)
    samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)[: args.n_test]

    entries = []
    t0 = time.time()
    for i, s in enumerate(tqdm(samples, desc="recompute")):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        h_L, logits, _, _ = extract_h_at_layer(model, tokenizer, prompt, device, args.layer_early)
        lf = logits[0, -1, :]
        lens_lf = compute_early_exit_logits(h_L, ln_final, W_U, b_U)[0, -1, :]

        cands = alias_candidates(tokenizer, s["answers"])
        if not cands:
            continue
        for c in cands:
            c["rank_final"] = get_rank(lf, c["tok_id"])
            c["rank_lens"] = get_rank(lens_lf, c["tok_id"])

        old_id = get_first_answer_token_id(tokenizer, s["answers"])  # 原口径（含 artifact）
        rank_old = get_rank(lf, old_id) if old_id is not None else -1
        rank_old_lens = get_rank(lens_lf, old_id) if old_id is not None else -1

        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct_exact(gen_text, s["answers"])

        r_final = [c["rank_final"] for c in cands]
        r_lens = [c["rank_lens"] for c in cands]
        entries.append({
            "sample_id": i,
            "question": s["question"],
            "answers": s["answers"],
            "is_correct": bool(is_correct),
            "n_aliases": len(cands),
            "candidates": cands,
            "old": {"tok_id": old_id, "rank": rank_old, "rank_lens": rank_old_lens,
                    "subset": assign(rank_old, is_correct, args.rank_threshold)},
            "fixed_first": {"tok_id": cands[0]["tok_id"], "rank": cands[0]["rank_final"],
                            "rank_lens": cands[0]["rank_lens"],
                            "space_artifact": cands[0]["space_artifact"],
                            "subset": assign(cands[0]["rank_final"], is_correct, args.rank_threshold)},
            "min_alias": {"rank": min(r_final), "rank_lens": min(r_lens),
                          "subset": assign(min(r_final), is_correct, args.rank_threshold)},
            "max_alias": {"rank": max(r_final)},
            "generated": gen_text[:80],
        })
    print(f"  done in {(time.time() - t0) / 60:.1f} min")

    # ── 汇总 ──────────────────────────────────────────────────────────────
    def counts(key):
        c = {"know_correct": 0, "know_wrong": 0, "dont_know": 0, "invalid": 0}
        for e in entries:
            c[e[key]["subset"]] += 1
        return c

    def subset_of(e, key):
        return e[key]["subset"]

    summary = {
        "model": args.model, "seed_test": args.seed_test, "n": len(entries),
        "rank_threshold": args.rank_threshold, "layer_early": args.layer_early,
        "counts": {k: counts(k) for k in ("old", "fixed_first", "min_alias")},
        "space_artifact_samples": sum(1 for e in entries if e["old"]["tok_id"] is not None
                                      and e["candidates"] and e["candidates"][0]["space_artifact"]),
        "n_alias_unstable_subset": sum(1 for e in entries
                                       if len({c["rank_final"] <= args.rank_threshold for c in e["candidates"]}) > 1),
        "subset_agreement": {
            "old_vs_fixed_first": sum(1 for e in entries if subset_of(e, "old") == subset_of(e, "fixed_first")),
            "old_vs_min_alias": sum(1 for e in entries if subset_of(e, "old") == subset_of(e, "min_alias")),
            "fixed_first_vs_min_alias": sum(1 for e in entries
                                            if subset_of(e, "fixed_first") == subset_of(e, "min_alias")),
        },
        # B4：final 与 ℓ* lens 是否落在阈值同侧（沿用旧口径 token，保持与已发布档案可比）
        "b4_final_vs_lens_same_side": {
            "old_token": sum(1 for e in entries
                             if e["old"]["rank"] > 0 and e["old"]["rank_lens"] > 0
                             and (e["old"]["rank"] <= args.rank_threshold)
                             == (e["old"]["rank_lens"] <= args.rank_threshold)),
            "n_valid": sum(1 for e in entries if e["old"]["rank"] > 0 and e["old"]["rank_lens"] > 0),
        },
    }

    with open(out_dir / f"rank_audit_seed{args.seed_test}_{tag}.json", "w") as f:
        json.dump({"summary": summary, "entries": entries}, f, ensure_ascii=False, indent=1)

    # ── markdown ─────────────────────────────────────────────────────────
    md = [f"# 秩代理口径复算 — {tag} seed{args.seed_test}（n={len(entries)}）", "",
          f"- ℓ* = L{args.layer_early}；阈值 rank ≤ {args.rank_threshold}；exact 标签",
          f"- 空格 artifact 样本（旧口径 token 为空白）：**{summary['space_artifact_samples']}**",
          f"- 别名改变 know 判定（rank≤thr 的真值随别名变化）的样本：**{summary['n_alias_unstable_subset']}**", "",
          "| 口径 | KC | KW | DK | invalid |", "|---|---|---|---|---|"]
    for k, v in summary["counts"].items():
        md.append(f"| {k} | {v['know_correct']} | {v['know_wrong']} | {v['dont_know']} | {v['invalid']} |")
    md += ["", "子集一致率（同判）：", ""]
    for k, v in summary["subset_agreement"].items():
        md.append(f"- {k}: {v}/{len(entries)} = {v/len(entries)*100:.1f}%")
    b4 = summary["b4_final_vs_lens_same_side"]
    md += ["", f"B4（final vs ℓ* lens 同侧）：{b4['old_token']}/{b4['n_valid']} = "
               f"{b4['old_token']/max(1,b4['n_valid'])*100:.1f}%"]
    (out_dir / f"summary_seed{args.seed_test}_{tag}.md").write_text("\n".join(md) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\n[out] {out_dir}/rank_audit_seed{args.seed_test}_{tag}.json")


if __name__ == "__main__":
    main()
