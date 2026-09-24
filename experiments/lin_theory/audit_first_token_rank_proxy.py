"""首 token 秩代理保真度审计（零 GPU；L7 地基待办）。

理论：docs/theory/theory-intervention-failure.md §1.2.1
  - 知识划分口径：know = rank_final(首 token) <= 50；KW = know ∧ 答错，KC = know ∧ 答对，DK = ¬know
  - 代理误差：假知道率 alpha = P(K_hat=1 | K*=0)；机制假说 = 功能词首 token 先验 P(t) 大
    ⇒ rank 无条件靠前 ⇒ 假知道
  - 可检验预测：P1 功能词首 token 的 K_hat=1 比例更高；P2 与序列 logprob 划分不一致集中于功能词；
    P3 观测知道率被高估

**预注册判据（2026-09-24 新拟，无历史依据；跑前写下，跑后不得改）**
  主判量 M1 = KW 子集中"首 token 为功能词"的样本占比（含 CP95 Wilson 区间）
  辅判量 M2 = DK 子集内 功能词组 vs 内容词组 的 rank 分布差异（单侧 Mann-Whitney，功能词更低？）
  判"实质污染"：M1 点估计 >= 10%  或  (M2 单侧 p < 0.05 且 rank 中位差 >= 10)
  判"近似保真"：M1 <= 5% 且 M2 不显著
  其余＝灰区 ⇒ 建议补一步轻 GPU 的"无知识先验 baseline"（同一首 token 在无上下文分布中的 rank）

不自称：本审计只测**标签口径质量**，不是重开检测线刷 AUROC（D04 红线）；结论不得外推为
"检测可救活"。零 GPU：只用已落盘档案 + tokenizer（Qwen3 家族共用词表）。

数据源
  - 8B（主）：experiments/outputs/phase4_subspace_review/subspace_review_s{123,456}_samples.json
    （含 answers / rank_final / subset / baseline_correct；n=300×2）
  - 8B（口径交叉核对）：experiments/outputs/lin_theory_8b/seed{123,456}_8b/s14_tldc_samples.json
  - 干预结果（M4 重算）：experiments/outputs/tldc_controls_8b/tldc_controls_{123,456}_*.json
  - 1.7B（次，跨规模）：experiments/outputs/lin_theory/seed*/s14_tldc_samples.json
    （无 answers ⇒ 用 datasets 缓存 load_triviaqa(300, seed) 建 question->answers 映射）

输出：experiments/outputs/label_audit_first_token/{audit.json,audit_report.md}

用法：
    ~/miniconda3/envs/pytorch_env0/bin/python experiments/lin_theory/audit_first_token_rank_proxy.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments" / "outputs" / "label_audit_first_token"

TOKENIZER_DIR_CANDIDATES = [
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots",
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots",
]

# ── 功能词表（英文；curated，跑前固定） ────────────────────────────────────────
FUNCTION_WORDS = {
    # articles / determiners
    "the", "a", "an", "this", "that", "these", "those", "some", "any", "each",
    "every", "no", "all", "both", "few", "many", "much", "more", "most",
    "other", "another", "such", "either", "neither",
    # prepositions
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "without",
    "into", "onto", "over", "under", "above", "below", "between", "among",
    "during", "after", "before", "since", "until", "till", "through",
    "throughout", "against", "about", "around", "across", "along", "behind",
    "beside", "beyond", "near", "off", "out", "up", "down", "upon", "within",
    "via", "per", "than", "as",
    # conjunctions
    "and", "or", "but", "nor", "so", "yet", "plus", "because", "although",
    "though", "while", "whereas", "if", "unless", "whether",
    # pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "mine", "yours",
    "hers", "ours", "theirs", "who", "whom", "whose", "which", "what",
    "where", "when", "why", "how",
    # auxiliaries / copulas
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "have", "has", "had", "having", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "ought",
    # foreign articles (proper names: "The" already covered; Dutch/German/Romance)
    "van", "von", "der", "die", "das", "la", "le", "les", "el", "los", "las",
    "de", "du", "del",
}

NUMBER_RE = re.compile(r"^[0-9]+([.,][0-9]+)*$")
ROMAN_RE = re.compile(r"^[IVXLCDM]+$")


def load_tokenizer():
    from transformers import AutoTokenizer

    for snaps in TOKENIZER_DIR_CANDIDATES:
        if not snaps.exists():
            continue
        for snap in sorted(snaps.iterdir()):
            if (snap / "tokenizer.json").exists():
                tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
                return tok, str(snap)
    raise FileNotFoundError("no local Qwen3 tokenizer found")


def first_answer_token_id(tokenizer, answers):
    """镜像 experiments/lin_theory/common.py::get_first_answer_token_id（口径必须一致）。"""
    for ans in answers:
        ans_clean = str(ans).strip()
        if not ans_clean:
            continue
        toks = tokenizer.encode(" " + ans_clean, add_special_tokens=False)
        if toks:
            return int(toks[0]), ans_clean
    return None, None


def classify(tok_str: str) -> str:
    w = tok_str.strip()
    if not w:
        # 空白 token：BPE 把 " 9" 切成 [' ', '9'] ⇒ 取到的"首 token"是空格本身（id 220），
        # 其 rank ≡ 1 ⇒ 该样本被无条件判为 know。**tokenization artifact**，非知识信号。
        return "artifact"
    if NUMBER_RE.match(w):
        return "number"
    if w.lower() in FUNCTION_WORDS:
        return "function"
    if ROMAN_RE.match(w) and len(w) <= 4:
        return "number"
    return "content"


def decode_token(tokenizer, tid: int) -> str:
    return tokenizer.decode([tid])


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def load_json(p: Path):
    with open(p) as f:
        return json.load(f)


def _s14_path(model: str, seed: int):
    """定位该模型/seed 的 s14 逐样本档案（已发布标签口径的唯一事实源）。"""
    if model == "8B":
        cands = [REPO / f"experiments/outputs/lin_theory_8b/seed{seed}_8b/s14_tldc_samples.json"]
    else:
        cands = [REPO / f"experiments/outputs/lin_theory/seed{seed}/s14_tldc_samples.json",
                 REPO / "experiments/outputs/lin_theory/s14_tldc_samples.json"]
    for p in cands:
        if not p.exists():
            continue
        d = load_json(p)
        cfg = d.get("config", {})
        if int(cfg.get("seed_test", -1)) == seed and int(cfg.get("n_test", 0)) == 300:
            return p, d
    return None, None


def build_samples(model: str):
    """标签取该模型自己的 s14 档案（已发布口径）；answers 借 1.7B subspace 档案同 seed 同题的金标元数据。"""
    recs, checks = [], []
    for seed in (123, 456):
        path, arch = _s14_path(model, seed)
        if arch is None:
            print(f"[warn] {model} seed{seed}: 无 s14 档案，跳过")
            continue
        sub_path = REPO / f"experiments/outputs/phase4_subspace_review/subspace_review_s{seed}_samples.json"
        gold = {s["question"]: s["answers"] for s in load_json(sub_path)["samples"].values()}
        n_join = 0
        for e in arch["samples"].values():
            ans = gold.get(e["question"])
            if ans is None:
                continue
            n_join += 1
            recs.append({"seed": seed, "model": model, "q": e["question"], "answers": ans,
                         "rank": int(e["rank"]), "subset": e["subset"],
                         "baseline_correct": bool(e["baseline_correct"])})
        checks.append({"model": model, "seed": seed, "path": str(path.relative_to(REPO)),
                       "n_archive": len(arch["samples"]), "n_joined": n_join,
                       "subset_counts": {k: sum(1 for v in arch["samples"].values() if v["subset"] == k)
                                         for k in ("know_correct", "know_wrong", "dont_know")}})
    return recs, checks


def load_tldc_8b_outcomes():
    """question -> {baseline_correct, correct_real_beta0.2}（real 臂）。"""
    out = {}
    for seed, fn in ((123, "tldc_controls_123_real-shuffle-gauss-anti-wrong_late-wrong_zero.json"),
                     (456, "tldc_controls_456_real-shuffle.json")):
        d = load_json(REPO / "experiments/outputs/tldc_controls_8b" / fn)
        for s in d["samples"].values():
            key = (seed, s["question"])
            out[key] = {
                "baseline_correct": bool(s["baseline_correct"]),
                "correct_real": bool(s.get("correct_real_beta0.2")),
                "correct_shuffle": bool(s.get("correct_shuffle_beta0.2")),
            }
    return out


def main():
    from scipy.stats import mannwhitneyu

    tokenizer, tok_path = load_tokenizer()
    print(f"[tok] {tok_path} vocab={tokenizer.vocab_size}")

    recs, checks = build_samples("8B")
    recs_1p7, checks_1p7 = build_samples("1.7B")
    outcomes = load_tldc_8b_outcomes()

    # ── 逐样本首 token 分类 ────────────────────────────────────────────────
    for r in recs + recs_1p7:
        tid, src = first_answer_token_id(tokenizer, r["answers"])
        r["first_tid"] = tid
        r["first_tok"] = decode_token(tokenizer, tid) if tid is not None else None
        r["first_src"] = src
        r["klass"] = classify(r["first_tok"]) if r["first_tok"] is not None else "none"
        # 别名敏感性：各别名首 token 是否同类
        classes = []
        for a in r["answers"]:
            t, _ = first_answer_token_id(tokenizer, [a])
            if t is not None:
                classes.append(classify(decode_token(tokenizer, t)))
        r["alias_klasses"] = classes
        r["alias_unstable"] = len(set(classes)) > 1

    report = {"tokenizer": tok_path, "vocab": tokenizer.vocab_size,
              "crosscheck_8b": checks, "crosscheck_1p7b": checks_1p7,
              "prereg": {
                  "M1_contaminated_if_ge": 0.10, "M1_clean_if_le": 0.05,
                  "M2_median_rank_gap": 10, "M2_alpha": 0.05,
              }}

    for tag, data in (("8B", recs), ("1.7B", recs_1p7)):
        if not data:
            continue
        R = np.array([r["rank"] for r in data])
        K = np.array([r["klass"] for r in data])
        S = np.array([r["subset"] for r in data])
        block = {"n": len(data)}

        # M0: 首 token 类别构成
        block["klass_counts"] = {str(c): int((K == c).sum()) for c in sorted(set(K))}

        # M1: 各子集内 可疑首 token（功能词 / 空白 artifact）占比
        m1 = {}
        for sub in ("know_correct", "know_wrong", "dont_know"):
            m = S == sub
            n = int(m.sum())
            k_f = int((K[m] == "function").sum())
            k_a = int((K[m] == "artifact").sum())
            p, lo, hi = wilson(k_f, n)
            ps, los, his = wilson(k_f + k_a, n)
            m1[sub] = {"n": n, "n_function": k_f, "share": p, "cp95": [lo, hi],
                       "n_artifact": k_a, "suspect_share": ps, "suspect_cp95": [los, his]}
        block["M1_function_share_by_subset"] = m1

        # M1b: 可疑占比 KW vs DK 的 Fisher 单侧检验（P1 的统计版）
        kw_m, dk_m = S == "know_wrong", S == "dont_know"
        sus = (K == "function") | (K == "artifact")
        tbl = [[int((sus & kw_m).sum()), int((~sus & kw_m).sum())],
               [int((sus & dk_m).sum()), int((~sus & dk_m).sum())]]
        try:
            from scipy.stats import fisher_exact
            block["M1b_fisher_KW_vs_DK"] = {"table": tbl,
                                            "p_one_sided_greater": float(fisher_exact(tbl, alternative="greater")[1]),
                                            "odds_ratio": float(fisher_exact(tbl)[0])}
        except Exception:
            pass

        # M2: DK 内 功能词 vs 内容词 rank 分布
        dk = S == "dont_know"
        a = R[dk & (K == "function")]
        b = R[dk & (K == "content")]
        m2 = {"n_function": int(a.size), "n_content": int(b.size)}
        if a.size >= 5 and b.size >= 5:
            u, pval = mannwhitneyu(a, b, alternative="less")
            m2.update({"median_function": float(np.median(a)), "median_content": float(np.median(b)),
                       "median_gap": float(np.median(b) - np.median(a)),
                       "U": float(u), "p_one_sided_less": float(pval)})
        block["M2_DK_rank"] = m2

        # M-extra: rank 分箱 × 类别（阈值 50 是否切在先验驱动区）
        bins = [0, 10, 50, 100, 500, 10**9]
        tab = {}
        for c in ("function", "content", "number"):
            m = K == c
            tab[c] = [int(((R[m] > bins[i]) & (R[m] <= bins[i + 1])).sum()) for i in range(len(bins) - 1)]
        block["rank_bins"] = {"edges": bins, "counts": tab}

        # M3/M4: KW 清洗前后 + TLDC 救回率（仅 8B 有干预结果）
        kw = S == "know_wrong"
        kw_sus = kw & ((K == "function") | (K == "artifact"))
        block["M3_kw"] = {"n_kw": int(kw.sum()), "n_kw_function_first": int((kw & (K == "function")).sum()),
                          "n_kw_artifact": int((kw & (K == "artifact")).sum()),
                          "n_kw_suspect": int(kw_sus.sum()),
                          "share_suspect": float(kw_sus.sum() / max(1, kw.sum())),
                          "suspect_cp95": [float(x) for x in wilson(int(kw_sus.sum()), int(kw.sum()))[1:]],
                          "klass_in_kw": {str(c): int(((K == c) & kw).sum()) for c in sorted(set(K))}}
        if tag == "8B":
            hits = {"n_kw_clean": 0, "rescue_clean": 0, "n_kw_all": 0, "rescue_all": 0,
                    "n_kw_fun": 0, "rescue_fun": 0, "n_kw_art": 0, "rescue_art": 0,
                    "mismatch_outcome_join": 0, "per_seed": {}}
            for r in data:
                o = outcomes.get((r["seed"], r["q"]))
                if o is None:
                    hits["mismatch_outcome_join"] += 1
                    continue
                if r["subset"] != "know_wrong":
                    continue
                rescued = (not o["baseline_correct"]) and o["correct_real"]
                hits["n_kw_all"] += 1
                hits["rescue_all"] += int(rescued)
                ps = hits["per_seed"].setdefault(r["seed"], {"n": 0, "rescue": 0, "n_clean": 0, "rescue_clean": 0})
                ps["n"] += 1
                ps["rescue"] += int(rescued)
                if r["klass"] == "function":
                    hits["n_kw_fun"] += 1
                    hits["rescue_fun"] += int(rescued)
                elif r["klass"] == "artifact":
                    hits["n_kw_art"] += 1
                    hits["rescue_art"] += int(rescued)
                else:
                    hits["n_kw_clean"] += 1
                    hits["rescue_clean"] += int(rescued)
                    ps["n_clean"] += 1
                    ps["rescue_clean"] += int(rescued)
            for k in ("all", "clean"):
                if hits[f"n_kw_{k}"] if k == "clean" else hits["n_kw_all"]:
                    hits[f"rate_{k}"] = (hits[f"rescue_{k}"] if k == "clean" else hits["rescue_all"]) / \
                                        (hits[f"n_kw_{k}"] if k == "clean" else hits["n_kw_all"])
            for seed, ps in list(hits["per_seed"].items()):
                if ps["n"]:
                    ps["rate"] = ps["rescue"] / ps["n"]
                if ps["n_clean"]:
                    ps["rate_clean"] = ps["rescue_clean"] / ps["n_clean"]
            # 可疑组 vs 干净组 救回率 Fisher
            try:
                from scipy.stats import fisher_exact
                hits["fisher_suspect_vs_clean_rescue"] = {
                    "table": [[hits["rescue_fun"] + hits["rescue_art"],
                               hits["n_kw_fun"] + hits["n_kw_art"] - hits["rescue_fun"] - hits["rescue_art"]],
                              [hits["rescue_clean"], hits["n_kw_clean"] - hits["rescue_clean"]]],
                    "p_two_sided": float(fisher_exact(
                        [[hits["rescue_fun"] + hits["rescue_art"],
                          hits["n_kw_fun"] + hits["n_kw_art"] - hits["rescue_fun"] - hits["rescue_art"]],
                         [hits["rescue_clean"], hits["n_kw_clean"] - hits["rescue_clean"]]])[1])}
            except Exception:
                pass
            block["M4_kw_clean_recompute"] = hits

        # M5: 别名敏感性
        unstable = np.array([r["alias_unstable"] for r in data])
        block["M5_alias"] = {"n_alias_unstable": int(unstable.sum()),
                             "share": float(unstable.mean())}

        report[tag] = block

    # ── 判读 ─────────────────────────────────────────────────────────────
    verdicts = {}
    for tag in ("8B", "1.7B"):
        if tag not in report:
            continue
        m1 = report[tag]["M1_function_share_by_subset"]["know_wrong"]
        m2 = report[tag]["M2_DK_rank"]
        cond_a = m1["share"] >= report["prereg"]["M1_contaminated_if_ge"]
        cond_b = (m2.get("p_one_sided_less", 1.0) < report["prereg"]["M2_alpha"]
                  and m2.get("median_gap", 0.0) >= report["prereg"]["M2_median_rank_gap"])
        clean = (m1["share"] <= report["prereg"]["M1_clean_if_le"]
                 and m2.get("p_one_sided_less", 1.0) >= report["prereg"]["M2_alpha"])
        verdicts[tag] = ("实质污染" if (cond_a or cond_b) else ("近似保真" if clean else "灰区"))
    report["verdict"] = verdicts

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "audit.json", "w") as f:
        json.dump({"report": report,
                   "samples": [{k: v for k, v in r.items() if k != "alias_klasses"} for r in recs + recs_1p7]},
                  f, ensure_ascii=False, indent=1)

    # ── markdown 报告 ────────────────────────────────────────────────────
    md = ["# 首 token 秩代理保真度审计（零 GPU）", "",
          f"- 运行：`experiments/lin_theory/audit_first_token_rank_proxy.py`；tokenizer=`{tok_path}`（vocab {tokenizer.vocab_size}）",
          "- 标签来源：各模型自己的 `s14_tldc_samples.json`（已发布口径，KW 计数与论文逐位一致）",
          "- 金标别名：借 `phase4_subspace_review`（1.7B 跑）同 seed/同题的 `answers` 字段",
          "- 理论：`docs/theory/theory-intervention-failure.md` §1.2.1；判据见脚本头部（跑前写死）", ""]
    for tag in ("8B", "1.7B"):
        if tag not in report:
            continue
        b = report[tag]
        md += [f"## {tag}（n={b['n']}）", "",
               f"首 token 类别构成：{b['klass_counts']}", "",
               "| 子集 | n | 功能词首 token | 占比 (CP95) | 空格 artifact | **可疑合计** (CP95) |",
               "|---|---|---|---|---|---|"]
        for sub, v in b["M1_function_share_by_subset"].items():
            md.append(f"| {sub} | {v['n']} | {v['n_function']} | {v['share']*100:.1f}% "
                      f"({v['cp95'][0]*100:.1f}–{v['cp95'][1]*100:.1f}) | {v['n_artifact']} | "
                      f"**{v['suspect_share']*100:.1f}%** "
                      f"({v['suspect_cp95'][0]*100:.1f}–{v['suspect_cp95'][1]*100:.1f}) |")
        f1 = b.get("M1b_fisher_KW_vs_DK", {})
        md += ["", f"- P1 统计：KW vs DK 可疑占比 Fisher 单侧 p = **{f1.get('p_one_sided_greater', float('nan')):.2e}**，"
                   f"OR = {f1.get('odds_ratio', float('nan')):.2f}",
               f"- rank 分箱（function/content/number）：{b['rank_bins']['counts']}（edges {b['rank_bins']['edges']}）",
               f"- M2（DK 内 功能词 vs 内容词 rank）：{b['M2_DK_rank']}",
               f"- M3（KW 污染）：{b['M3_kw']}",
               f"- M5（别名敏感性）：{b['M5_alias']}"]
        if b.get("M4_kw_clean_recompute"):
            h = b["M4_kw_clean_recompute"]
            md += ["", "**M4 KW 清洗重算（TLDC real 臂 β=0.20）**", "",
                   f"- 全部 KW：{h['rescue_all']}/{h['n_kw_all']} = {h.get('rate_all', float('nan'))*100:.1f}%",
                   f"- 剔除可疑后 KW：{h['rescue_clean']}/{h['n_kw_clean']} = {h.get('rate_clean', float('nan'))*100:.1f}%",
                   f"- 功能词组：{h['rescue_fun']}/{h['n_kw_fun']}｜空格 artifact 组：{h['rescue_art']}/{h['n_kw_art']}",
                   f"- 分 seed：{h['per_seed']}",
                   f"- 可疑组 vs 干净组救回率 Fisher p = {h.get('fisher_suspect_vs_clean_rescue', {}).get('p_two_sided', float('nan')):.3f}"]
        md += ["", f"判读（预注册规则）：**{report['verdict'][tag]}**", ""]
    md += ["## 与已发布数字的关系", "",
           "- 标签计数与论文逐位一致（8B KW 58/64、KC 122/115；1.7B KW 71/64、KC 76/75）⇒ 本审计未改口径，只是量化其误差。",
           "- 空格 artifact 为**本轮新发现**（预注册规则未覆盖）⇒ 阈值不追溯套用，仅作并列披露。", ""]
    (OUT_DIR / "audit_report.md").write_text("\n".join(md))

    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\n[out] {OUT_DIR/'audit.json'}  {OUT_DIR/'audit_report.md'}")


if __name__ == "__main__":
    main()
