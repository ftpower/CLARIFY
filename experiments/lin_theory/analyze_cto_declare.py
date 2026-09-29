"""方向三·CTO 校准三元输出 S0 预判读（零 GPU）。

理论锚点与判据：docs/protocol/cto-calibrated-ternary-output-20260929.md §5
  - S0-A（正确性三元可行性，代理）：8B oof 探针分 scores_peak + 固定阈值 0.7/0.3 →
    知道/存疑/不知道；要求 P(答对|知道) ≥0.80 ∧ P(答错|不知道) ≥0.80 ∧ 覆盖 ≥30%。
  - S0-B（token 可行性）：候选词表在 1.7B/8B tokenizer 上单 token 编码 + decode 往返一致。
  - S0-C（对照，只报不判）：知道档保留准确率 vs 报告图 5.5 工作点曲线同覆盖率读数。
"""
import argparse
import json
from pathlib import Path

OOF_DEFAULT = "experiments/outputs/lin_theory_8b/detect_lr_probe_oof.json"
OUT_DEFAULT = "experiments/outputs/cto_declare"
T_HI, T_LO = 0.7, 0.3              # 固定声明阈值（执行前设定，不随数据调）
PREC_KNOW_MIN = 0.80               # 知道档答对精度线
PREC_UNK_MIN = 0.80                # 不知道档答错精度线
COVERAGE_MIN = 0.30                # 极端两档覆盖线
CANDIDATES = ["Know", "Unknown", "Unsure", "知道", "不知道", "存疑"]


def declare(scores, y):
    """固定阈值三元声明 → (知道, 存疑, 不知道) 读数 dict。"""
    know = [y[i] for i, s in enumerate(scores) if s >= T_HI]
    unsure = [y[i] for i, s in enumerate(scores) if T_LO < s < T_HI]
    unk = [y[i] for i, s in enumerate(scores) if s <= T_LO]
    n = len(scores)
    p_know = sum(know) / len(know) if know else float("nan")
    p_unk = 1 - sum(unk) / len(unk) if unk else float("nan")  # P(答错|不知道)
    cov = (len(know) + len(unk)) / n
    return dict(n=n, n_know=len(know), n_unsure=len(unsure), n_unk=len(unk),
                p_know=p_know, p_unk=p_unk, coverage=cov)


def verdict_a(res):
    if not (res["p_know"] >= PREC_KNOW_MIN and res["p_unk"] >= PREC_UNK_MIN):
        return False, (f"精度线未过：P(答对|知道)={res['p_know']:.3f}（≥{PREC_KNOW_MIN}）、"
                       f"P(答错|不知道)={res['p_unk']:.3f}（≥{PREC_UNK_MIN}）")
    if res["coverage"] < COVERAGE_MIN:
        return False, f"覆盖 {res['coverage']:.3f} < {COVERAGE_MIN}"
    return True, (f"P(答对|知道)={res['p_know']:.3f} ∧ P(答错|不知道)={res['p_unk']:.3f} "
                  f"∧ 覆盖 {res['coverage']:.3f} ⇒ 进 S1")


def check_tokens(models):
    """S0-B：候选词表 token 可行性。返回 {word: {model: (ok, note)}}。"""
    from transformers import AutoTokenizer
    out = {}
    for word in CANDIDATES:
        out[word] = {}
        for m in models:
            try:
                tok = AutoTokenizer.from_pretrained(m, local_files_only=True)
                ids = tok.encode(word, add_special_tokens=False)
                ok = len(ids) == 1 and tok.decode(ids) == word
                out[word][m] = (bool(ok), ids, tok.decode(ids))
            except Exception as e:
                out[word][m] = (False, [], f"tokenizer 不可用：{type(e).__name__}")
    return out


def selftest():
    ok = 0
    # 声明函数：构造 10 样本——4 高分全对、3 低分全错、3 中间 2 对 1 错
    scores = [0.9, 0.85, 0.8, 0.75, 0.5, 0.45, 0.4, 0.25, 0.2, 0.15]
    y =      [1,   1,    1,   1,    1,   1,    0,   0,    0,   0]
    r = declare(scores, y)
    assert r["n_know"] == 4 and r["n_unk"] == 3 and r["n_unsure"] == 3
    assert abs(r["p_know"] - 1.0) < 1e-9 and abs(r["p_unk"] - 1.0) < 1e-9
    assert abs(r["coverage"] - 0.7) < 1e-9
    ok += 1
    # 判定：精度不过 → False
    r2 = declare([0.9, 0.9, 0.25, 0.25], [1, 0, 1, 0])  # 知道档 1/2、不知道档 1/2
    assert verdict_a(r2)[0] is False
    # 覆盖不过 → False
    r3 = declare([0.9, 0.5, 0.5, 0.5], [1, 1, 1, 1])
    assert verdict_a(r3)[0] is False
    ok += 1
    print(f"SELFTEST PASS: {ok}/2 checks (declare / verdict-a)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=OOF_DEFAULT)
    ap.add_argument("--models", nargs="+",
                    default=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"])
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    d = json.loads(Path(args.oof).read_text())
    scores, y = d["scores_peak"], d["y"]
    res = declare(scores, y)
    a_ok, a_why = verdict_a(res)

    tok_res = check_tokens(args.models)
    b_words = [w for w, mm in tok_res.items()
               if all(v[0] for v in mm.values())]
    b_ok = len(b_words) >= 3

    lines = ["# CTO S0 预判读（零 GPU）", "",
             f"- 判据：`docs/protocol/cto-calibrated-ternary-output-20260929.md` §5"
             f"（A：精度 ≥{PREC_KNOW_MIN}/{PREC_UNK_MIN} ∧ 覆盖 ≥{COVERAGE_MIN}；B：声明词单 token）",
             f"- 资产：`{args.oof}`（n={res['n']}，答对 {sum(y)}）；固定阈值 {T_HI}/{T_LO}", "",
             "## S0-A 正确性三元（代理）", "",
             f"- 知道 {res['n_know']}｜存疑 {res['n_unsure']}｜不知道 {res['n_unk']}｜"
             f"覆盖 **{res['coverage']:.3f}**",
             f"- P(答对|知道) **{res['p_know']:.3f}**｜P(答错|不知道) **{res['p_unk']:.3f}**",
             f"- **A 判定：{'✅ 过' if a_ok else '❌ 不过'}** — {a_why}", "",
             "## S0-B token 可行性", ""]
    for w, mm in tok_res.items():
        flags = " ".join(f"{m.split('/')[-1]}:{'✅' if v[0] else '❌'}"
                         for m, v in mm.items())
        detail = " ".join(f"{m.split('/')[-1]}={v[1]}" for m, v in mm.items()
                          if isinstance(v[1], list) and v[1])
        lines.append(f"- `{w}` — {flags} {detail}")
    lines += ["", f"- 主档声明词表（全模型单 token）：**{b_words or '不足三词'}**",
              f"- **B 判定：{'✅ 过' if b_ok else '❌ 不过（换候选词表重验，§5）'}**", "",
              "## S0-C 对照（只报不判）", "",
              f"- 「知道」档保留准确率 **{res['p_know']:.3f}** @ 覆盖 {res['n_know'] / res['n']:.2f}"
              f"（报告图 5.5：弃权 50% 保留准确率 0.870，覆盖口径不同，仅并列）", "",
              "## 判定", ""]
    if a_ok and b_ok:
        final = "enter"
        why = "A∧B 过 ⇒ 进 S1（真机重打知识标签 + 声明写入小样本试运行）"
    elif not a_ok:
        final = "close"
        why = "A 不过（正确性三元精度线不达标 ⇒ 知识三元无望）"
    else:
        final = "instrument"
        why = "B 不过 ⇒ 换候选词表重验一次（登记 §7），仍不过则关闭"
    lines += [f"- **{final}** — {why}", ""]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "cto_declare_report.md").write_text("\n".join(lines))
    with open(out_dir / "cto_declare.json", "w") as f:
        json.dump({"a": res, "a_ok": a_ok, "b_words": b_words, "final": final},
                  f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_dir / 'cto_declare_report.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
