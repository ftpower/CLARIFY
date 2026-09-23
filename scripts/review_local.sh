#!/usr/bin/env bash
# 复核实验一键脚本（2026-09-21）——避免长命令粘贴时反斜杠被空格打断。
#
# 用法：
#   bash scripts/review_local.sh smoke-rome        # ROME 冒烟（小样本，先跑这个）
#   bash scripts/review_local.sh rome-poscontrol   # ROME λ 敏感性/阳性对照（网格 0/5/20/100）
#   bash scripts/review_local.sh rome              # ROME 正式（seed 123）
#   bash scripts/review_local.sh rome-456          # ROME 正式（seed 456）
#   bash scripts/review_local.sh subspace          # 子空间正式（seed 123）
#   bash scripts/review_local.sh subspace-456      # 子空间正式（seed 456，复用缓存）
#   bash scripts/review_local.sh subspace-fast     # 快档（只扫 subtract；SEED/LAM 可覆盖，需已有缓存）
#   bash scripts/review_local.sh dola-baseline     # DoLa 1.7B baseline
#   bash scripts/review_local.sh dola-static       # DoLa 1.7B 静态档
#   bash scripts/review_local.sh dola-dynamic      # DoLa 1.7B 动态档
#   bash scripts/review_local.sh dola-8b-baseline  # DoLa 8B baseline（服务器用）
#   bash scripts/review_local.sh beta-sweep        # Phase 24 β sweep（服务器用，改 BETA=...）
#   bash scripts/review_local.sh geometry          # FAD/S5 几何档案：baseline 轨迹（seed 123）
#   bash scripts/review_local.sh geometry-456      # 同上（seed 456）
#   bash scripts/review_local.sh geometry-lift     # 阶段2 单抬支轨迹（BETA=0.05 默认）
#   bash scripts/review_local.sh geometry-damp     # 阶段2 单压支轨迹
#   bash scripts/review_local.sh geometry-sym      # 阶段2 对称 TLDC 轨迹（干预后档案，SYM_BETA 默认 0.20）
#   bash scripts/review_local.sh geometry-sym-8b   # 同上 8B（服务器用；SEED/SYM_BETA 可覆盖）
#   bash scripts/review_local.sh geometry-8b-baseline  # 8B 基线轨迹（服务器用；SEED=456 补双 seed 基线档）
#   bash scripts/review_local.sh ctrl-smoke        # 零机制对照臂冒烟（n=30，先跑）
#   bash scripts/review_local.sh ctrl-tldc         # 对照臂全臂 β=0.20（seed123，1.7B）
#   bash scripts/review_local.sh ctrl-tldc-456     # 对照臂 real+shuffle（seed456）
#   bash scripts/review_local.sh ctrl-tldc-lowbeta # 次判据档 β=0.03（real+shuffle）
#   bash scripts/review_local.sh gated-tldc        # T3 门控族（TAU=0.2/0.3 可覆盖）
#   bash scripts/review_local.sh gated-tldc-456    # T3 门控第二 seed（real+gated_margin）
#   bash scripts/review_local.sh gated-tldc-8b     # T3 门控 8B（服务器用；SEED/TAU 可覆盖）
#   bash scripts/review_local.sh verified-tldc     # 验证器臂（一步前瞻，θ=0.5 预注册）
#   bash scripts/review_local.sh verified-tldc-8b  # 验证器臂 8B（服务器用；SEED/THETA 可覆盖）
#   bash scripts/review_local.sh verified-rand     # 验证器 footprint 安慰剂（同批 real+sym+rand，KEEP_P 可覆盖）
#   bash scripts/review_local.sh verified-rand-smoke # 同上小样本冒烟（n=30，先跑这个验证链路）
#   bash scripts/review_local.sh verified-rand-8b  # 同上 8B（服务器用；**必须**显式传 KEEP_P=同批标定）
#
# 说明：每条命令写成单行（`\` 续行在部分终端粘贴时会因行尾空格失效）。
# 服务器命令请自行补 `unset HF_ENDPOINT && HF_HOME=...` 前缀（见 runbook §2）。
# ⚠️ 服务器跑 8B case：**`--model` 传 repo id `Qwen/Qwen3-8B`**（各 case 默认值即可），
#    不要传快照完整路径——`src/model_loader._find_local_path` 会自行解析 HF_HOME 缓存快照，
#    而传快照路径会在 `get_official_model_name` 处 ValueError（2026-09-22 实测踩过，见 runbook §5.2b）；
#    快照路径只适用于 `train_lora_delta.py` / `analyze_tldc_per_token.py` 这类 transformers 直连脚本。

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_1P7B="${MODEL_1P7B:-Qwen/Qwen3-1.7B}"
MODEL_8B="${MODEL_8B:-Qwen/Qwen3-8B}"
ROME_LOAD="${ROME_LOAD:-experiments/phase9_multi_state/outputs_phase9/phase9_extract_compact.json}"
BETA="${BETA:-0.3}"

case "${1:-}" in
  smoke-rome)
    python experiments/phase16_untried/phase16_rome.py --load "$ROME_LOAD" --model "$MODEL_1P7B" --layers 11 --n_calibrate 20 --seed_cal 42 --n_val 20 --seed_val 789 --n_test 30 --seed_test 123 --layer_early 20 --output_dir experiments/outputs/_smoke_rome_gpu
    ;;
  rome)
    # 正式档（宽网格）：λ 有效区间由 2026-09-21 阳性对照确定——
    # λ=5 只破坏不救回、λ=20 首次出现 KW 救回(val 4/22)、λ=100 过强崩盘(All 44→10)；
    # 原 ±1/±2 网格在 val 上 7 个 λ 结果全同（编辑太弱），已废弃。
    python experiments/phase16_untried/phase16_rome.py --load "$ROME_LOAD" --model "$MODEL_1P7B" --layers 11 --n_calibrate 200 --seed_cal 42 --n_val 100 --seed_val 789 --n_test 300 --seed_test 123 --lambdas 0.0 5.0 10.0 20.0 30.0 50.0 --layer_early 20
    ;;
  rome-456)
    python experiments/phase16_untried/phase16_rome.py --load "$ROME_LOAD" --model "$MODEL_1P7B" --layers 11 --n_calibrate 200 --seed_cal 42 --n_val 100 --seed_val 789 --n_test 300 --seed_test 456 --lambdas 0.0 5.0 10.0 20.0 30.0 50.0 --layer_early 20
    ;;
  subspace)
    python experiments/phase4_generalization/main_subspace_intervention.py --n_dir 300 --n_eval 200 --model "$MODEL_1P7B" --layers 11 --k_pca 64 --lam 0.3 0.5 1.0 --seed 42 --n_test 300 --seed_test 123
    ;;
  subspace-456)
    python experiments/phase4_generalization/main_subspace_intervention.py --n_dir 300 --n_eval 200 --model "$MODEL_1P7B" --layers 11 --k_pca 64 --lam 0.3 0.5 1.0 --seed 42 --n_test 300 --seed_test 456 --skip_collect
    ;;
  subspace-fast)
    # 快档：只扫 subtract 模式（val 扫描时间减半，~20 分钟）；λ 网格用 LAM 覆盖
    python experiments/phase4_generalization/main_subspace_intervention.py --n_dir 300 --n_eval 200 --model "$MODEL_1P7B" --layers 11 --k_pca 64 --lam ${LAM:-0.5 1.0} --modes subtract --seed 42 --n_test 300 --seed_test ${SEED:-123} --skip_collect
    ;;
  rome-poscontrol)
    # 阳性对照 + λ 敏感性判定：网格放宽到 0/5/20/100。
    # 判读：val sweep 四行若开始分化 → 是网格太窄；若 λ=100 仍与 λ=0 完全相同 → 编辑未进前向（真 bug）。
    python experiments/phase16_untried/phase16_rome.py --load "$ROME_LOAD" --model "$MODEL_1P7B" --layers 11 --n_calibrate 50 --seed_cal 42 --n_val 100 --seed_val 789 --n_test 30 --seed_test 123 --lambdas 0.0 5.0 20.0 100.0 --layer_early 20 --output_dir experiments/outputs/rome_poscontrol
    ;;
  dola-baseline)
    python experiments/lin_theory/main_dola_baseline.py --model "$MODEL_1P7B" --mode baseline --layer_early 20 --n_test 300 --seed_test 123 --save_samples --output_dir experiments/outputs/dola_baseline_review_1p7b
    ;;
  dola-static)
    python experiments/lin_theory/main_dola_baseline.py --model "$MODEL_1P7B" --mode dola-static --premature_layer 7 --layer_early 20 --n_test 300 --seed_test 123 --alpha 0.1 --candidate_stride 4 --save_samples --output_dir experiments/outputs/dola_baseline_review_1p7b
    ;;
  dola-dynamic)
    python experiments/lin_theory/main_dola_baseline.py --model "$MODEL_1P7B" --mode dola-dynamic --layer_early 20 --n_test 300 --seed_test 123 --alpha 0.1 --candidate_stride 4 --save_samples --output_dir experiments/outputs/dola_baseline_review_1p7b
    ;;
  dola-8b-baseline)
    env -u HF_ENDPOINT HF_HOME="${HF_HOME_SERVER:-/root/autodl-tmp/huggingface_cache}" python -u experiments/lin_theory/main_dola_baseline.py --model "$MODEL_8B" --mode baseline --layer_early 28 --n_test 300 --seed_test 123 --max_new 20 --save_samples --output_dir experiments/outputs/dola_baseline_review
    ;;
  beta-sweep)
    env -u HF_ENDPOINT HF_HOME="${HF_HOME_SERVER:-/root/autodl-tmp/huggingface_cache}" python -u experiments/lin_theory/train_lora_delta.py --mode train --model_path "$MODEL_8B" --n_train 200 --n_test 800 --n_val 200 --epochs 1 --kc_ce_only --kl_beta "$BETA"
    ;;
  geometry)
    python experiments/lin_theory/dump_geometry_archive.py --model "$MODEL_1P7B" --layer_early 20 --n_test 300 --seed_test 123 --operator baseline --output_dir experiments/outputs/geometry_archive
    ;;
  geometry-456)
    python experiments/lin_theory/dump_geometry_archive.py --model "$MODEL_1P7B" --layer_early 20 --n_test 300 --seed_test 456 --operator baseline --output_dir experiments/outputs/geometry_archive
    ;;
  geometry-lift)
    python experiments/lin_theory/dump_geometry_archive.py --model "$MODEL_1P7B" --layer_early 20 --n_test 300 --seed_test 123 --operator lift --beta "${BETA:-0.05}" --output_dir experiments/outputs/geometry_archive
    ;;
  geometry-damp)
    python experiments/lin_theory/dump_geometry_archive.py --model "$MODEL_1P7B" --layer_early 20 --n_test 300 --seed_test 123 --operator damp --beta "${BETA:-0.05}" --output_dir experiments/outputs/geometry_archive
    ;;
  geometry-sym)
    # 阶段2 对称 TLDC 轨迹（干预后档案）；SYM_BETA 默认 0.20 = 主判据档（方向可分性检验用）
    python experiments/lin_theory/dump_geometry_archive.py --model "$MODEL_1P7B" --layer_early 20 --n_test 300 --seed_test "${SEED:-123}" --operator sym --beta "${SYM_BETA:-0.20}" --output_dir experiments/outputs/geometry_archive
    ;;
  geometry-sym-8b)
    # 阶段2 对称 TLDC 轨迹（8B，服务器用）；SEED / SYM_BETA 可覆盖
    env -u HF_ENDPOINT HF_HOME="${HF_HOME_SERVER:-/root/autodl-tmp/huggingface_cache}" python -u experiments/lin_theory/dump_geometry_archive.py --model "$MODEL_8B" --layer_early 28 --n_test 300 --seed_test "${SEED:-123}" --operator sym --beta "${SYM_BETA:-0.20}" --output_dir experiments/outputs/geometry_archive_8b_sym
    ;;
  ctrl-smoke)
    # 零机制对照臂冒烟（n=30，主判据档 β=0.20；先跑这个验证代码链路）
    python experiments/lin_theory/main_tldc_controls.py --model "$MODEL_1P7B" --layer_early 20 --n_test 30 --seed_test 123 --arms real shuffle --betas 0.2 --output_dir experiments/outputs/_smoke_tldc_ctrl
    ;;
  ctrl-tldc)
    # 本地 1.7B 全臂 @ 主判据档 β=0.20（6 臂；预估 20–40 分钟）
    python experiments/lin_theory/main_tldc_controls.py --model "$MODEL_1P7B" --layer_early 20 --n_test 300 --seed_test 123 --arms real shuffle gauss anti wrong_late wrong_zero --betas 0.2 --output_dir experiments/outputs/tldc_controls
    ;;
  ctrl-tldc-456)
    # 第二 seed（主判据两臂：real vs shuffle）
    python experiments/lin_theory/main_tldc_controls.py --model "$MODEL_1P7B" --layer_early 20 --n_test 300 --seed_test 456 --arms real shuffle --betas 0.2 --output_dir experiments/outputs/tldc_controls
    ;;
  ctrl-tldc-lowbeta)
    # 次判据档 β=0.03（real + shuffle，一致性检查）
    python experiments/lin_theory/main_tldc_controls.py --model "$MODEL_1P7B" --layer_early 20 --n_test 300 --seed_test 123 --arms real shuffle --betas 0.03 --output_dir experiments/outputs/tldc_controls
    ;;
  gated-tldc)
    # T3 门控族（τ 由 TAU 覆盖，预注册 0.2/0.3）：real 作同批基准，配对判读复用 analyze_control_arms.py
    python experiments/lin_theory/main_tldc_controls.py --model "$MODEL_1P7B" --layer_early 20 --n_test 300 --seed_test 123 --arms real gated_margin gated_betastar gated_damp --betas 0.2 --gate_tau "${TAU:-0.2}" --output_dir experiments/outputs/tldc_gated
    ;;
  gated-tldc-456)
    # 第二 seed（门控主臂两臂：real + gated_margin）
    python experiments/lin_theory/main_tldc_controls.py --model "$MODEL_1P7B" --layer_early 20 --n_test 300 --seed_test 456 --arms real gated_margin --betas 0.2 --gate_tau "${TAU:-0.2}" --output_dir experiments/outputs/tldc_gated
    ;;
  verified-tldc)
    # 验证器臂（§5.8b；θ=0.5 预注册，THETA 可覆盖但改 θ 即为破预注册，须在报告中声明）
    python experiments/lin_theory/main_tldc_controls.py --model "$MODEL_1P7B" --layer_early 20 --n_test 300 --seed_test "${SEED:-123}" --arms real verified_sym --betas 0.2 --verify_theta "${THETA:-0.5}" --output_dir experiments/outputs/tldc_verified
    ;;
  verified-tldc-8b)
    # 验证器臂 8B（服务器用）；SEED / THETA 可覆盖
    env -u HF_ENDPOINT HF_HOME="${HF_HOME_SERVER:-/root/autodl-tmp/huggingface_cache}" python -u experiments/lin_theory/main_tldc_controls.py --model "$MODEL_8B" --layer_early 28 --ctrl_layer 31 --n_test 300 --seed_test "${SEED:-123}" --arms real verified_sym --betas 0.2 --verify_theta "${THETA:-0.5}" --output_dir experiments/outputs/tldc_verified_8b
    ;;
  verified-rand-smoke)
    # 修复验证冒烟（2026-09-23）：verified_rand 首次上真机即 UnboundLocalError（分支条件与 d_ctrl
    # 守卫不一致，见 main_tldc_controls.loop_branch 注释）⇒ 先小样本真跑一次确认链路再上正式档。
    # n=30、GPU 数分钟；classify 缓存按 n 命名，与 n=300 正式档互不干扰。
    python experiments/lin_theory/main_tldc_controls.py --model "$MODEL_1P7B" --layer_early 20 --n_test 30 --seed_test 123 --arms real verified_sym verified_rand --betas 0.2 --verify_theta "${THETA:-0.5}" --output_dir experiments/outputs/_smoke_tldc_rand
    ;;
  verified-rand)
    # V1 footprint 安慰剂（2026-09-23）：同批 real + verified_sym + verified_rand 三臂。
    # KEEP_P 逐子集保留概率＝**V1 实测 keep_rate_of_flips**（KW 60.53% / KC 69.91% / DK 65.06%），
    # 只按决策计数标定、不看结果 ⇒ 安慰剂与真验证器的期望扰动步数相同。
    python experiments/lin_theory/main_tldc_controls.py --model "$MODEL_1P7B" --layer_early 20 --n_test 300 --seed_test "${SEED:-123}" --arms real verified_sym verified_rand --betas 0.2 --verify_theta "${THETA:-0.5}" --verify_keep_prob "${KEEP_P:-know_wrong=0.6053,know_correct=0.6991,dont_know=0.6506}" --output_dir experiments/outputs/tldc_verified_placebo
    ;;
  verified-rand-8b)
    # 验证器臂 8B 的 footprint 安慰剂（服务器用）。⚠️ KEEP_P **必须**用同批 verified-tldc-8b 实测标定，
    # 不得沿用 1.7B 的数（故这里强制显式传入：KEEP_P='know_wrong=..,know_correct=..,dont_know=..'）。
    env -u HF_ENDPOINT HF_HOME="${HF_HOME_SERVER:-/root/autodl-tmp/huggingface_cache}" python -u experiments/lin_theory/main_tldc_controls.py --model "$MODEL_8B" --layer_early 28 --ctrl_layer 31 --n_test 300 --seed_test "${SEED:-123}" --arms real verified_sym verified_rand --betas 0.2 --verify_theta "${THETA:-0.5}" --verify_keep_prob "${KEEP_P:?必须先由 verified-tldc-8b 实测标定，如 KEEP_P='know_wrong=0.19,know_correct=0.04,dont_know=0.12'}" --output_dir experiments/outputs/tldc_verified_8b_placebo
    ;;
  gated-tldc-8b)
    # T3 门控族 8B（服务器用；H1′ 成立后"门控作为改进主张"的规模复验，ℓ*=28 与 8B TLDC 同协议）
    # SEED / TAU 可覆盖：SEED=456 / TAU=0.3
    env -u HF_ENDPOINT HF_HOME="${HF_HOME_SERVER:-/root/autodl-tmp/huggingface_cache}" python -u experiments/lin_theory/main_tldc_controls.py --model "$MODEL_8B" --layer_early 28 --ctrl_layer 31 --n_test 300 --seed_test "${SEED:-123}" --arms real gated_margin gated_betastar gated_damp --betas 0.2 --gate_tau "${TAU:-0.2}" --output_dir experiments/outputs/tldc_gated_8b
    ;;
  geometry-8b-baseline)
    # V2 判读需**双 seed** 基线（`analyze_posthoc_direction.py --seeds 123 456`）：
    # seed456 用 SEED=456 再跑一次（两档都存同一目录 geometry_archive_8b/）
    env -u HF_ENDPOINT HF_HOME="${HF_HOME_SERVER:-/root/autodl-tmp/huggingface_cache}" python -u experiments/lin_theory/dump_geometry_archive.py --model "$MODEL_8B" --layer_early 28 --n_test 300 --seed_test "${SEED:-123}" --operator baseline --output_dir experiments/outputs/geometry_archive_8b
    ;;
  *)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac
