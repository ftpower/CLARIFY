#!/bin/bash
# ============================================================
# CLARIFY 服务器环境自动化配置脚本
# 适用：AutoDL / 矩池云 / 恒源云 等云 GPU 平台
# 基础镜像：PyTorch 2.8.0 / Python 3.12 / CUDA 12.8 / Ubuntu 22.04
# 用法：bash setup_server.sh
# ============================================================
set -e

echo "===== CLARIFY 服务器环境配置 ====="
echo ""

# ---- 基本配置 ----
REPO_URL="https://github.com/ftpower/CLARIFY.git"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 要下载的模型列表（空格分隔）
MODELS=(
  "Qwen/Qwen3-8B"
)

# 要预下载的数据集
DATASETS=(
  "trivia_qa:rc"
  "squad_v2"
  "hellaswag"
)

# ---- Step 0: 检测平台 ----
echo "[0/6] 检测平台..."

if [ -d "/root/autodl-tmp" ]; then
    PLATFORM="AutoDL"
    DATA_DIR="/root/autodl-tmp"
elif [ -d "/home/matpool" ]; then
    PLATFORM="矩池云"
    DATA_DIR="/home/matpool"
else
    PLATFORM="通用"
    DATA_DIR="$HOME"
fi
echo "  平台: $PLATFORM"
echo "  数据盘: $DATA_DIR"
echo ""

# ---- Step 1: 验证基础环境 ----
echo "[1/6] 验证基础环境 (PyTorch 2.8.0 + CUDA 12.8)..."

python -c "
import torch
print(f'  Python:   {__import__(\"sys\").version.split()[0]}')
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.version.cuda}')
print(f'  GPU:      {torch.cuda.get_device_name(0)}')
print(f'  VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
assert torch.cuda.is_available(), 'CUDA not available!'
"
echo ""

# ---- Step 2: 配置 HuggingFace 镜像 ----
echo "[2/6] 配置 HuggingFace 镜像..."

export HF_ENDPOINT="$HF_ENDPOINT"
if ! grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null; then
    echo "export HF_ENDPOINT=$HF_ENDPOINT" >> ~/.bashrc
    echo "  HF_ENDPOINT = $HF_ENDPOINT (已写入 ~/.bashrc)"
else
    echo "  HF_ENDPOINT 已配置"
fi
echo ""

# ---- Step 3: 安装项目依赖 ----
echo "[3/6] 安装项目依赖..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

if [ -f "$REQ_FILE" ]; then
    # 过滤掉镜像已预装的包（torch/nvidia/triton）和不需要的包
    grep -v -E "^(torch==|torchvision==|torchaudio==|triton==|nvidia-|open.clip|sentry-sdk|wandb)" "$REQ_FILE" > /tmp/requirements_clean.txt
    pip install -r /tmp/requirements_clean.txt --quiet
else
    # fallback：直接装核心包
    pip install transformer_lens datasets scipy scikit-learn matplotlib \
        transformers tokenizers accelerate sentencepiece safetensors \
        einops jaxtyping pandas fsspec rich --quiet
fi
echo "  依赖安装完成"
echo ""

# ---- Step 4: 克隆代码仓库 ----
echo "[4/6] 克隆代码仓库..."

PROJECT_DIR="$HOME/CLARIFY"

if [ -n "$REPO_URL" ]; then
    if [ -d "$PROJECT_DIR" ]; then
        echo "  $PROJECT_DIR 已存在，git pull 更新..."
        cd "$PROJECT_DIR" && git pull
    else
        git clone "$REPO_URL" "$PROJECT_DIR"
        echo "  仓库已克隆到 $PROJECT_DIR"
    fi
else
    echo "  [跳过] REPO_URL 未配置，请手动 clone 代码到 $PROJECT_DIR"
    echo "  然后重新运行: bash setup_server.sh"
    echo ""
    echo "===== 配置中止（缺少代码仓库）====="
    exit 0
fi
echo ""

# ---- Step 5: 下载模型到数据盘 ----
echo "[5/6] 下载模型到数据盘..."

MODEL_DIR="$DATA_DIR/checkpoints"
mkdir -p "$MODEL_DIR"

for model in "${MODELS[@]}"; do
    safe_name=$(echo "$model" | tr '/' '-')
    target="$MODEL_DIR/models--$safe_name"

    if [ -d "$target" ] && [ "$(ls -A "$target" 2>/dev/null)" ]; then
        echo "  $model 已存在，跳过"
    else
        echo "  下载 $model ..."
        huggingface-cli download "$model" --local-dir "$target" --resume-download
    fi
done

# 创建软链接，让 model_utils.py 的 CHECKPOINT_DIR 能找到模型
mkdir -p "$PROJECT_DIR/checkpoints"
for model in "${MODELS[@]}"; do
    safe_name=$(echo "$model" | tr '/' '-')
    link_name="$PROJECT_DIR/checkpoints/models--$safe_name"
    if [ ! -e "$link_name" ]; then
        ln -s "$MODEL_DIR/models--$safe_name" "$link_name"
        echo "  软链接: $link_name → $MODEL_DIR/models--$safe_name"
    fi
done
echo ""

# ---- Step 6: 预下载数据集 ----
echo "[6/6] 预下载数据集..."

for ds in "${DATASETS[@]}"; do
    echo "  下载 $ds ..."
    python -c "
from datasets import load_dataset
ds_name, *config = '$ds'.split(':')
if config:
    load_dataset(ds_name, config[0], split='validation', trust_remote_code=False)
else:
    load_dataset(ds_name, split='validation', trust_remote_code=False)
print('    OK')
"
done
echo ""

# ---- 验证 ----
echo "===== 验证环境 ====="

python -c "
import torch, sys, os
import transformer_lens, datasets, transformers

print(f'Python:          {sys.version.split()[0]}')
print(f'PyTorch:         {torch.__version__}')
print(f'CUDA:            {torch.version.cuda}')
print(f'GPU:             {torch.cuda.get_device_name(0)}')
print(f'VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'transformer_lens: {transformer_lens.__version__}')
print(f'datasets:        {datasets.__version__}')
print(f'transformers:    {transformers.__version__}')

# 验证模型文件可访问
checkpoint_dir = os.path.expanduser('$PROJECT_DIR/checkpoints')
if os.path.isdir(checkpoint_dir):
    print(f'Checkpoints:     {os.listdir(checkpoint_dir)}')
else:
    print('Checkpoints:     未找到!')
"

echo ""
echo "===== 配置完成 ====="
echo ""
echo "后续步骤:"
echo "  cd $PROJECT_DIR/experiments/phase1"
echo "  python main.py --n_samples 10 --model Qwen/Qwen3-8B --output_dir outputs_test"
echo ""
echo "首次测试通过后，正式跑:"
echo "  python main.py --n_samples 200 --model Qwen/Qwen3-8B --output_dir outputs_qwen3_8b_200"
