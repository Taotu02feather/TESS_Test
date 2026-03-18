#!/usr/bin/env bash
set -e

ENV_NAME="TESS_cuda"
REQ_FILE="requirements_fix.txt"

# ====== 基础检查 ======
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

if [ ! -f "${REQ_FILE}" ]; then
    echo "[ERROR] ${REQ_FILE} not found in current directory."
    exit 1
fi

# 初始化 conda（兼容非交互 shell）
source "$(conda info --base)/etc/profile.d/conda.sh"

# ====== 环境是否存在 ======
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[INFO] Conda environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and reinstall it? [y/N]: " REPLY
    if [[ "${REPLY}" =~ ^[Yy]$ ]]; then
        echo "[INFO] Removing existing environment '${ENV_NAME}'..."
        conda env remove -n "${ENV_NAME}"
    else
        echo "[INFO] Aborting installation."
        exit 0
    fi
fi

# ====== 创建环境 ======
echo "[INFO] Creating conda environment '${ENV_NAME}'..."
conda create -n "${ENV_NAME}" python=3.9 -y

conda activate "${ENV_NAME}"

# ====== 升级 pip ======
python -m pip install --upgrade pip

# ====== 安装 requirements ======
echo "[INFO] Installing packages from ${REQ_FILE}..."

pip install -r "${REQ_FILE}" \
    --extra-index-url https://download.pytorch.org/whl/cu118

# ====== 验证 ======
echo "[INFO] Verifying installation..."
python - << EOF
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
EOF

echo "[SUCCESS] Environment '${ENV_NAME}' setup completed."
echo "Activate it with: conda activate ${ENV_NAME}"
