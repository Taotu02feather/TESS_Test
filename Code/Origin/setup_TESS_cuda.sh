#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="TESS_cuda"
PYTHON_VERSION="3.9"

echo "=================================================="
echo "[INFO] Start setting up CUDA environment: ${ENV_NAME}"
echo "=================================================="

# ====== 1) 基础检查 ======
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# 初始化 conda（兼容非交互 shell）
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ====== 2) 检查环境是否存在 ======
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[INFO] Conda environment '${ENV_NAME}' already exists."
    echo "[INFO] Reusing existing environment and continuing configuration..."
else
    echo "[INFO] Conda environment '${ENV_NAME}' does not exist."
    echo "[INFO] Creating environment '${ENV_NAME}' with Python ${PYTHON_VERSION} ..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# ====== 3) 激活环境 ======
conda activate "${ENV_NAME}"

echo "[INFO] Python executable: $(which python)"
python --version

# ====== 4) 先升级安装工具 ======
echo "[INFO] Upgrading pip/setuptools/wheel ..."
python -m pip install --upgrade pip setuptools wheel packaging

# ninja 有些包会用到，先装上，避免构建时报找不到
python -m pip install --upgrade ninja

# ====== 5) 安装 PyTorch CUDA 11.8 版本 ======
echo "[INFO] Installing torch==2.2.1+cu118 and torchvision==0.17.1+cu118 ..."
python -m pip install --prefer-binary --no-cache-dir \
    torch==2.2.1+cu118 \
    torchvision==0.17.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ====== 6) 安装其余依赖 ======
# 说明：
# - mlflow 依赖链里常会拉到 pandas
# - 显式指定 pandas 的 wheel 版本，尽量避免 metadata-generation/source build 问题
# - numpy 固定为要求的 1.24.1
echo "[INFO] Installing the remaining packages ..."
python -m pip install --prefer-binary --no-cache-dir \
    joblib==1.3.2 \
    mlflow==2.10.2 \
    numpy==1.24.1 \
    pandas==2.1.4 \
    Pillow==11.2.1 \
    tonic==1.4.3

# ====== 7) 依赖检查 ======
echo "[INFO] Running pip check ..."
python -m pip check || true

# ====== 8) 验证安装 ======
echo "[INFO] Verifying installation ..."
python - << 'EOF'
import sys

print("Python version:", sys.version)

try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Built with CUDA:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("[WARN] Torch verification failed:", e)

packages = [
    "joblib",
    "mlflow",
    "numpy",
    "pandas",
    "PIL",
    "tonic",
    "torchvision",
]

for pkg in packages:
    try:
        mod = __import__(pkg if pkg != "PIL" else "PIL")
        ver = getattr(mod, "__version__", "unknown")
        print(f"{pkg}: {ver}")
    except Exception as e:
        print(f"[WARN] Failed to import {pkg}: {e}")
EOF

echo "=================================================="
echo "[SUCCESS] Environment '${ENV_NAME}' is ready."
echo "[INFO] Activate it with: conda activate ${ENV_NAME}"
echo "=================================================="
