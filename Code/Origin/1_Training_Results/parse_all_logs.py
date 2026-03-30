import os
import re
import pandas as pd

# ======================
# 路径
# ======================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.join(SCRIPT_DIR, "..", "experiments")
BASE_DIR = os.path.abspath(BASE_DIR)

ALPHA_DIR = os.path.join(BASE_DIR, "alpha_post_test")


CURVE_OUT_DIR = "parsed_curves"
SUMMARY_OUT_DIR = "parsed_summary"

os.makedirs(CURVE_OUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_OUT_DIR, exist_ok=True)


# ======================
# 工具：解析目录名
# ======================
def parse_alpha_folder(name):
    """
    CIFAR10_VGG_TESS_apo_neg1
    -> dataset=CIFAR10, method=TESS, alpha=-1
    """
    if "apo_neg1" in name:
        alpha = -1
    elif "apo_zero" in name:
        alpha = 0
    elif "apo_pos1" in name:
        alpha = 1
    else:
        return None

    # dataset
    dataset = name.split("_")[0]

    # method
    if "TESS" in name:
        method = "TESS"
    elif "BPTT" in name:
        method = "BPTT"
    else:
        method = "Unknown"

    return dataset, method, alpha


def parse_baseline_folder(name):
    """
    CIFAR10_VGG_BPTT_
    CIFAR10_VGG_TESS_
    -> dataset, method, alpha=None
    """
    dataset = name.split("_")[0]

    if "BPTT" in name:
        method = "BPTT"
    elif "TESS" in name:
        method = "TESS"
    else:
        return None

    return dataset, method


# ======================
# 核心：解析 log
# ======================
def parse_log(filepath):
    train_re = re.compile(r"@Training\s+\*\s+Acc@1\s+([0-9.]+).*Loss\s+([0-9.]+)")
    test_re = re.compile(r"@Testing\s+\*\s+Acc@1\s+([0-9.]+).*Loss\s+([0-9.]+)")
    best_re = re.compile(r"Best acc at epoch\s+(\d+):\s+([0-9.eE+-]+)")
    epoch_re = re.compile(r"Epoch\s+(\d+)")

    current_epoch = None
    data = {}
    best_acc = None
    best_epoch = None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:

            # epoch
            if "Epoch" in line and "..." in line:
                m = epoch_re.search(line)
                if m:
                    current_epoch = int(m.group(1))
                    data[current_epoch] = {
                        "epoch": current_epoch,
                        "train_acc": None,
                        "train_loss": None,
                        "test_acc": None,
                        "test_loss": None,
                    }

            # train
            m = train_re.search(line)
            if m and current_epoch:
                data[current_epoch]["train_acc"] = float(m.group(1))
                data[current_epoch]["train_loss"] = float(m.group(2))

            # test
            m = test_re.search(line)
            if m and current_epoch:
                data[current_epoch]["test_acc"] = float(m.group(1))
                data[current_epoch]["test_loss"] = float(m.group(2))

            # best
            m = best_re.search(line)
            if m:
                best_epoch = int(m.group(1))
                best_acc = float(m.group(2))

    df = pd.DataFrame(sorted(data.values(), key=lambda x: x["epoch"]))

    # final
    final = df.iloc[-1] if len(df) > 0 else None

    summary = {
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "final_train_acc": final["train_acc"] if final is not None else None,
        "final_test_acc": final["test_acc"] if final is not None else None,
        "final_test_loss": final["test_loss"] if final is not None else None,
    }

    return df, summary


# ======================
# 主函数
# ======================
all_summary = []

# ===== 1️⃣ alpha_post 实验 =====
for folder in os.listdir(ALPHA_DIR):
    full_path = os.path.join(ALPHA_DIR, folder)

    if not os.path.isdir(full_path):
        continue

    parsed = parse_alpha_folder(folder)
    if parsed is None:
        continue

    dataset, method, alpha = parsed

    log_path = os.path.join(full_path, "log.log")
    if not os.path.exists(log_path):
        continue

    df, summary = parse_log(log_path)

    # 保存曲线
    df.insert(0, "alpha", alpha)
    df.insert(0, "method", method)
    df.insert(0, "dataset", dataset)

    out_name = f"{dataset}_{method}_alpha{alpha}.csv"
    df.to_csv(os.path.join(CURVE_OUT_DIR, out_name), index=False)

    summary.update({
        "dataset": dataset,
        "method": method,
        "alpha": alpha,
    })

    all_summary.append(summary)

    print("[OK]", out_name)


# ===== 2️⃣ baseline =====
for folder in os.listdir(BASE_DIR):
    full_path = os.path.join(BASE_DIR, folder)

    if not os.path.isdir(full_path):
        continue

    if folder == "alpha_post_test":
        continue

    parsed = parse_baseline_folder(folder)
    if parsed is None:
        continue

    dataset, method = parsed

    log_path = os.path.join(full_path, "log.log")
    if not os.path.exists(log_path):
        continue

    df, summary = parse_log(log_path)

    df.insert(0, "alpha", "baseline")
    df.insert(0, "method", method)
    df.insert(0, "dataset", dataset)

    out_name = f"{dataset}_{method}_baseline.csv"
    df.to_csv(os.path.join(CURVE_OUT_DIR, out_name), index=False)

    summary.update({
        "dataset": dataset,
        "method": method,
        "alpha": "baseline",
    })

    all_summary.append(summary)

    print("[OK]", out_name)


# ===== 汇总 =====
summary_df = pd.DataFrame(all_summary)
summary_df.to_csv(os.path.join(SUMMARY_OUT_DIR, "summary.csv"), index=False)

print("\nDONE.")