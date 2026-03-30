# ABLATION STUDY ON INCLUDING NON-CAUSAL TERMS

## Training Scripts:

- `script_alpha_post.sh`

- `script_CIFAR10DVS_alpha.sh`

`DVSCIFAR10`部分是重复的，因为修复问题后重新对此部分进行了训练

## Result Logs:

Under the Dir:`\experiments\alpha_post_test`

## 分析文件：

- `parse_alpha_ablation.py`

- `plot_alpha_ablation.m`

只接受严格合法的 ablation 目录名，例如：
- `CIFAR10_VGG_TESS_apo_neg1`

- `CIFAR10_VGG_TESS_apo_zero`

- `CIFAR10_VGG_TESS_apo_pos1`

- `CIFAR100_VGG_TESS_apo_neg1`

- `DVSGesture_TESS_apo_pos1`

