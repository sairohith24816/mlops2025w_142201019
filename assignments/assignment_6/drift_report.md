# Drift Report

**Baseline:** 0.1800 | **Threshold:** 0.1
**Samples:** 200 | **Alerts:** 3

## Results
| scenario   |   accuracy |   drop |   drop_pct | alert   |
|:-----------|-----------:|-------:|-----------:|:--------|
| dark_30    |      0.105 |  0.075 |    41.6667 | False   |
| dark_20    |      0.065 |  0.115 |    63.8889 | True    |
| bright_180 |      0.145 |  0.035 |    19.4444 | False   |
| noise_low  |      0.11  |  0.07  |    38.8889 | False   |
| noise_high |      0.03  |  0.15  |    83.3333 | True    |
| combined   |      0.02  |  0.16  |    88.8889 | True    |

## W&B Links
- [dark_30](https://wandb.ai/ir2023/tinyimagenet-resnet/runs/k3oy28qu)
- [dark_20](https://wandb.ai/ir2023/tinyimagenet-resnet/runs/utaybjbj)
- [bright_180](https://wandb.ai/ir2023/tinyimagenet-resnet/runs/fvpwuh26)
- [noise_low](https://wandb.ai/ir2023/tinyimagenet-resnet/runs/1o3ykngs)
- [noise_high](https://wandb.ai/ir2023/tinyimagenet-resnet/runs/uavvlian)
- [combined](https://wandb.ai/ir2023/tinyimagenet-resnet/runs/mc0rtr4g)
