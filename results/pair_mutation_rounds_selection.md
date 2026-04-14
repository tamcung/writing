# Pair 变形轮数选择备忘

## 目的

为 `experiment2` 的 pair 构造选择更合理的默认 `mutation_rounds`，避免继续使用缺乏专门依据的经验值。

## 分析对象

- 数据口径：`formal_modsec_decoded`
- 主数据集：`modsec_learn_value_windows`
- SQLi 算子：`official_wafamole`
- benign 对照扰动：value 级 HTTP 协议等价 nuisance transform
- 固定参数：
  - `mutation_retries = 8`
  - `pair_max_chars = 896`

## 分析流程

### 1. 构造侧审计

脚本：

- `experiments/formal/analyze_pair_mutation_rounds.py`

比较 `rounds = 1/3/5/7/8` 下的：

- `sqli_changed_rate`
- `mean_sqli_chain_len`
- `ratio_mean`
- `ratio_p95`

其中：

- `ratio_mean` 表示 `len(mutated) / len(source)` 的平均值
- `ratio_p95` 表示长度膨胀比的 95 分位数

### 2. 小型下游验证

同一脚本下，使用：

- backbone：`TextCNN`
- method：`pair_canonical`
- seeds：`11/22/33`
- `attack_per_class = 100`
- `search_steps = 12`
- `candidates_per_state = 32`
- `beam_size = 4`

比较：

- targeted recall
- attack success
- mean prob drop

说明：

- 这一步用于参数选择，是探索性验证，不作为“正式显著性结论”。

## 关键结果

### 构造侧

| rounds | changed rate | mean chain | ratio mean | ratio p95 |
|---|---:|---:|---:|---:|
| 1 | 0.9847 | 0.9847 | 1.0431 | 1.1918 |
| 3 | 1.0000 | 1.9130 | 1.0793 | 1.3118 |
| 5 | 1.0000 | 2.9887 | 1.1185 | 1.4470 |
| 7 | 1.0000 | 4.1093 | 1.1564 | 1.5508 |
| 8 | 1.0000 | 4.6890 | 1.1758 | 1.6100 |

结论：

- 从 `3` 轮开始，`changed rate` 已经饱和。
- 更高轮数的主要效果，是继续增加变形链长度和文本膨胀。

### 下游验证

| rounds | targeted recall | attack success | mean prob drop |
|---|---:|---:|---:|
| 1 | 0.9067 | 0.0933 | 0.0749 |
| 3 | 0.9100 | 0.0900 | 0.0754 |
| 5 | 0.9100 | 0.0900 | 0.0740 |
| 7 | 0.9533 | 0.0467 | 0.0395 |
| 8 | 0.9600 | 0.0400 | 0.0388 |

结论：

- `1 -> 3 -> 5` 的提升非常有限。
- `5 -> 7` 出现了明显的鲁棒性改善。
- `7 -> 8` 仍有小幅提升，但收益已经进入边际区间。

## 最终决策

正式默认值选择：

- `mutation_rounds = 7`

理由：

1. `7` 相比 `5` 带来更明显的 targeted robustness 改善。
2. `8` 虽然略优于 `7`，但长度膨胀继续增加，收益边际较小。
3. 因此，`7` 更适合作为“鲁棒收益与分布代价折中”下的正式默认配置。

## 当前正式 pair 数据

已重建：

- `data/derived/formal_modsec_decoded/experiment2/pairs`

当前 manifest 配置：

- `mutation_rounds = 7`
- `mutation_retries = 8`
- `max_chars = 896`

当前正式 pair 统计：

- `sqli_changed_rate` 均值约 `0.9998`
- `benign_changed_rate = 1.0000`
- `mean_sqli_chain_len` 约 `4.06 - 4.14`

## 论文可用表述

可写为：

> 本文通过构造侧审计与小型下游验证对 pair 构造的随机变形轮数进行了敏感性分析。结果表明，从 3 轮开始样本变化率已趋于饱和，而 7 轮相较 5 轮能够带来更明显的鲁棒性收益；继续增加到 8 轮仅产生边际提升，同时伴随更大的文本长度膨胀。因此，本文将 pair 构造的默认随机变形轮数设为 7。
