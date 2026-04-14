# Experiment 2 Pair Training Targeted Summary

## Formal Classic Backbones

- Source file: `experiments/formal/results_experiment2_pair_training_targeted_classic_10seed.json`
- Seeds: `11, 22, 33, 44, 55, 66, 77, 88, 99, 111`
- Backbones: `textcnn`, `bilstm`
- Methods: `clean_ce`, `pair_ce`, `pair_proj_ce`, `pair_canonical`
- Pair source: `data/derived/formal_v3/experiment2/pairs`
- Pair mutation: official WAF-A-MoLE operators, random chain, `rounds=5`, `retries=8`
- Attack samples per class: `300`
- Search budget: `steps=20`, `candidates_per_state=48`, `beam_size=5`
- Attack operator set: `official_wafamole`
- Elapsed seconds: `21041.84`

## Main Results

| Backbone | Method | Clean F1 | Mutated F1 | Mutated Recall | Mutated p10 | Attack Success | Mean Drop |
|---|---|---:|---:|---:|---:|---:|---:|
| TextCNN | `clean_ce` | 0.9924 | 0.2726 | 0.1720 | 0.1784 | 0.8280 | 0.5525 |
| TextCNN | `pair_ce` | 0.9921 | 0.6842 | 0.5373 | 0.3223 | 0.4627 | 0.3244 |
| TextCNN | `pair_proj_ce` | 0.9918 | 0.7759 | 0.6473 | 0.2491 | 0.3527 | 0.2526 |
| TextCNN | `pair_canonical` | 0.9907 | 0.9156 | 0.8527 | 0.5298 | 0.1473 | 0.1048 |
| BiLSTM | `clean_ce` | 0.9916 | 0.6772 | 0.5187 | 0.1777 | 0.4813 | 0.3410 |
| BiLSTM | `pair_ce` | 0.9801 | 0.8526 | 0.7690 | 0.3566 | 0.2310 | 0.1819 |
| BiLSTM | `pair_proj_ce` | 0.9892 | 0.8752 | 0.7903 | 0.2294 | 0.2097 | 0.1675 |
| BiLSTM | `pair_canonical` | 0.9660 | 0.8909 | 0.8533 | 0.4681 | 0.1467 | 0.1126 |

## Paired Differences

| Backbone | Comparison | Recall Diff | F1 Diff | p10 Diff | Attack Success Diff | Wilcoxon Summary |
|---|---|---:|---:|---:|---:|---|
| TextCNN | `pair_ce - clean_ce` | +0.3653 | +0.4116 | +0.1439 | -0.3653 | all p = 0.001953 |
| TextCNN | `pair_proj_ce - pair_ce` | +0.1100 | +0.0917 | -0.0732 | -0.1100 | recall/F1/success p = 0.001953; p10 p = 0.01367 |
| TextCNN | `pair_canonical - pair_proj_ce` | +0.2053 | +0.1396 | +0.2807 | -0.2053 | all p = 0.001953 |
| BiLSTM | `pair_ce - clean_ce` | +0.2503 | +0.1754 | +0.1790 | -0.2503 | recall/F1/success p = 0.001953; p10 p = 0.005859 |
| BiLSTM | `pair_proj_ce - pair_ce` | +0.0213 | +0.0226 | -0.1272 | -0.0213 | not significant under Wilcoxon |
| BiLSTM | `pair_canonical - pair_proj_ce` | +0.0630 | +0.0157 | +0.2387 | -0.0630 | recall/success p = 0.005859; p10 p = 0.001953; F1 not significant |

## Interpretation

The 10-seed classic-backbone result supports Experiment 2. Pair training substantially improves robustness over `clean_ce` on both TextCNN and BiLSTM. On TextCNN, `pair_proj_ce` adds a stable gain over `pair_ce` in recall, F1, and attack success rate, but its p10 probability is lower than `pair_ce`. On BiLSTM, `pair_proj_ce` is not a stable improvement over `pair_ce`.

The strongest method in this run is `pair_canonical`. It consistently reduces attack success rate and improves mutated recall on both backbones. This suggests the thesis story should be adjusted: `pair_proj_ce` can remain an important projection baseline, but `pair_canonical` is currently the best-supported robust representation method for the classic neural backbones.
