# CodeBERT WAF-A-MoLE Probe

- Source file: `experiments/results_experiment1_official_wafamole_codebert_3seed_fast.json`
- Device: Colab CUDA
- Seeds: `11, 22, 33`
- Attack samples per class: `50`
- Search budget: `steps=6`, `candidates_per_state=16`, `beam_size=2`
- CodeBERT training: `epochs=1`, `batch_size=64`, `max_len=160`
- Elapsed seconds: `273.87`

## Summary

| View | Model | F1 | Recall | Precision | p10_sqli_prob |
|---|---|---:|---:|---:|---:|
| clean_attack_matched | codebert | 0.9967 | 1.0000 | 0.9935 | 0.9977 |
| targeted_official_wafamole | codebert | 0.8146 | 0.6933 | 0.9896 | 0.1775 |
| valid | codebert | 0.9950 | 0.9917 | 0.9983 | 0.9973 |

## Attack Summary

| Model | Attack Success Rate | Mean Prob Drop | Mean Adversarial SQLi Prob | Mean Queries |
|---|---:|---:|---:|---:|
| codebert | 0.3067 | 0.2619 | 0.7359 | 80.33 |

## Interpretation

The fast 3-seed probe suggests that CodeBERT is more robust than TextCNN and BiLSTM under the current low-budget official WAF-A-MoLE targeted search, but it still degrades clearly: recall drops from `1.0000` on clean matched samples to `0.6933` on targeted mutated samples, and the lower-tail SQLi confidence drops from `0.9977` to `0.1775`.

This is a probe result rather than the final formal setting because it uses `epochs=1`, `max_len=160`, `attack_per_class=50`, and a reduced search budget.
