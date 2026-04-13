# Experiment 1 Official WAF-A-MoLE Summary

## Formal Classic Baselines

- Source file: `experiments/formal/results_experiment1_official_wafamole_classic_10seed.json`
- Seeds: `11, 22, 33, 44, 55, 66, 77, 88, 99, 111`
- Backbones: `word_svc`, `textcnn`, `bilstm`
- Attack samples per class: `300`
- Search budget: `steps=20`, `candidates_per_state=48`, `beam_size=5`
- Operator set: `official_wafamole`
- Lowercase: enabled for neural baselines
- Device: local MPS
- Elapsed seconds: `5292.03`

| Model | Clean F1 | Clean Recall | Clean p10 | Mutated F1 | Mutated Recall | Mutated p10 | Attack Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| word_svc | 0.9797 | 0.9907 | 0.9994 | 0.8006 | 0.6907 | 0.4088 | 0.3093 |
| textcnn | 0.9924 | 0.9867 | 0.9980 | 0.2718 | 0.1717 | 0.1746 | 0.8283 |
| bilstm | 0.9918 | 0.9873 | 0.9995 | 0.7109 | 0.5667 | 0.1936 | 0.4333 |

## CodeBERT Probe

- Source file: `experiments/results_experiment1_official_wafamole_codebert_3seed_fast.json`
- Seeds: `11, 22, 33`
- Backbones: `codebert`
- Attack samples per class: `50`
- Search budget: `steps=6`, `candidates_per_state=16`, `beam_size=2`
- CodeBERT training: `epochs=1`, `batch_size=64`, `max_len=160`
- Device: Colab CUDA
- Elapsed seconds: `273.87`

| Model | Clean F1 | Clean Recall | Clean p10 | Mutated F1 | Mutated Recall | Mutated p10 | Attack Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| codebert | 0.9967 | 1.0000 | 0.9977 | 0.8146 | 0.6933 | 0.1775 | 0.3067 |

## Interpretation

The formal 10-seed classic-baseline result supports the first hypothesis: SQLi detectors trained only on clean samples degrade under official WAF-A-MoLE SQL-level semantic-preserving mutation with finite-budget targeted search. TextCNN is the most vulnerable baseline under this protocol, while word-SVC is more stable but still degrades clearly.

The CodeBERT result is not directly comparable to the classic 10-seed setting because it uses a smaller budget and fewer seeds. It is still useful as a probe: CodeBERT appears more robust than TextCNN and BiLSTM under the reduced search budget, but its mutated recall still drops from `1.0000` to `0.6933`, and the lower-tail SQLi probability drops from `0.9977` to `0.1775`.

For thesis writing, the main Experiment 1 table should use the classic 10-seed baselines. CodeBERT should be reported as a supplementary probe unless a full-budget CodeBERT run is completed later.
