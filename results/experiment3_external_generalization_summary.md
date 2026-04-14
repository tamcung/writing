# Experiment 3 External Generalization Summary

## Setting

- Source file: `experiments/formal/results_experiment3_external_generalization_classic_10seed.json`
- Seeds: `11, 22, 33, 44, 55, 66, 77, 88, 99, 111`
- Backbones: `textcnn`, `bilstm`, `codebert`
- Methods: `clean_ce`, `pair_ce`, `pair_proj_ce`, `pair_canonical`
- Training source: `SQLiV3_clean` formal split only
- Pair source: `data/derived/formal_v3/experiment2/pairs`
- External views:
  - `sqliv5_new_sqli_only`: positive-only real transformed SQLi samples
  - `modsec_learn_cleaned_balanced`: balanced external binary classification set
  - `web_attacks_long_test_balanced`: balanced external binary classification set
- Elapsed seconds: `41108.24`

## In-Distribution Reference

| Backbone | Method | F1 | Precision | Recall | P10 |
|---|---|---:|---:|---:|---:|
| TextCNN | `clean_ce` | 0.9930 | 0.9980 | 0.9880 | 0.9981 |
| TextCNN | `pair_ce` | 0.9925 | 0.9958 | 0.9893 | 0.9989 |
| TextCNN | `pair_proj_ce` | 0.9922 | 0.9951 | 0.9894 | 1.0000 |
| TextCNN | `pair_canonical` | 0.9916 | 0.9928 | 0.9905 | 1.0000 |
| BiLSTM | `clean_ce` | 0.9926 | 0.9964 | 0.9889 | 0.9999 |
| BiLSTM | `pair_ce` | 0.9805 | 0.9765 | 0.9867 | 0.9994 |
| BiLSTM | `pair_proj_ce` | 0.9899 | 0.9903 | 0.9895 | 1.0000 |
| BiLSTM | `pair_canonical` | 0.9667 | 0.9483 | 0.9911 | 1.0000 |
| CodeBERT | `clean_ce` | 0.9963 | 0.9992 | 0.9935 | 0.9997 |
| CodeBERT | `pair_ce` | 0.9964 | 0.9983 | 0.9945 | 1.0000 |
| CodeBERT | `pair_proj_ce` | 0.9960 | 0.9984 | 0.9937 | 1.0000 |
| CodeBERT | `pair_canonical` | 0.9958 | 0.9986 | 0.9930 | 1.0000 |

## SQLiV5 New SQLi Only

| Backbone | Method | Recall | P10 |
|---|---|---:|---:|
| TextCNN | `clean_ce` | 0.8554 | 0.3491 |
| TextCNN | `pair_ce` | 0.9335 | 0.7377 |
| TextCNN | `pair_proj_ce` | 0.9450 | 0.8450 |
| TextCNN | `pair_canonical` | 0.9749 | 0.9959 |
| BiLSTM | `clean_ce` | 0.9535 | 0.9274 |
| BiLSTM | `pair_ce` | 0.9530 | 0.9394 |
| BiLSTM | `pair_proj_ce` | 0.9184 | 0.7189 |
| BiLSTM | `pair_canonical` | 0.9421 | 0.9479 |
| CodeBERT | `clean_ce` | 0.9666 | 0.9360 |
| CodeBERT | `pair_ce` | 0.9776 | 0.9911 |
| CodeBERT | `pair_proj_ce` | 0.9845 | 1.0000 |
| CodeBERT | `pair_canonical` | 0.9786 | 0.9991 |

## External Binary Sets

### ModSec-Learn-Cleaned

| Backbone | Method | F1 | Precision | Recall | P10 |
|---|---|---:|---:|---:|---:|
| TextCNN | `clean_ce` | 0.3518 | 0.3675 | 0.3861 | 0.1001 |
| TextCNN | `pair_ce` | 0.4771 | 0.5170 | 0.5480 | 0.1441 |
| TextCNN | `pair_proj_ce` | 0.5222 | 0.5093 | 0.6090 | 0.1723 |
| TextCNN | `pair_canonical` | 0.6053 | 0.5195 | 0.7716 | 0.4048 |
| BiLSTM | `clean_ce` | 0.6139 | 0.5427 | 0.7779 | 0.5045 |
| BiLSTM | `pair_ce` | 0.5575 | 0.5538 | 0.6517 | 0.3185 |
| BiLSTM | `pair_proj_ce` | 0.5878 | 0.6220 | 0.6511 | 0.2796 |
| BiLSTM | `pair_canonical` | 0.5614 | 0.5661 | 0.6521 | 0.3780 |
| CodeBERT | `clean_ce` | 0.9074 | 0.8757 | 0.9586 | 0.8176 |
| CodeBERT | `pair_ce` | 0.7893 | 0.8670 | 0.8274 | 0.7557 |
| CodeBERT | `pair_proj_ce` | 0.8667 | 0.8649 | 0.9013 | 0.7718 |
| CodeBERT | `pair_canonical` | 0.8632 | 0.8619 | 0.9005 | 0.7998 |

### Web-Attacks-Long

| Backbone | Method | F1 | Precision | Recall | P10 |
|---|---|---:|---:|---:|---:|
| TextCNN | `clean_ce` | 0.9612 | 0.9446 | 0.9802 | 0.9977 |
| TextCNN | `pair_ce` | 0.9319 | 0.8846 | 0.9882 | 0.9979 |
| TextCNN | `pair_proj_ce` | 0.9117 | 0.8484 | 0.9879 | 1.0000 |
| TextCNN | `pair_canonical` | 0.8652 | 0.7697 | 0.9918 | 1.0000 |
| BiLSTM | `clean_ce` | 0.9105 | 0.8508 | 0.9853 | 0.9998 |
| BiLSTM | `pair_ce` | 0.8995 | 0.8322 | 0.9827 | 0.9993 |
| BiLSTM | `pair_proj_ce` | 0.9066 | 0.8396 | 0.9870 | 1.0000 |
| BiLSTM | `pair_canonical` | 0.8760 | 0.7886 | 0.9882 | 1.0000 |
| CodeBERT | `clean_ce` | 0.8811 | 0.9237 | 0.8656 | 0.6008 |
| CodeBERT | `pair_ce` | 0.9466 | 0.9808 | 0.9169 | 0.6157 |
| CodeBERT | `pair_proj_ce` | 0.9551 | 0.9819 | 0.9321 | 0.7840 |
| CodeBERT | `pair_canonical` | 0.9584 | 0.9830 | 0.9359 | 0.9005 |

## Paired-Difference Highlights

- On `sqliv5_new_sqli_only`, TextCNN benefits strongly from `pair_canonical`: recall improves over `clean_ce` by `+0.1196` and over `pair_ce` by `+0.0415` with Wilcoxon `p = 0.001953` for both.
- On `sqliv5_new_sqli_only`, CodeBERT also benefits over `clean_ce` in recall (`+0.0121`, `p = 0.009766`), but `pair_proj_ce` has the highest mean recall (`0.9845`).
- On `modsec_learn_cleaned_balanced`, TextCNN benefits strongly from `pair_canonical`: F1 improves over `clean_ce` by `+0.2535` and over `pair_ce` by `+0.1282`.
- On `modsec_learn_cleaned_balanced`, `clean_ce` remains best for BiLSTM and CodeBERT, so the external-generalization claim is not uniformly supported across all backbones.
- On `web_attacks_long_test_balanced`, `pair_canonical` hurts TextCNN/BiLSTM F1 because recall rises but precision drops. CodeBERT benefits most: F1 improves from `0.8811` to `0.9584`.

## Interpretation

Experiment 3 supports a nuanced external-generalization story rather than a universal win. The robust paired methods, especially `pair_canonical`, transfer well for TextCNN on SQLiV5 and ModSec-Learn, and for CodeBERT on web-attacks-long. However, `clean_ce` remains stronger for BiLSTM and CodeBERT on ModSec-Learn, and TextCNN/BiLSTM lose precision on web-attacks-long after paired training.

For the thesis, this result should be written as a boundary-aware finding: canonical paired representation learning improves robustness under semantic-preserving SQLi transformations and some external distributions, but its effect depends on backbone and target distribution. This is stronger and safer than claiming it universally improves cross-dataset generalization.
