# 正式实验方案

## 一、当前重构口径

从 `2026-04-14` 起，正式实验主线切换为 `formal_modsec_decoded`：

- 不再以 `SQLiV3_clean` 作为主训练集。
- 不再以 `SQLiV5` 作为正文主变形验证集。
- 旧 `formal_v3` 结果保留为探索记录和备选材料，但后续正式设计、命令和结论优先使用 `formal_modsec_decoded`。
- 当前主数据集为 `ModSec-Learn` 的 value-only length-matched windows 视图。

这次重构的动机是：`SQLiV3` 的 benign 中存在大量合法 SQL-like 文本，导致数据分布更接近“SQL 语句分类”；而 `ModSec-Learn` 更接近 HTTP 参数 / payload 级输入。进一步审计发现，原始 `key=value` 形式存在 key 结构泄漏：SQLi 样本几乎全部为 `p=payload`，因此正式主线去除参数名，仅保留参数值。

## 二、数据集处理

### 1. 主数据集：`modsec_learn_value_windows`

- 原始来源：`https://github.com/pralab/modsec-learn-dataset`
- 处理脚本：`experiments/formal/prepare_modsec_decoded_dataset.py`
- processed 目录：`data/processed/formal_modsec_decoded`
- 主数据文件：`data/processed/formal_modsec_decoded/datasets/modsec_learn_value_windows.json`
- 处理规则：
  - 默认排除 `sqli_kaggle` 来源，降低与常见 Kaggle SQLi 数据的同源风险。
  - 从原始 query-string 中解析参数值，并对参数值做一次 `URL decode` / `unquote_plus`。
  - 对 SQLi 样本，去除 `p=` 等参数名，仅保留 payload value。
  - 对 benign 样本，去除参数名后，将同一请求中的多个短 value 拼成长度匹配窗口。
  - benign window 默认最小长度为 `32`，目标长度为 `96`，每条原始 benign 请求最多构造 `1` 个窗口。
  - value 拼接使用普通空格，不引入 `[PARAM]` 等只在 benign 中出现的专用分隔符。
  - 删除空文本。
  - 删除解码/构造后长度超过 `448` 的文本。
  - 按文本去重；若同一文本存在冲突标签则丢弃。
  - benign / legitimate 映射为 `0`，SQLi / malicious 映射为 `1`。

当前 processed 规模：

- 总计：`70267`
- benign：`50721`
- SQLi：`19546`
- `decode_passes = 1`
- `max_len = 448`

同时保留 `modsec_learn_decoded` 作为审计/消融视图，但不作为当前主实验默认输入。

### 2. 切分协议

- split 目录：`data/derived/formal_modsec_decoded/experiment1/splits`
- 切分脚本：`experiments/formal/build_experiment1_splits.py`
- 每个 seed 固定：
  - train：每类 `3000`
  - valid：每类 `500`
  - clean_test：每类 `3000`
- seeds：
  - `11 22 33 44 55 66 77 88 99 111`

### 3. 配对训练数据

- pair 目录：`data/derived/formal_modsec_decoded/experiment2/pairs`
- 构造脚本：`experiments/formal/build_experiment2_pairs.py`
- SQLi 侧：
  - 使用 WAF-A-MoLE 官方 SQL-level operators 随机多轮变形。
  - 默认 `mutation_rounds = 7`，`mutation_retries = 8`，`max_chars = 896`。
- benign 侧：
  - 不使用 SQL 语义变形。
  - 使用 HTTP 参数级 URL 等价编码作为 nuisance transform。
  - 目的：让 benign 样本也出现表面变化，避免模型学习到 `changed => malicious` 的捷径。

当前 pair 构造结果：

- 每个 seed：`6000` 对。
- SQLi changed rate：均值约 `0.9998`，最小值约 `0.9993`。
- benign changed rate：`1.0000`。
- SQLi 平均有效变形链长：约 `4.06-4.14`。

轮数选择依据，保留为正式配置说明：

- 先做构造侧审计，再做轻量下游验证，不凭经验拍脑袋定轮数。
- 构造侧结果：
  - `1` 轮时 `sqli_changed_rate` 约 `0.985`，仍有少量样本无法稳定改写。
  - 从 `3` 轮开始，`sqli_changed_rate` 已达 `1.000`，说明“是否发生变化”这一目标已基本饱和。
  - 更高轮数主要继续增加变形链长度与长度膨胀。
- `TextCNN + pair_canonical` 三 seed 探索性验证结果：
  - targeted recall：`1轮 0.9067`，`3轮 0.9100`，`5轮 0.9100`，`7轮 0.9533`，`8轮 0.9600`
  - attack success：`1轮 0.0933`，`3轮 0.0900`，`5轮 0.0900`，`7轮 0.0467`，`8轮 0.0400`
- 选择 `7` 而不是 `8` 的原因：
  - `7` 相比 `5` 出现了更明显的鲁棒性提升；
  - `8` 相比 `7` 只有边际收益，但长度膨胀继续增大。
  - 因此 `7` 更适合作为“收益与分布代价折中”下的默认配置。

## 三、变形测试协议

正式变形测试使用“基于 WAF-A-MoLE 官方 SQL 变形算子的有限预算目标导向搜索”。

- 算子来源：WAF-A-MoLE 官方 `SqlFuzzer.strategies`
- 当前使用的官方算子：
  - `spaces_to_comments`
  - `random_case`
  - `swap_keywords`
  - `swap_int_repr`
  - `spaces_to_whitespaces_alternatives`
  - `comment_rewriting`
  - `change_tautologies`
  - `logical_invariant`
  - `reset_inline_comments`
- 搜索逻辑：
  - 对 clean_test 中的 SQLi 样本生成多轮候选变形。
  - 使用当前检测器输出的 SQLi 概率给候选排序。
  - 保留 SQLi 概率最低的若干候选继续搜索。
  - 若候选 SQLi 概率低于 `0.5`，记为攻击成功并提前停止。
  - 若预算耗尽仍未低于 `0.5`，保留最低概率候选用于评估。
- 正式默认预算：
  - `search_steps = 20`
  - `candidates_per_state = 48`
  - `beam_size = 5`
  - `attack_per_class = 300`
  - `threshold = 0.5`

## 四、实验矩阵

### 实验 1：问题存在性验证

目的：验证普通 SQLi 检测器在 ModSec-decoded 参数级数据上，即使 clean 测试表现很高，面对 WAF-A-MoLE 式目标导向语义保持变形仍会退化。

训练方法：

- `clean_ce`

模型：

- `word_svc`
- `textcnn`
- `bilstm`
- `codebert`，单独跑，避免拖慢经典模型实验。

测试视图：

- `clean_attack_matched`
- `targeted_official_wafamole`

主要指标：

- `F1`
- `Recall`
- `Attack Success Rate`
- `p10_sqli_prob`
- `mean_prob_drop`
- `mean_queries`

长度调整前 probe 结果，非正式，仅保留为调参参考：

- `textcnn`, seed `11`, `attack_per_class=100`, `steps=20`, `candidates=48`, `beam=5`
- `clean_ce`: clean F1 `1.0000`，targeted F1 `0.8166`，targeted recall `0.6900`，attack success `0.3100`。

当前 `max_len=448` 口径已完成 smoke 验证，但还没有重新跑中等预算 probe；是否仍稳定退化需要 10-seed 正式结果确认。

### 实验 2：主方法有效性验证

目的：验证配对训练、projection representation 和 canonical-anchor alignment 是否能降低目标导向变形攻击成功率。

方法：

- `clean_ce`：只使用干净样本做普通监督训练。
- `pair_ce`：对 `x_canon` 和 `x_raw_mut` 同时做分类交叉熵。
- `pair_proj_ce`：使用 projector 表示做分类，检验 projection 本身是否有用。
- `pair_canonical`：在 `pair_proj_ce` 基础上增加变形样本向 canonical anchor 表示靠近的对齐损失，作为当前主候选。

模型：

- `textcnn`
- `bilstm`
- `codebert`，单独跑。

长度调整前 probe 结果，非正式，仅保留为调参参考：

| 方法 | clean F1 | targeted F1 | targeted recall | p10 | attack success | mean drop |
|---|---:|---:|---:|---:|---:|---:|
| `clean_ce` | `1.0000` | `0.8166` | `0.6900` | `0.2230` | `0.3100` | `0.2567` |
| `pair_ce` | `1.0000` | `0.8235` | `0.7000` | `0.2903` | `0.3000` | `0.2361` |
| `pair_proj_ce` | `0.9950` | `0.8304` | `0.7100` | `0.3018` | `0.2900` | `0.2161` |
| `pair_canonical` | `1.0000` | `0.8636` | `0.7600` | `0.2222` | `0.2400` | `0.1857` |

初步判断：

- `pair_canonical` 在长度调整前的单 seed probe 中优于 `pair_ce` 和 `pair_proj_ce`。
- 改善幅度存在，但不算巨大，需要在 `max_len=448` 口径下用 10-seed 正式结果确认稳定性。
- `p10_sqli_prob` 未同步提升，说明后续分析不能只挑单一指标，需要同时报告 recall、success 和 mean drop。

## 五、正式运行建议

经典模型正式实验建议先跑：

```bash
python -u experiments/formal/run_experiment1_targeted_attack.py \
  --splits-dir data/derived/formal_modsec_decoded/experiment1/splits \
  --backbones word_svc textcnn bilstm \
  --seeds 11 22 33 44 55 66 77 88 99 111 \
  --operator-set official_wafamole \
  --attack-per-class 300 \
  --search-steps 20 \
  --candidates-per-state 48 \
  --beam-size 5 \
  --max-chars 896 \
  --epochs 8 \
  --batch-size 128 \
  --max-tokens 256 \
  --lowercase \
  --device mps \
  --resume \
  --output experiments/formal/results_modsec_decoded_experiment1_classic_10seed.json
```

```bash
python -u experiments/formal/run_experiment2_pair_training_targeted.py \
  --splits-dir data/derived/formal_modsec_decoded/experiment1/splits \
  --backbones textcnn bilstm \
  --methods clean_ce pair_ce pair_proj_ce pair_canonical \
  --seeds 11 22 33 44 55 66 77 88 99 111 \
  --pairs-dir data/derived/formal_modsec_decoded/experiment2/pairs \
  --require-pairs \
  --attack-per-class 300 \
  --search-steps 20 \
  --candidates-per-state 48 \
  --beam-size 5 \
  --max-chars 896 \
  --pair-max-chars 896 \
  --train-operator-set official_wafamole \
  --attack-operator-set official_wafamole \
  --epochs 8 \
  --batch-size 128 \
  --max-tokens 256 \
  --lowercase \
  --device mps \
  --resume \
  --output experiments/formal/results_modsec_decoded_experiment2_classic_10seed.json
```

CodeBERT 单独跑，避免和经典模型互相拖慢：

```bash
python -u experiments/formal/run_experiment2_pair_training_targeted.py \
  --splits-dir data/derived/formal_modsec_decoded/experiment1/splits \
  --backbones codebert \
  --methods clean_ce pair_ce pair_proj_ce pair_canonical \
  --seeds 11 22 33 44 55 66 77 88 99 111 \
  --pairs-dir data/derived/formal_modsec_decoded/experiment2/pairs \
  --require-pairs \
  --attack-per-class 300 \
  --search-steps 20 \
  --candidates-per-state 48 \
  --beam-size 5 \
  --max-chars 896 \
  --pair-max-chars 896 \
  --train-operator-set official_wafamole \
  --attack-operator-set official_wafamole \
  --codebert-epochs 2 \
  --codebert-batch-size 16 \
  --max-len 512 \
  --allow-download \
  --device cuda \
  --resume \
  --output experiments/formal/results_modsec_decoded_experiment2_codebert_10seed.json
```

## 六、当前风险

- ModSec-decoded 的 SQLi 样本 SQL 关键词密度很高，clean 分类可能过于容易。
- 当前 probe 中 `pair_canonical` 有优势，但幅度不大，必须用多 seed 判断是否稳定。
- benign 侧 URL 等价扰动是为了控制 `changed => malicious` 捷径，但论文中必须明确它是 HTTP 参数级等价扰动，而不是 SQL 语义变形。
- 如果正式 10-seed 结果显示优势不稳定，需要把 `formal_modsec_decoded` 定位为“更贴近参数级输入的补充主线”，而不是彻底否定旧 `formal_v3` 探索。
