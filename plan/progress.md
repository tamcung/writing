# 当前进度

## 已确认

- 论文类型：中文硕士毕业论文
- 学科方向：网络空间安全
- 最终题目：面向语义保持变形的鲁棒 SQL 注入检测方法研究与实现
- 方法主角：`RSQLi-PR`（以 `pair_proj_ce` 为核心）
- 扩展机制：`pair_canonical`
- 正文主实验主角：`CodeBERT`
- 强对照：`TextCNN`
- 补充基线：`BiLSTM`、`word-SVC`
- 系统部分保留
- 数据集最终方案：
  - 主训练/同分布：`SQLiV3_clean`
  - 主跨变形：`semantic family holdout`
  - 真实变形补充：`SQLiV5`
  - 主外部验证：`ModSec-Learn-cleaned`
  - 补充外部验证：`web-attacks-long`
- 正式 processed 版本：
  - `formal_v3`
  - `max_len = 320`
  - 不剔除完整 SQL-like 样本
  - 统一按参数 / payload / pattern 级字符串处理

## 下一步

1. 让用户运行 experiment1 的正式 `clean_ce` 主结果：`CodeBERT / TextCNN / BiLSTM / word-SVC`
2. experiment1 主表同时报告 `mixed` 与 `mixed_hard`
3. 重新设计 `boolean_equivalent`，避免语义风险
4. 基于 experiment1 结果再进入后续 paired / RSQLi-PR 实验

## 已落地文件

- `experiments/prepare_modsec_learn_cleaned.py`
- `data/raw/ModSec-Learn-cleaned.json`
- `data/raw/ModSec-Learn-cleaned_audit.json`
- `experiments/audit_sqliv5_protocol.py`
- `data/raw/SQLiV5_audit.json`
- `experiments/formal/raw_processing.py`
- `experiments/formal/prepare_raw_datasets.py`
- `experiments/formal/semantic_mutation.py`
- `experiments/formal/build_experiment1_splits.py`
- `experiments/formal/build_experiment1_mutation_views.py`
- `experiments/formal/audit_experiment1_mutation_families.py`
- `experiments/formal/audit_experiment1_mutation_families_multi.py`
- `experiments/formal/run_experiment1_clean_ce.py`
- `data/processed/formal_v3/manifest.json`
- `data/processed/formal_v3/datasets/*.json`
- `data/processed/formal_v3/audits/*.json`
- `data/processed/formal_v3/length_audit.json`
- `data/derived/formal_v3/experiment1/splits/*`
- `data/derived/formal_v3/experiment1/views/*`
- `data/derived/formal_v3/experiment1/audit/*`

## 当前发现

- experiment1 单 seed smoke 已跑通：
  - split：每类 `1000/200/1000`
  - mixed 变形可用 SQLi `996/1000`
  - `surface_obfuscation` 可用 SQLi `994/1000`
  - `numeric_repr` 可用 SQLi `967/1000`
  - `string_construction` 初版可用 SQLi `174/1000`
- experiment1 的 10-seed split 与 mutation views 已全部生成
- experiment1 clean_ce 专用 runner 已完成：
  - `experiments/formal/run_experiment1_clean_ce.py`
  - smoke 已通过：`word_svc + textcnn + seed=11`
  - 已支持 `mixed_hard` 视图
- 结论：
  - `surface_obfuscation` 与 `numeric_repr` 已具备进入实验 1 主测试的条件
  - `string_construction` 已完成一次保守升级，当前可用 SQLi 提升到 `331/1000`
  - `string_construction` 仍是中覆盖家族，更适合作为补充家族而非实验 1 主家族
  - `boolean_equivalent` 当前可用 SQLi `293/1000`，且现规则存在语义风险，不宜直接进入主协议

- `seed=11` 家族审计结果：
  - `surface_obfuscation`: `994/1000`
  - `numeric_repr`: `967/1000`
  - `string_construction`: `331/1000`
  - `boolean_equivalent`: `293/1000`
- 10-seed 家族覆盖率统计：
  - `surface_obfuscation`: mean `0.9931`, min `0.9900`, max `0.9950`
  - `numeric_repr`: mean `0.9561`, min `0.9470`, max `0.9670`
  - `string_construction`: mean `0.3614`, min `0.3310`, max `0.3870`
  - `boolean_equivalent`: mean `0.2818`, min `0.2670`, max `0.2980`
- 强度升级 smoke：
  - `mixed_hard` 已落地到 `experiment1/views`
  - `textcnn + seed=11` 上，`mixed_hard` 退化强于 `mixed`
  - `ΔF1`: `-0.0420` vs `-0.0312`
  - `ΔRecall`: `-0.0803` vs `-0.0602`
  - `ΔP10`: `-0.3477` vs `-0.1793`

## 2026-04-13 变形协议强度校正

- 问题：旧版 `mixed_hard` 可能抽到 `numeric_repr+numeric_repr+numeric_repr`，仍保留大量 SQL 关键词，导致 `word_svc` 基本不退化。
- 曾短暂尝试全量 HTTP percent-encoding：
  - 单 seed 探针能把 `word_svc/textcnn/bilstm` 的 recall 打到 `0.0000`。
  - 但该协议被否决为主实验，因为只需一次 URL 解码即可恢复原 payload，攻击过于便宜，不能支撑核心论文贡献。
- 当前回退状态：
  - `transport_encoding` 不进入正式主协议。
  - `mixed_hard` 回到 `mixed_primary_hard_forced_surface`：先做值级等价改写，再强制做一次 SQL-level surface obfuscation。
- 下一步：
  - 需要设计非 URL 编码的强变形：优先考虑 WAF-A-MoLE 式 SQL-level operators + 目标导向搜索，而不是简单全量编码。

## 2026-04-13 Targeted SQL-level Evasion

- 已新增正式模块：
  - `experiments/formal/targeted_sql_mutation.py`
  - `experiments/formal/run_experiment1_targeted_attack.py`
- 设计：
  - 算子是 SQL-level，不使用 URL percent-encoding。
  - `conservative` 算子集：大小写、空白/注释分隔、普通注释重写、数值表示、字符串构造、布尔恒真式替换/插入。
  - `wafamole_style` 算子集：在 conservative 基础上加入 MySQL 风格操作符同义改写、注释标记改写、MySQL executable comments。
  - 搜索方式：黑盒目标导向 beam search，每轮生成候选，用当前检测器的 SQLi 概率排序，保留概率最低的候选继续搜索。
- 语义边界：
  - 当前强算子中 MySQL executable comments、`||/&&`、`LIKE` tautology 属于 MySQL-style / WAF-style 语义假设，论文需要单独说明数据库方言假设。
  - 已修复 `reset_inline_comments`，避免误删 `/*!50000...*/` 可执行注释导致语义不可靠。
- 单 seed probe（非正式结果）：
  - `word_svc`, seed `11`, 50 SQLi, search `40x96`, beam `8`: clean recall `1.0000` -> targeted recall `0.8200`, p10 `0.4673`, success `0.1800`。
  - `textcnn`, seed `11`, 100 SQLi, search `20x48`, beam `5`: clean recall `1.0000` -> targeted recall `0.1300`, p10 `0.0490`, success `0.8700`。
  - `bilstm`, seed `11`, 100 SQLi, search `20x48`, beam `5`: clean recall `1.0000` -> targeted recall `0.3200`, p10 `0.1003`, success `0.6800`。
- 当前判断：
  - 目标导向 SQL-level 变形能显著打击 `textcnn/bilstm`。
  - `word_svc` 是更强的词袋基线，在当前算子和预算下不宜强行写成“大面积被打穿”，更适合作为鲁棒基线/对照。

## 2026-04-13 Real Benign Hard Negative Probe

- 已新增：
  - `experiments/formal/hard_negative.py`
  - `run_experiment1_targeted_attack.py` 支持 `--hard-negative-extra`、`--hard-negative-min-score`、`--no-hard-negative-balance-sqli`
- 设计：
  - 不人工造 benign。
  - 从当前训练 split 的真实 benign 中按 SQLi 表面相似度打分，选 top-K benign 过采样。
  - 默认同时过采样 K 个 SQLi，避免类别比例变化。
- seed=11 训练集审计：
  - benign hard negative eligible (`score >= 3.0`): `609/1000`
  - top examples 是正常 SQL 查询/WordPress 查询类 benign。
- 单 seed probe（非正式）：
  - `word_svc`, 50 SQLi, search `40x96`, beam `8`:
    - baseline targeted recall `0.8200`, p10 `0.4673`, success `0.1800`
    - hardneg targeted recall `0.7400`, p10 `0.4570`, success `0.2600`
  - `textcnn`, 100 SQLi, search `20x48`, beam `5`:
    - baseline targeted recall `0.1300`, p10 `0.0490`, success `0.8700`
    - hardneg targeted recall `0.1300`, p10 `0.0257`, success `0.8700`
  - `bilstm`, 100 SQLi, search `20x48`, beam `5`:
    - baseline targeted recall `0.3200`, p10 `0.1003`, success `0.6800`
    - hardneg targeted recall `0.0500`, p10 `0.0276`, success `0.9500`
- 当前判断：
  - 仅加入真实 benign hard negatives 不是鲁棒增强，反而可能削弱“SQL-like surface => attack”的捷径，使变形攻击更容易把恶意样本推向 benign 区域。
  - 这支持后续方法价值：需要同时加入变形恶意正样本 / paired consistency / canonical alignment，而不是只做 hard-negative benign。

## 2026-04-13 实验一正式收敛版

- 实验目的：
  - 验证未经过变形鲁棒训练的 SQLi 检测器，在面对 WAF-A-MoLE 式 SQL 级语义保持变形时是否出现性能退化。
- 正式表述：
  - 使用“基于 WAF-A-MoLE 官方 SQL 变形算子的目标导向鲁棒性测试”。
  - 不把变形方法本身写成本文方法贡献。
  - 变形算子来源于 WAF-A-MoLE 官方 `SqlFuzzer.strategies`。
  - 搜索与实验管线由本文实现，因此不写成“完整复刻 WAF-A-MoLE”，而写成“基于 WAF-A-MoLE 算子的有限预算目标导向搜索”。
- 官方算子：
  - `spaces_to_comments`
  - `random_case`
  - `swap_keywords`
  - `swap_int_repr`
  - `spaces_to_whitespaces_alternatives`
  - `comment_rewriting`
  - `change_tautologies`
  - `logical_invariant`
  - `reset_inline_comments`
- 目标导向搜索逻辑：
  - 先用干净训练集训练普通检测器。
  - 对测试集 SQLi 样本生成多轮候选变形。
  - 每轮用当前检测器输出的 SQLi 概率给候选排序。
  - 保留 SQLi 概率最低的若干候选继续搜索。
  - 若候选 SQLi 概率低于阈值 `0.5`，提前停止并记为攻击成功。
  - 若固定预算内未低于阈值，则记为攻击失败，保留当前最低概率候选用于评估。
- 当前固定预算建议：
  - `search_steps = 20`
  - `candidates_per_state = 48`
  - `beam_size = 5`
  - `threshold = 0.5`
  - `max_chars = 640`
  - `early_stop = True`
- 正式数据口径：
  - 主数据集：`SQLiV3_clean`
  - 训练：仅干净样本，不加入变形样本。
  - 测试：同一批 SQLi 样本分别构造 `clean_attack_matched` 与 `targeted_official_wafamole`。
  - 对照目标是“变形前后同一批攻击样本的检测表现差异”。
- 正式 baseline 建议：
  - `word_svc`：传统稀疏特征强基线，保留但不要求被大幅打穿。
  - `textcnn + lowercase`：深度文本分类基线，使用 lowercase 避免只被大小写扰动打穿。
  - `bilstm + lowercase`：序列模型基线，使用 lowercase。
  - `codebert`：预训练代码模型基线，因目标搜索很慢，建议单独低预算或少 seed 跑。
- 核心指标：
  - `Recall`
  - `F1`
  - `Attack Success Rate`
  - `p10_sqli_prob`
  - `mean_queries`
- seed=11 小规模探针（非正式结果，每类 100 条）：
  - `word_svc`: clean recall `1.0000` -> mutated recall `0.6600`, attack success `0.3400`, p10 `0.4361`
  - `textcnn`: clean recall `1.0000` -> mutated recall `0.1100`, attack success `0.8900`, p10 `0.0725`
  - `bilstm`: clean recall `1.0000` -> mutated recall `0.2600`, attack success `0.7400`, p10 `0.0246`
- seed=11 lowercase 公平性探针（非正式结果，每类 100 条）：
  - `textcnn + lowercase`: clean recall `1.0000` -> mutated recall `0.1800`, attack success `0.8200`, p10 `0.2350`
  - `bilstm + lowercase`: clean recall `1.0000` -> mutated recall `0.5000`, attack success `0.5000`, p10 `0.1532`
- 当前结论：
  - 第一个假设已被初步验证：未经过变形鲁棒训练的检测器在 WAF-A-MoLE 式官方语义保持变形下会出现性能退化。
  - 该结论目前是探针级证据，正式论文需要多 seed 与更大样本规模确认。
  - `word_svc` 表现相对更稳，应作为传统强基线/边界对照，而不是强行写成“大面积被打穿”。
  - 探索中过的 URL/percent encoding、自定义 `wafamole_style` 扩展、real benign hard negative 不进入实验一正式主线。
- 已落地文件：
  - `experiments/formal/targeted_sql_mutation.py` 支持 `official_wafamole`
  - `experiments/formal/run_experiment1_targeted_attack.py` 支持 `--operator-set official_wafamole`
  - `run_experiment1_targeted_attack.py` 已新增 partial 输出：`<output>.partial.json`
- 运行注意：
  - 不要一次性把 `10 seeds * word_svc/textcnn/bilstm/codebert * 高预算目标搜索` 全塞进一个命令。
  - 经典模型与 CodeBERT 应拆开跑。
  - CodeBERT 的慢点主要来自目标搜索阶段大量 `predict_proba` 查询，不是普通训练本身。
