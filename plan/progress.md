# 当前进度

## 已确认

- 论文类型：中文硕士毕业论文
- 学科方向：网络空间安全
- 最终题目：面向语义保持变形的鲁棒 SQL 注入检测方法研究与实现
- 方法线：基于配对样本的鲁棒表示学习
- 当前主候选：`pair_canonical`
- 消融方法：`pair_proj_ce`
- 正文主实验主角：`CodeBERT`
- 强对照：`TextCNN`
- 补充基线：`BiLSTM`、`word-SVC`
- 系统部分保留
- 当前正式数据口径：
  - 主训练/同分布：`modsec_learn_value_windows`
  - 主跨变形：`targeted_official_wafamole`
  - `SQLiV3/SQLiV5`：不再作为当前正式主线，只保留为历史探索和备选讨论材料
- 正式 processed 版本：
  - `formal_modsec_decoded`
  - 数据层 `max_len = 448`
  - `decode_passes = 1`
  - benign window `target_len = 96`
  - pair/search `max_chars = 896`
  - pair 默认 `mutation_rounds = 7`
  - CodeBERT 模型输入 `max_len = 512`
  - 主输入为去 key 后的 value-only length-matched windows

## 下一步

1. 基于 `formal_modsec_decoded` 跑实验一经典模型 10-seed，确认普通检测器在新主数据线下是否稳定退化。
2. 基于 `formal_modsec_decoded` 跑实验二经典模型 10-seed，确认 `pair_canonical` 的优势是否稳定。
3. 单独跑实验二的 `CodeBERT` 版本，避免和经典模型混在一个长命令里。
4. 根据新主线结果决定是否还需要把旧 `formal_v3` 作为补充实验写入正文。

## 已落地文件

- `experiments/formal/prepare_modsec_decoded_dataset.py`
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
- `experiments/formal/pair_data.py`
- `experiments/formal/build_experiment2_pairs.py`
- `experiments/formal/paired_models.py`
- `experiments/formal/run_experiment2_pair_training_targeted.py`
- `data/processed/formal_v3/manifest.json`
- `data/processed/formal_v3/datasets/*.json`
- `data/processed/formal_v3/audits/*.json`
- `data/processed/formal_v3/length_audit.json`
- `data/derived/formal_v3/experiment1/splits/*`
- `data/derived/formal_v3/experiment1/views/*`
- `data/derived/formal_v3/experiment1/audit/*`

## 2026-04-14 ModSec-decoded 重构

- 新增主数据处理脚本：
  - `experiments/formal/prepare_modsec_decoded_dataset.py`
- 新 processed 数据线：
  - `data/processed/formal_modsec_decoded/manifest.json`
  - `data/processed/formal_modsec_decoded/datasets/modsec_learn_decoded.json`
  - `data/processed/formal_modsec_decoded/datasets/modsec_learn_value_windows.json`
- 当前规模：
  - 主数据集 `modsec_learn_value_windows`
  - total `70267`
  - benign `50721`
  - SQLi `19546`
  - 数据层 `max_len = 448`
  - `decode_passes = 1`
  - benign window min length `32`
  - benign window target length `96`
- 新 split：
  - `data/derived/formal_modsec_decoded/experiment1/splits`
  - 当前 split dataset：`modsec_learn_value_windows`
  - 10 seeds
  - train 每类 `3000`
  - valid 每类 `500`
  - clean_test 每类 `3000`
- 新 pair：
  - `data/derived/formal_modsec_decoded/experiment2/pairs`
  - 每个 seed `6000` 对
  - SQLi 使用 WAF-A-MoLE 官方 SQL-level operators 随机多轮变形
  - benign 使用 HTTP 参数级 URL 等价编码扰动
  - SQLi changed rate 均值约 `0.9998`
  - benign changed rate `1.0000`
  - 当前默认 `mutation_rounds = 7`
- 关键修正：
  - 旧 benign nuisance 在 ModSec-decoded 上 changed rate 只有约 `0.05`，存在 `changed => malicious` 的伪特征风险。
  - 已改为 URL 等价编码扰动，不做参数重排，因为解码后值中可能含有由 `%26` 还原出的字面量 `&`，无法可靠区分参数分隔符。
  - 原始 key-value 视图中 SQLi 几乎全部为 `p=payload`，存在 key 结构泄漏风险。
  - 已新增 value-only length-matched windows 主视图：SQLi 去除 `p=` 只保留 payload；benign 去除 key 后将多个短 value 拼成长度接近 SQLi 的窗口。
  - value 拼接使用普通空格，不使用 `[PARAM]` 专用分隔符，避免分隔符本身成为 benign 特征。
- 长度调整前单 seed probe，非正式，仅保留为调参参考：
  - `textcnn/clean_ce`, seed `11`, `attack_per_class=100`, `steps=20`, `candidates=48`, `beam=5`
  - clean F1 `1.0000`
  - targeted F1 `0.8166`
  - targeted recall `0.6900`
  - attack success `0.3100`
- 长度调整前单 seed 方法 probe，非正式，仅保留为调参参考：
  - `clean_ce`: targeted recall `0.6900`, success `0.3100`, mean drop `0.2567`
  - `pair_ce`: targeted recall `0.7000`, success `0.3000`, mean drop `0.2361`
  - `pair_proj_ce`: targeted recall `0.7100`, success `0.2900`, mean drop `0.2161`
  - `pair_canonical`: targeted recall `0.7600`, success `0.2400`, mean drop `0.1857`
- 当前判断：
  - 新数据线能跑通，且目标导向变形在长度调整前能打出一定退化。
  - `pair_canonical` 在长度调整前单 seed probe 中有优势，但幅度不大，需要在当前 `max_len=448` 正式口径下用 10-seed 结果确认。
  - 后续不再把 SQLiV3/SQLiV5 作为默认主线引用，除非明确讨论历史探索或补充实验。

## 2026-04-14 长度阈值调整

- 最终原则改为：先按恶意 payload 长度分布确定 `max_len`，再据此构造 benign length-matched windows。
- 恶意 payload（去 key、解码一次后）的字符长度分布：
  - mean `116.65`
  - p50 `99`
  - p75 `159`
  - p90 `232`
  - p95 `266`
  - p99 `331`
- 候选阈值比较后，最终选择数据层 `max_len = 448`：
  - `320` 会额外删除较多 SQLi；
  - `448` 仅删除极少数超长 SQLi，同时比 `512` 更好地压制 benign 长尾。
- 在 `max_len = 448` 固定后，再匹配 benign 窗口长度，最终选定：
  - benign window min length `32`
  - benign window target length `96`
- 当前规模：
  - raw decoded total `65190`
  - raw decoded benign `45646`
  - raw decoded SQLi `19544`
  - value-window total `70267`
  - value-window benign `50721`
  - value-window SQLi `19546`
- 当前长度分布：
  - raw decoded benign mean `165.56`, p50 `126`, p90 `378`, p95 `412`, p99 `440`, max `448`
  - raw decoded SQLi mean `118.35`, p50 `101`, p90 `234`, p95 `268`, p99 `332`, max `447`
  - value-window benign mean `113.32`, p50 `98`, p90 `219`, p95 `274`, p99 `399`, max `448`
  - value-window SQLi mean `116.38`, p50 `99`, p90 `232`, p95 `266`, p99 `331`, max `448`
- 配套参数：
  - 经典模型正式命令建议 `max_tokens = 256`
  - CodeBERT 模型输入建议 `max_len = 512`
  - 目标搜索与 pair 构造建议 `max_chars = 896`
- 已重建：
  - `data/processed/formal_modsec_decoded`
  - `data/derived/formal_modsec_decoded/experiment1/splits`
  - `data/derived/formal_modsec_decoded/experiment2/pairs`
- `448` 口径 smoke：
  - `textcnn/clean_ce` 与 `textcnn/pair_canonical` 均跑通
  - `pair_stats`: SQLi changed rate `0.9997`, benign changed rate `1.0000`

## 2026-04-14 Pair 轮数选择

- 问题：
  - 原默认 `mutation_rounds = 5` 是工程折中值，但缺乏专门的参数选择依据。
- 分析脚本：
  - `experiments/formal/analyze_pair_mutation_rounds.py`
- 分析方式：
  - 先做构造侧审计，比较 `rounds = 1/3/5/7/8` 的 changed rate、有效链长与长度膨胀。
  - 再用 `TextCNN + pair_canonical` 做三 seed 探索性下游验证，比较 targeted recall、attack success 与 mean prob drop。
- 构造侧结论：
  - `1` 轮：`sqli_changed_rate ≈ 0.985`
  - `3/5/7/8` 轮：`sqli_changed_rate = 1.000`
  - 说明从 `3` 轮起，“是否发生变化”这一目标已经饱和。
  - 长度膨胀随轮数递增：
    - `ratio_mean`: `1轮 1.043`，`3轮 1.079`，`5轮 1.118`，`7轮 1.156`，`8轮 1.176`
    - `ratio_p95`: `7轮 1.551`，`8轮 1.610`
- 下游验证结论：
  - targeted recall：
    - `1轮 0.9067`
    - `3轮 0.9100`
    - `5轮 0.9100`
    - `7轮 0.9533`
    - `8轮 0.9600`
  - attack success：
    - `1轮 0.0933`
    - `3轮 0.0900`
    - `5轮 0.0900`
    - `7轮 0.0467`
    - `8轮 0.0400`
- 决策：
  - 正式默认轮数从 `5` 调整为 `7`。
  - 选择理由：
    - `7` 相比 `5` 带来更明显的鲁棒性收益；
    - `8` 仅有边际收益，但长度膨胀继续上升；
    - 因此 `7` 更适合作为正式实验的默认 pair 构造强度。
- 注意：
  - 该轮数选择属于参数敏感性分析，当前下游验证仅为三 seed 探索性结果，不在正文中写成“显著性已证实”的结论。
  - 正式实验后续统一使用 `mutation_rounds = 7`。

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
  - `max_chars = 1024`
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
  - 第一个假设已被正式经典模型 10-seed 结果支持：未经过变形鲁棒训练的检测器在 WAF-A-MoLE 式官方语义保持变形下会出现性能退化。
  - CodeBERT 当前已有 3-seed 低预算结果，仍等待云端 full-budget 结果替换。
  - `word_svc` 表现相对更稳，应作为传统强基线/边界对照，而不是强行写成“大面积被打穿”。
  - 探索中过的 URL/percent encoding、自定义 `wafamole_style` 扩展、real benign hard negative 不进入实验一正式主线。
- 经典模型 10-seed 正式结果：
  - `word_svc`: clean recall `0.9907` -> mutated recall `0.6907`, attack success `0.3093`
  - `textcnn`: clean recall `0.9867` -> mutated recall `0.1717`, attack success `0.8283`
  - `bilstm`: clean recall `0.9873` -> mutated recall `0.5667`, attack success `0.4333`
- 已落地文件：
  - `experiments/formal/targeted_sql_mutation.py` 支持 `official_wafamole`
  - `experiments/formal/run_experiment1_targeted_attack.py` 支持 `--operator-set official_wafamole`
  - `run_experiment1_targeted_attack.py` 已新增 partial 输出：`<output>.partial.json`
- 运行注意：
  - 不要一次性把 `10 seeds * word_svc/textcnn/bilstm/codebert * 高预算目标搜索` 全塞进一个命令。
  - 经典模型与 CodeBERT 应拆开跑。
  - CodeBERT 的慢点主要来自目标搜索阶段大量 `predict_proba` 查询，不是普通训练本身。

## 2026-04-13 实验二正式 runner

- 实验目的：
  - 在实验一的问题存在性基础上，验证配对变形训练与表示投影/规范锚定是否能提升目标导向变形下的鲁棒性。
- 已新增：
  - `experiments/formal/pair_data.py`
  - `experiments/formal/build_experiment2_pairs.py`
  - `experiments/formal/paired_models.py`
  - `experiments/formal/run_experiment2_pair_training_targeted.py`
  - `experiments/formal/targeted_sql_mutation.py` 中的 `random_operator_chain`
- 已落盘：
  - `data/derived/formal_v3/experiment2/pairs/manifest.json`
  - `data/derived/formal_v3/experiment2/pairs/seed_*/train_pairs.json`
  - 每个 seed `2000` 对：benign `1000` 对，SQLi `1000` 对。
  - SQLi changed rate 范围约 `0.9900-0.9960`，平均有效链长约 `2.914-3.098`。
- 方法对比：
  - `clean_ce`：只用干净训练样本。
  - `pair_ce`：对原始样本和随机官方变形样本同时做分类交叉熵。
  - `pair_proj_ce`：在 `pair_ce` 基础上使用 projection representation 做分类。
  - `pair_canonical`：在 `pair_proj_ce` 基础上增加变形样本向原始样本表示靠近的 canonical-anchor loss。
- 训练对构造：
  - SQLi 样本：使用 WAF-A-MoLE 官方 SQL 算子随机多轮变形。
  - 说明：这里记录的是早期 `formal_v3` 历史线，当时默认 `rounds=5`；当前正式 `formal_modsec_decoded` 主线已改为 `mutation_rounds = 7`。
  - benign 样本：只做 nuisance 级大小写/空白扰动，避免让模型把“发生变形”直接学成“恶意”。
- 测试协议：
  - 与实验一相同，使用 `targeted_official_wafamole` 有限预算目标导向搜索。
  - 每个模型都接受各自的 adaptive targeted search，因此可比较 `Attack Success Rate` 与变形后 `Recall/F1/P10`。
- smoke 验证：
  - `TextCNN + seed=11 + attack_per_class=8 + steps=2 + candidates=4` 已跑通。
  - `BiLSTM + seed=11 + attack_per_class=3 + steps=1 + candidates=3` 已跑通。
  - 当时的训练变形审计显示：`rounds=3` 的 SQLi changed rate 约 `0.99`，平均有效链长约 `1.94`；`rounds=5` 的 SQLi changed rate 约 `0.99`，平均有效链长约 `2.98`，因此在旧 `formal_v3` 线上曾暂定 `5` 轮为默认强度。
  - benign nuisance changed rate 约 `0.503`。
  - smoke 只验证流程，不作为论文结果。

## 2026-04-13 实验二收敛结论

- 结果文件：
  - `experiments/formal/results_experiment2_pair_training_targeted_classic_10seed.json`
  - `results/experiment2_pair_training_targeted_summary.md`
- 实验设置：
  - backbone：`TextCNN`、`BiLSTM`
  - seed：`11, 22, 33, 44, 55, 66, 77, 88, 99, 111`
  - 方法：`clean_ce`、`pair_ce`、`pair_proj_ce`、`pair_canonical`
  - 测试：official WAF-A-MoLE 目标导向搜索，`attack_per_class=300`、`steps=20`、`candidates=48`、`beam=5`
- 主要结果：
  - TextCNN mutated recall：`clean_ce 0.1720`、`pair_ce 0.5373`、`pair_proj_ce 0.6473`、`pair_canonical 0.8527`
  - TextCNN attack success：`clean_ce 0.8280`、`pair_ce 0.4627`、`pair_proj_ce 0.3527`、`pair_canonical 0.1473`
  - BiLSTM mutated recall：`clean_ce 0.5187`、`pair_ce 0.7690`、`pair_proj_ce 0.7903`、`pair_canonical 0.8533`
  - BiLSTM attack success：`clean_ce 0.4813`、`pair_ce 0.2310`、`pair_proj_ce 0.2097`、`pair_canonical 0.1467`
- 收敛判断：
  - `pair_ce > clean_ce` 稳定成立，说明配对变形训练能缓解实验一暴露的鲁棒性退化。
  - `pair_proj_ce > pair_ce` 只在 TextCNN 上稳定成立，在 BiLSTM 上不稳定，因此不能把 `pair_proj_ce` 单独写成跨 backbone 稳定主方法。
  - `pair_canonical` 在两个 backbone 上均表现最强，尤其显著降低 attack success，应作为后续 CodeBERT 和外部验证的主候选方法。
  - 实验二经典模型论据已足够，不再追加同类经典模型实验。

## 2026-04-13 实验三外部泛化设计

- 实验目的：
  - 验证实验二的鲁棒性提升是否能迁移到真实变形样本和独立来源数据集，而不是只对 `targeted_official_wafamole` 测试协议有效。
- 训练口径：
  - 固定使用 `SQLiV3_clean` formal split 训练集。
  - 外部数据集不参与训练、验证或调参。
  - 继续使用实验二已落盘的官方 WAF-A-MoLE 配对训练样本。
- 外部测试视图：
  - `sqliv3_clean_holdout`：同分布参照。
  - `sqliv5_new_sqli_only`：真实变形恶意样本补充，只报告恶意类指标。
  - `modsec_learn_cleaned_balanced`：主跨数据集二分类验证。
  - `web_attacks_long_test_balanced`：补充跨数据集二分类验证。
- 方法口径：
  - 主比较：`clean_ce`、`pair_ce`、`pair_canonical`。
  - `pair_proj_ce` 保留为消融补充，不再作为主方法。
- 指标口径：
  - 二分类外部集报告 `F1`、`Precision`、`Recall`、`P10 SQLi probability`。
  - `sqliv5_new_sqli_only` 报告 `Recall`、`Mean SQLi probability`、`P10 SQLi probability`，不强行作为完整二分类 F1 主证据。
- 统计口径：
  - 同 seed 下比较不同方法，使用配对 Wilcoxon signed-rank test。
  - 主比较为 `pair_canonical - clean_ce` 与 `pair_canonical - pair_ce`。
- 下一步实现：
  - 新增独立外部泛化 runner，复用实验二的训练逻辑，但将评估视图换成 `SQLiV5`、`ModSec-Learn-cleaned` 与 `web-attacks-long`。

## 2026-04-14 实验三外部泛化结果

- 结果文件：
  - `experiments/formal/results_experiment3_external_generalization_classic_10seed.json`
  - `results/experiment3_external_generalization_summary.md`
- 实验设置：
  - backbone：`TextCNN`、`BiLSTM`、`CodeBERT`
  - seed：`11, 22, 33, 44, 55, 66, 77, 88, 99, 111`
  - 方法：`clean_ce`、`pair_ce`、`pair_proj_ce`、`pair_canonical`
  - 外部测试：`sqliv5_new_sqli_only`、`modsec_learn_cleaned_balanced`、`web_attacks_long_test_balanced`
- 主要结果：
  - `SQLiV5_new_sqli_only`：
    - TextCNN recall：`clean_ce 0.8554`、`pair_ce 0.9335`、`pair_proj_ce 0.9450`、`pair_canonical 0.9749`
    - CodeBERT recall：`clean_ce 0.9666`、`pair_ce 0.9776`、`pair_proj_ce 0.9845`、`pair_canonical 0.9786`
  - `ModSec-Learn-cleaned`：
    - TextCNN F1：`clean_ce 0.3518`、`pair_ce 0.4771`、`pair_proj_ce 0.5222`、`pair_canonical 0.6053`
    - BiLSTM 和 CodeBERT 上 `clean_ce` 仍是最强或接近最强，说明外部泛化收益不具备跨 backbone 普适性。
  - `web-attacks-long`：
    - CodeBERT F1：`clean_ce 0.8811`、`pair_ce 0.9466`、`pair_proj_ce 0.9551`、`pair_canonical 0.9584`
    - TextCNN/BiLSTM 的 paired 方法提升 recall 但明显降低 precision，导致 F1 下降。
- 收敛判断：
  - 实验三支持“配对规范锚定能提升部分外部场景的鲁棒性”，尤其是 TextCNN 的 SQLiV5/ModSec-Learn 与 CodeBERT 的 web-attacks-long。
  - 但实验三不支持“所有 backbone、所有外部数据集都提升”的强说法。
  - 正文应写成边界清晰的外部泛化结果：方法对语义保持变形和部分外部分布有收益，但收益依赖模型架构与目标数据分布。
