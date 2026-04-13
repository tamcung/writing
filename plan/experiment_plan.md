# 正式实验方案

## 一、实验目标

本组正式实验只回答三个问题：

1. 现有 SQL 注入检测器在语义保持变形和外部分布偏移下是否会显著失稳。
2. `pair_proj_ce` 是否比普通监督训练和普通配对增强更鲁棒。
3. 鲁棒性提升究竟来自“看到了更多变形样本”，还是来自“显式的表示学习约束”。

## 二、数据集收敛

### 1. 正文主训练集：`SQLiV3_clean`

- 文件：`data/raw/SQLiV3_clean.json`
- 角色：主训练集 + 同分布测试集
- 保留字段：payload 文本 `pattern` 与标签 `type`
- 正式版本：`formal_v3`
- 正式规则：
  - `max_len = 320`
  - 不再剔除完整 SQL-like 样本
  - 去除空文本
  - 仅保留 `valid` 和 `sqli`
  - 以文本去重；若同一文本出现冲突标签，则直接丢弃
  - 统一按参数 / payload / pattern 级字符串处理
- 当前过滤后规模：
  - 总计 `29895`
  - benign `19494`
  - sqli `10401`
- 选择原因：
  - 干净、稳定，便于方法开发与多 seed 对比
  - 与 `SQLiV5` / `WAF-A-MoLE` 路线天然衔接
  - 适合作为全文统一主训练源

### 2. 正文主变形基准：`targeted_official_wafamole`

- 角色：主语义保持变形鲁棒性测试
- formal 构造模块：
  - `experiments/formal/targeted_sql_mutation.py`
  - `experiments/formal/run_experiment1_targeted_attack.py`
  - `experiments/formal/run_experiment2_pair_training_targeted.py`
- 变形算子来源：WAF-A-MoLE 官方 `SqlFuzzer.strategies`
- 测试协议：
  - 对测试集 SQLi 样本执行有限预算目标导向搜索
  - 每轮生成多个官方算子变形候选
  - 用当前检测器的 SQLi 概率排序
  - 保留低置信度候选继续搜索
  - 若 SQLi 概率低于 `0.5`，记为攻击成功
- 选择原因：
  - 直接对应本文“语义保持变形绕过 ML 型 SQLi 检测器”的核心问题
  - 比静态随机变形更能暴露模型弱点
  - 算子来源有文献和开源工具背书，避免把自造弱变形作为主证据

### 3. 真实变形补充基准：`SQLiV5`

- 文件：`data/raw/SQLiV5.json`
- 角色：WAF-A-MoLE 风格真实变形测试
- 使用方式：
  - 不作为主训练集
  - 作为对 `targeted_official_wafamole` 的补充验证
  - 正文中强调其与 `WAF-A-MoLE` 路线的对应关系
- 选择原因：
  - 直接对应“语义保持变形绕过”相关文献
  - 能证明方法对真实攻击风格变形的有效性
- 风险说明：
  - 其分布与 `SQLiV3` 关系较近
  - 不宜作为唯一变形证据
- 当前审计结果：
  - 过滤后总计 `36740`
  - benign `19494`
  - sqli `17246`
  - 与 `SQLiV3_clean` exact overlap `29895`
  - 未见于 `SQLiV3_clean` 的新增 SQLi `6845`
- 正式口径：
  - 不报告“整个 SQLiV5”上的结果作为主证据
  - 优先报告其新增 SQLi 部分，作为真实变形补充基准

### 4. 正文主外部集：`ModSec-Learn-cleaned`

- 原始来源：`https://github.com/pralab/modsec-learn-dataset`
- 角色：主跨数据集 / 外部分布泛化测试
- 清洗原则：
  - 恶意部分去掉 `sqli_kaggle` 来源，避免与 `SQLiV3_clean` 的同源性过强
  - 再删掉与 `SQLiV3_clean` 的 exact overlap
  - 良性部分保留 `openappsec`
  - 恶意部分优先保留 `sqlmap`、`httpparams`、`openappsec`
- 选择原因：
  - 公开可获取
  - 来源混合，能较好体现跨源外部泛化
  - 比单一数据源更接近真实 WAF 场景
- 当前清洗后规模：
  - 总计 `50724`
  - benign `34798`
  - sqli `15926`
  - 与 `SQLiV3_clean` exact overlap `0`
- 来源保留情况：
  - benign：`openappsec`
  - malicious：`openappsec`、`httpparams`、`sqlmap`
  - excluded：`sqli_kaggle`

### 5. 补充外部集：`web-attacks-long`

- 文件：`data/raw/web_attacks_long/test.csv`
- 角色：补充外部分布验证
- 处理规则：
  - 仅保留 `normal` 和 `SQLi`
  - 去除空文本
  - 去除长度超过 `320` 字符的样本
  - 按文本去重
  - 去掉与 `SQLiV3_clean` 完全相同的样本
  - 按 seed 做类别平衡采样
- 当前过滤后规模：
  - 总计 `3477`
  - benign `942`
  - sqli `2535`
  - 与 `SQLiV3_clean` 的 overlap 审计以 `formal_v3/manifest.json` 为准
- 选择原因：
  - 与主训练源差异较大
  - 可作为第二外部证据增强说服力

### 6. 不进入正文主实验的数据集

- `HttpParamsDataset`
  - 原因：保留作相关工作对齐与讨论，不作为本论文最终主数据方案
- `ai-waf-dataset`
  - 原因：是多攻击类型请求级数据，不是纯 SQLi payload 二分类

## 三、样本处理与配对规范

### 0. 正式输入层

- 从现在开始，正式实验统一读取：
  - `data/processed/formal_v3/datasets/*.json`
- 不再直接从 `data/raw/*` 或历史试验脚本中读取
- 当前统一入口脚本：
  - `experiments/formal/prepare_raw_datasets.py`
- 当前统一清洗模块：
  - `experiments/formal/raw_processing.py`
- 当前统一变形模块：
  - `experiments/formal/semantic_mutation.py`
  - `experiments/formal/targeted_sql_mutation.py`
- 当前配对训练集：
  - `data/derived/formal_v3/experiment2/pairs/seed_*/train_pairs.json`
  - 构造脚本：`experiments/formal/build_experiment2_pairs.py`
- 当前统一 manifest：
  - `data/processed/formal_v3/manifest.json`

### 1. 标签统一

- benign / normal / valid 统一映射为 `0`
- sqli / SQLi 统一映射为 `1`

### 2. 数据集与实验视图区分

- `sqliv3_clean`、`sqliv5`、`modsec_learn_cleaned`、`web_attacks_long_test`
  - 这些是 formal 输入层里的基础数据集
- `sqliv3_clean_holdout`
  - 不是独立数据集
  - 而是从 `sqliv3_clean` 按 seed 做分层抽样后切出的同分布测试视图
- `targeted_official_wafamole`
  - 不是独立数据集
  - 而是基于 `sqliv3_clean_holdout` 中的恶意样本，再通过官方 WAF-A-MoLE 算子和目标导向搜索得到的变形测试视图

### 3. 训练/测试划分

- 基于 `SQLiV3_clean` 做按类采样切分
- 基于 processed 版本的 `sqliv3_clean.json` 做按类采样切分
- 每个 seed 固定：
  - 训练集：每类 `1000`
  - 验证集：每类 `200`
  - 干净测试集：每类 `1000`
- 说明：
  - 也便于跨 backbone 的公平比较

### 4. 配对样本定义

- `x_canon`
  - 训练集中原始 payload
- `x_raw_mut`
  - 对 `x_canon` 施加语义保持变形后的样本
- 恶意样本对：
  - 使用 WAF-A-MoLE 官方 SQL 变形算子随机构造
  - 正式默认 `rounds=5`
- 良性样本对：
  - 不做 SQLi 语义变形
  - 使用 nuisance 级表面扰动，例如大小写与空白表示变化
  - 目的是防止模型将“发生变形”误学成“恶意”

### 5. 变形策略使用原则

- 训练时：
  - `clean_ce` 仅使用干净样本
  - `pair_ce / pair_proj_ce / pair_canonical` 使用官方 WAF-A-MoLE 算子的随机变形样本构造训练对
- 测试时：
  - 主变形测试使用官方 WAF-A-MoLE 算子的目标导向搜索
  - 真实变形补充测试使用 `SQLiV5`
- 这对应论文里的核心概念：
  - 不是只比较静态增强样本
  - 而是比较有限预算目标导向攻击下的鲁棒性

## 四、正式实验矩阵

### 实验 1：问题存在性验证

- 目的：证明同分布高分并不代表鲁棒
- 对象：
  - `word-SVC`
  - `BiLSTM`
  - `TextCNN`
  - `CodeBERT`
- 方法：
  - `clean_ce`
- 测试视图：
  - `sqliv3_clean_holdout`
  - `targeted_official_wafamole`
- 正文产出：
  - 一张“变形前 vs 变形后”的问题存在性主表

### 实验 2：主方法有效性验证

- 目的：证明 `pair_proj_ce` 优于普通监督和普通配对增强
- 主角 backbone：
  - `CodeBERT`
- 强对照：
  - `TextCNN`
- 补充：
  - `BiLSTM`
- 对比方法：
  - `clean_ce`
  - `pair_ce`
  - `pair_proj_ce`
  - `pair_canonical` 作为消融扩展
- 测试视图：
  - `clean_attack_matched`
  - `targeted_official_wafamole`
- 正文产出：
  - 一张主方法对比表
  - 一张目标导向攻击成功率对比表

### 实验 3：机制与消融实验

- 目的：解释主增益来自哪里
- 主对象：
  - `CodeBERT`
- 对比：
  - `pair_ce` vs `pair_proj_ce`
  - `pair_proj_ce` vs `pair_canonical`
- 分析点：
  - `F1`
  - `Recall`
  - `P10 SQLi probability`
  - `Attack Success Rate`
  - `mean_queries`
- 正文产出：
  - 一张消融表
  - 一张 hardest samples 置信度对比图
  - 一张 `SQLiV5` 补充结果表

### 实验 4：架构泛化对比

- 目的：说明 `pair_proj_ce` 的收益是否依赖 backbone
- 对象：
  - `BiLSTM`
  - `TextCNN`
  - `CodeBERT`
- 对比方法：
  - `clean_ce`
  - `pair_ce`
  - `pair_proj_ce`
- 说明：
  - `word-SVC` 只作为经典基线，不进入这一张方法消融表

### 实验 5：补充外部验证

- 目的：增强跨数据集说服力
- 数据集：
  - `web-attacks-long`
- 对象：
  - `CodeBERT`
  - `TextCNN`
- 方法：
  - `clean_ce`
  - `pair_ce`
  - `pair_proj_ce`

### 实验 6：系统验证实验

- 目的：支撑“研究与实现”
- 最终模型：
  - `CodeBERT + pair_proj_ce`
- 内容：
  - 单条 payload 检测
  - 批量检测
  - 置信度输出
  - 典型成功样例与失败样例展示

## 五、正文与附录的边界

### 正文保留

- `SQLiV3_clean`
- `targeted_official_wafamole`
- `SQLiV5`
- `ModSec-Learn-cleaned`
- `web-attacks-long`
- `CodeBERT / TextCNN / BiLSTM / word-SVC`
- `clean_ce / pair_ce / pair_proj_ce`
- `pair_canonical` 仅作为消融扩展

### 附录或补充材料

- `CharCNN`
- 早期自定义 semantic family holdout
- `boolean_equivalent` 单独结果
- 自定义 `wafamole_style` 扩展
- `ai-waf-dataset`
- 各种历史试验性对比

## 六、统计与报告规范

### 1. 随机种子

- 主结果统一使用 `10 seeds`
- seed 集合固定，全文保持一致

### 2. 报告方式

- 每个指标报告：
  - mean
  - std
  - min / max（可放附录）
- 正文主指标：
  - `F1`
  - `Recall`
  - `Precision`
  - `P10 SQLi probability`
  - `Attack Success Rate`

### 3. 统计检验

- 对同一 seed 下的模型差值做配对统计
- 默认使用：
  - `Wilcoxon signed-rank test`
- 同时报告：
  - mean difference
  - effect size 或至少标准差
- 多重比较时使用保守校正

## 七、执行顺序

1. 固定 seed 列表与统一数据过滤规则
2. 完成 `clean_ce` 在 `targeted_official_wafamole` 下的问题存在性实验
3. 重跑 `CodeBERT / TextCNN / BiLSTM` 的 `clean_ce / pair_ce / pair_proj_ce / pair_canonical`
4. 单独补 `SQLiV5`、`ModSec-Learn-cleaned` 与 `web-attacks-long` 外部验证
5. 汇总成正文主表与附录表
6. 最后基于最佳 `CodeBERT + pair_proj_ce` 做系统展示
