# B3 在 RunPod 运行手册

## 选机 & 建议

| 规模 | 骨干范围 | 单 GPU 时长（RTX 4090 / A100） | 成本预估 |
|---|---|---|---|
| 最小（必做） | TextCNN × {WAF-A-MoLE, AdvSQLi} × 5 种子 | ~2 h | 低 |
| 推荐 | TextCNN + BiLSTM × 2 算子集 × 5 种子 | ~5 h | 中 |
| 完整 | 3 骨干 × 2 算子集 × 5 种子 | ~14 h（CodeBERT 占大头） | 高 |

推荐：**TextCNN + BiLSTM 的"推荐"档**即可把 §4.4 表 4-6 的浅层骨干两行（占论文叙事核心）改为严格同批。CodeBERT 上的投影层贡献已在消融中可见（λ=0.0 CodeBERT 0.886/0.858），且 CodeBERT 对投影层略负的结论已经清晰，再补一次严格值收益不大。

## RunPod 机器设置

### 0. 选 template
- **Image**：`runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04` 或任意带 CUDA 12.4 + PyTorch ≥2.6 的镜像
- **GPU**：RTX 4090（24GB）足够，A100 更快
- **磁盘**：`Container Disk` 30 GB，`Volume` 可选（持久化数据用）
- **启动时暴露端口**：22 (SSH), 8888 (Jupyter) — 二选一即可

### 1. 首次环境设置

```bash
# 进入 RunPod 终端后
cd /workspace

# 克隆代码（用你自己的 repo URL 或 rsync 上传）
git clone https://github.com/tamcung/writing.git
cd writing

# Python 依赖
pip install -U pip
pip install torch==2.6.0 transformers==4.44.0 scipy==1.17.1 numpy pandas
pip install matplotlib scikit-learn tqdm

# 可选：如果使用 local CodeBERT，把本地预训练模型 rsync 上传到 ~/hf_cache，
# 或者启动时加 --allow-download 让 HuggingFace 自动下载
```

### 2. 上传实验输入数据

两种方式任选：

**方式 A**：直接 rsync 本地 `data/`、`experiments/*.tar.gz` 到 RunPod：

```bash
# 本地执行
rsync -avz --progress \
  experiment1_inputs.tar.gz \
  experiment2_inputs.tar.gz \
  data/ \
  root@<RUNPOD_IP>:/workspace/writing/
```

**方式 B**：在 RunPod 上解压仓库自带的 tar.gz（如果你已经 git push 上去）：

```bash
cd /workspace/writing
bash experiments/setup_exp2.sh
# 或手工：tar xzf experiment2_inputs.tar.gz
```

验证：`ls data/splits/seed_11/` 应该看到 train.json、valid.json、clean_test.json；`ls data/pairs/seed_11/` 应看到 pairs json。

### 3. 一键执行 B3（推荐档：TextCNN + BiLSTM × 2 算子集）

把以下命令贴到一个 `run_b3.sh` 里顺序执行，每段都带 `--resume`，中断可继续：

```bash
#!/usr/bin/env bash
set -e
cd /workspace/writing

# 共享 flags
COMMON="--seeds 11 22 33 44 55 \
  --require-pairs \
  --train-operator-set wafamole \
  --attack-per-class 300 \
  --search-steps 20 \
  --candidates-per-state 48 \
  --beam-size 5 \
  --attack-search-group-size 128 \
  --epochs 8 \
  --batch-size 128 \
  --max-tokens 256 \
  --device cuda \
  --resume"

# --- TextCNN × WAF-A-MoLE ---
PYTHONUNBUFFERED=1 python -u experiments/run_exp2.py \
  --backbones textcnn \
  --methods pair_proj_ce \
  --attack-operator-set wafamole \
  $COMMON \
  --output experiments/results_b3_textcnn_wafamole.json

# --- TextCNN × AdvSQLi ---
PYTHONUNBUFFERED=1 python -u experiments/run_exp2.py \
  --backbones textcnn \
  --methods pair_proj_ce \
  --attack-operator-set advsqli \
  $COMMON \
  --output experiments/results_b3_textcnn_advsqli.json

# --- BiLSTM × WAF-A-MoLE ---
PYTHONUNBUFFERED=1 python -u experiments/run_exp2.py \
  --backbones bilstm \
  --methods pair_proj_ce \
  --attack-operator-set wafamole \
  $COMMON \
  --output experiments/results_b3_bilstm_wafamole.json

# --- BiLSTM × AdvSQLi ---
PYTHONUNBUFFERED=1 python -u experiments/run_exp2.py \
  --backbones bilstm \
  --methods pair_proj_ce \
  --attack-operator-set advsqli \
  $COMMON \
  --output experiments/results_b3_bilstm_advsqli.json

echo "B3 done. 4 result JSONs in experiments/"
```

```bash
chmod +x run_b3.sh
nohup bash run_b3.sh > b3.log 2>&1 &
tail -f b3.log
```

### 4.（可选）扩展到 CodeBERT

如果时间预算允许（≈12 h 加一块 4090），追加：

```bash
PYTHONUNBUFFERED=1 python -u experiments/run_exp2.py \
  --backbones codebert \
  --methods pair_proj_ce \
  --seeds 11 22 33 44 55 \
  --require-pairs \
  --train-operator-set wafamole \
  --attack-operator-set wafamole \
  --attack-per-class 300 \
  --search-steps 20 \
  --candidates-per-state 48 \
  --beam-size 5 \
  --attack-search-group-size 128 \
  --epochs 8 \
  --batch-size 128 \
  --max-tokens 256 \
  --codebert-lr 1e-4 \
  --device cuda \
  --resume \
  --output experiments/results_b3_codebert_wafamole.json

PYTHONUNBUFFERED=1 python -u experiments/run_exp2.py \
  --backbones codebert \
  --methods pair_proj_ce \
  --seeds 11 22 33 44 55 \
  --require-pairs \
  --train-operator-set wafamole \
  --attack-operator-set advsqli \
  --attack-per-class 300 \
  --search-steps 20 \
  --candidates-per-state 48 \
  --beam-size 5 \
  --attack-search-group-size 128 \
  --epochs 8 \
  --batch-size 128 \
  --max-tokens 256 \
  --codebert-lr 1e-4 \
  --device cuda \
  --resume \
  --output experiments/results_b3_codebert_advsqli.json
```

### 5. 拉回结果

```bash
# 本地执行
rsync -avz root@<RUNPOD_IP>:/workspace/writing/experiments/results_b3_*.json \
  experiments/
```

## 结果汇总与论文更新

拿到 4（或 6）个 `results_b3_*.json` 后，每个 JSON 里有 5 个种子的 `attack_recall` 均值/标准差。我用 `experiments/summarize_results.py` 或手工读 JSON 抽出后，用以下替换更新论文：

### 预期更新位置：`chapters/04-experiment-design.md` 表 4-6

原来（近似）：

```
| TextCNN | WAF-A-MoLE | 0.615 | 0.636 | +0.021 | 0.912 (λ=1.0) | +0.276 | +0.297 |
```

改为（严格；X.XXX 为 B3 实测）：

```
| TextCNN | WAF-A-MoLE | 0.615 | X.XXX | +Y.YYY | 0.912 (λ=1.0) | +Z.ZZZ | +W.WWW |
```

其中 `X.XXX` = pair_proj_ce @ attack-per-class=300 的 5 种子均值。另外把表注"近似分解"改为"严格同批对照"，删除 300 vs 100 口径差异说明。

拿回结果后告诉我具体数值，或者把 JSON 路径贴给我，我可以读出后替换。

## 中断 / 重启

- `--resume` 会读取已有 `results_b3_*.json` 和 `.json.partial.json`，跳过已完成的 `(seed, backbone, method)` 组合
- 断线重连：`ssh` 或 `tmux attach` 回 RunPod，`tail -f b3.log` 继续观察
- 成本控制：RunPod 支持"Spot"实例，如果被抢占，`--resume` 会自动从上次进度继续

## 常见故障

- **"pair_ids 不存在"**：`setup_exp2.sh` 没跑，或 `data/pairs/seed_*/` 为空。重跑 `bash experiments/setup_exp2.sh` 或手工运行 `experiments/prepare_pairs.py`。
- **CUDA OOM（CodeBERT）**：把 `--codebert-batch-size` 从 8 改到 4 或 2；或换 A100。
- **`ValueError: Paired methods do not support backbone=word_svc`**：确保命令中 `--backbones` 不包含 word_svc。
- **数据未上传**：`data/splits/` 或 `data/pairs/` 目录为空，先做"上传实验输入数据"那一步。
