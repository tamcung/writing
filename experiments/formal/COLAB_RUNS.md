# Remote Experiment Runs

## 环境准备（每台机器只需执行一次）

```bash
cd /content  # 或远端工作目录
git clone https://github.com/tamcung/writing.git
cd writing
bash experiments/formal/setup_colab_exp1.sh   # 解压数据 + clone WAF-A-MoLE
```

---

## 实验一：clean 模型攻击脆弱性

### Run 1-A：official_wafamole 攻击
```bash
PYTHONUNBUFFERED=1 python -u experiments/formal/run_exp1.py \
  --backbones word_svc textcnn bilstm codebert \
  --seeds 11 22 33 \
  --operator-set official_wafamole \
  --attack-per-class 300 \
  --search-steps 20 \
  --candidates-per-state 48 \
  --beam-size 5 \
  --attack-search-group-size 128 \
  --epochs 8 \
  --batch-size 128 \
  --max-tokens 256 \
  --device cuda \
  --resume \
  --output experiments/formal/results_exp1_official.json
```

### Run 1-B：advsqli 攻击
```bash
PYTHONUNBUFFERED=1 python -u experiments/formal/run_exp1.py \
  --backbones word_svc textcnn bilstm codebert \
  --seeds 11 22 33 \
  --operator-set advsqli \
  --attack-per-class 300 \
  --search-steps 20 \
  --candidates-per-state 48 \
  --beam-size 5 \
  --attack-search-group-size 128 \
  --epochs 8 \
  --batch-size 128 \
  --max-tokens 256 \
  --device cuda \
  --resume \
  --output experiments/formal/results_exp1_advsqli.json
```

---

## 实验二：pair training 防御对比

先准备好数据（如果 pairs 目录不存在）：
```bash
bash experiments/formal/setup_colab_exp2.sh
```

### Run 2-A：official_wafamole 攻击
```bash
PYTHONUNBUFFERED=1 python -u experiments/formal/run_exp2.py \
  --backbones textcnn bilstm codebert \
  --methods clean_ce pair_ce pair_canonical \
  --seeds 11 22 33 \
  --require-pairs \
  --train-operator-set official_wafamole \
  --attack-operator-set official_wafamole \
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
  --output experiments/formal/results_exp2_official.json
```

### Run 2-B：advsqli 攻击（跨算子泛化）
```bash
PYTHONUNBUFFERED=1 python -u experiments/formal/run_exp2.py \
  --backbones textcnn bilstm codebert \
  --methods clean_ce pair_ce pair_canonical \
  --seeds 11 22 33 \
  --require-pairs \
  --train-operator-set official_wafamole \
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
  --output experiments/formal/results_exp2_advsqli.json
```

---

## 消融实验：consistency_weight 扫描

```bash
PYTHONUNBUFFERED=1 python -u experiments/formal/sweep_ablation.py \
  --backbone bilstm \
  --seeds 11 22 33 \
  --attack-per-class 100 \
  --search-steps 20 \
  --candidates-per-state 48 \
  --beam-size 5 \
  --attack-search-group-size 128 \
  --device cuda \
  --output experiments/formal/results_ablation_consistency_weight.json
```

---

## 注意事项

- 所有命令加了 `--resume`（实验一除消融外），中断后可继续
- CodeBERT pair training 用 `--codebert-lr 1e-4`（防止 CUDA 上梯度不稳定）
- 5 个 run 可以并行跑在不同机器上，结果文件名不同互不干扰
- 实验一的两个 run 结果需要合并分析（不同 operator-set 分开存储）
