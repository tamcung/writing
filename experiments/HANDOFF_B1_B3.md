# B1 / B3 训练命令手册

本文件列出 B1（补齐 CodeBERT + pair_canonical 到系统）与 B3（严格隔离投影层与对齐损失贡献）两项实验命令。

## B1：CodeBERT + pair_canonical 训练 + 部署

**目的**：让系统 §5.4.2 的 `models_loaded` 从 4 升到 5、`models_available` 从 8 升到 9，消除"方法主张 pair_canonical 但系统没部署"的不一致。

```bash
PYTHONUNBUFFERED=1 python -u experiments/run_exp2.py \
  --backbones codebert \
  --methods pair_canonical \
  --seeds 11 \
  --require-pairs \
  --train-operator-set wafamole \
  --attack-operator-set wafamole \
  --attack-per-class 300 \
  --epochs 8 \
  --batch-size 128 \
  --max-tokens 256 \
  --codebert-lr 1e-4 \
  --device cuda \
  --save-checkpoints \
  --skip-attack \
  --output experiments/results_exp2_codebert_pair_canonical_ckpt.json
```

**预计**：1 GPU × 1.5 小时（RTX 4090）。

**训练后**：
1. 校验：`ls -la system/backend/storage/checkpoints/codebert_pair_canonical.pt`
2. 启动后端：`bash system/scripts/run_host_demo.sh` 或已在运行的服务会自动扫描 checkpoint 目录
3. 校验接口：
   ```bash
   curl -s localhost:8000/api/v1/health | jq '{models_loaded, models_available}'
   # 预期：{"models_loaded": 5, "models_available": 9}
   curl -s localhost:8000/api/v1/models | jq '.[] | select(.identifier | contains("codebert"))'
   # 预期：出现 codebert:pair_canonical 条目
   ```
4. **论文同步**：训练成功后把 `chapters/05-system.md` 中下列数字改掉：
   - `models_loaded=4` → `models_loaded=5`（§5.4.2 两处、表 5-5 一处）
   - `models_available=8` → `models_available=9`
   - `8 个已登记模型` → `9 个已登记模型`
   - `CodeBERT 当前接入 clean_ce 与 pair_ce 两个版本` → `三个版本（新增 pair_canonical）`
   - 删除我在 A1 里加的"codebert:pair_canonical 仍在训练补齐中"括注
5. 可选：重跑 `bash system/scripts/capture_demo_screenshots.sh` 重拍 §5 图 5-5、5-6 以显示新模型。

---

## B3：严格消融——pair_proj_ce 对照（投影层贡献单独量化）

**目的**：把 §4.4 表 4-6 当前的"近似差值"（因 attack-per-class 300 vs 100 造成不完全同批对照）升级为严格同批值。

**TextCNN 必跑**（首选骨干，训练最快）：

```bash
PYTHONUNBUFFERED=1 python -u experiments/run_exp2.py \
  --backbones textcnn \
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
  --device cuda \
  --resume \
  --output experiments/results_exp2_textcnn_proj_ablation_wafamole.json
```

**可选同样跑一遍 AdvSQLi 算子**：

```bash
PYTHONUNBUFFERED=1 python -u experiments/run_exp2.py \
  --backbones textcnn \
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
  --device cuda \
  --resume \
  --output experiments/results_exp2_textcnn_proj_ablation_advsqli.json
```

**预计**：TextCNN × 5 seeds ≈ 1 小时/算子集 × 2 = 2 GPU 小时。

**可选扩展到 BiLSTM / CodeBERT**（若有时间预算）：把 `--backbones textcnn` 改为对应骨干名；BiLSTM 约 1.5 小时、CodeBERT 约 6 小时。

**训练后**：
1. 用 `experiments/summarize_results.py` 或手工从 JSON 读取 pair_proj_ce 在两攻击集的 attacked recall mean±std。
2. **论文同步**：用实测值替换 `chapters/04-experiment-design.md` 表 4-6 的"投影层 Δ"与"合计 Δ"列；同时把表注中"近似分解"改为"严格同批对照"，删除 attack-per-class 差异说明。

---

## 核心代码已具备以下支持，**无需修改**：

- `paired_models.py` L193 `projected = method in {"pair_proj_ce", "pair_canonical"}`
- `paired_models.py` L274 `if method != "pair_canonical": return 0.0`（所以 pair_proj_ce 自动只启用投影层、不启用对齐损失）
- `paired_models.py` L424 CodeBERT 同样
- `run_exp2.py` `train_method` 把任何 non-clean_ce 路由到 `build_pair_model`，支持任意 method 名

Resume 机制（`--resume`）会在中断后继续，所以可以分段跑。
