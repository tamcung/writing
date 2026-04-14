# Colab Runs

## Experiment 1

```bash
cd /content
git clone https://github.com/tamcung/writing.git
cd writing
bash experiments/formal/setup_colab_experiment1.sh

PYTHONUNBUFFERED=1 python -u experiments/formal/run_experiment1_targeted_attack.py \
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
  --device cuda \
  --resume \
  --output experiments/formal/results_modsec_decoded_experiment1_classic_10seed.json
```

## Experiment 2

```bash
cd /content
git clone https://github.com/tamcung/writing.git
cd writing
bash experiments/formal/setup_colab_experiment2.sh

PYTHONUNBUFFERED=1 python -u experiments/formal/run_experiment2_pair_training_targeted.py \
  --backbones textcnn bilstm \
  --methods clean_ce pair_ce pair_proj_ce pair_canonical \
  --seeds 11 22 33 44 55 66 77 88 99 111 \
  --require-pairs \
  --train-operator-set official_wafamole \
  --attack-operator-set official_wafamole \
  --attack-per-class 300 \
  --search-steps 20 \
  --candidates-per-state 48 \
  --beam-size 5 \
  --pair-max-chars 896 \
  --max-chars 896 \
  --epochs 8 \
  --batch-size 128 \
  --max-tokens 256 \
  --lowercase \
  --device cuda \
  --resume \
  --output experiments/formal/results_modsec_decoded_experiment2_classic_10seed.json
```
