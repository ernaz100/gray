# Raven (MPCDF): 4 A100 40 GB per node, 24 h limit. Same account layout as Viper, but its own /ptmp. Notes: mpcdf-raven.md
GRAY_CODE=/u/enazaren/projects/gray
GRAY_DATA=/ptmp/enazaren/gray
GRAY_PYTHON=/u/enazaren/envs/torch/bin/python
GRAY_SBATCH="--constraint=gpu --gres=gpu:a100:1 -c 18 --mem=125000"
GRAY_PARTITIONS="default"
GRAY_ENV=""
GRAY_WORKERS=12
# Data here: nothing yet (25 Nov 2026). See README.md in this folder for what to copy.
