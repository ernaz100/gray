# Viper-GPU (MPCDF): 2 AMD MI300A per node (ROCm), 24 h limit; jobs of 1 h start soonest. Notes: docs of the workspace, mpcdf-viper.md
GRAY_CODE=/u/enazaren/projects/gray
GRAY_DATA=/ptmp/enazaren/gray
GRAY_PYTHON=/u/enazaren/envs/torch/bin/python
GRAY_SBATCH="--constraint=apu --gres=gpu:1 -c 24 --mem=110000"
GRAY_PARTITIONS="default"                    # the submit filter picks the partition from the request
GRAY_ENV="TORCH_BLAS_PREFER_HIPBLASLT=0"     # the default BLAS is very slow for our layers on ROCm
GRAY_WORKERS=12
# Data here: everything (Nymeria features and grids, moments, ground, body, hml3d, ee4d).
