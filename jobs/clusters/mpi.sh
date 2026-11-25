# MPI-INF: A40, L40, A100, H100, RTX 8000; 24 h limit; one partition per job, so the fastest of several is asked for. Notes: mpi-inf.md
GRAY_CODE=$HOME/projects/gray
GRAY_DATA=/scratch/inf0/user/enazaren/gray
GRAY_PYTHON=$HOME/projects/STEER/models/ActionPlan-Code/venv/bin/python
GRAY_SBATCH="--gres=gpu:1 -c 16 --mem=100G"
GRAY_PARTITIONS="gpu-a40 gpu-l40 gpu-a100 gpu-h100"
GRAY_ENV=""
GRAY_WORKERS=12
# Data here: nothing yet (25 Nov 2026). The scratch file system is shared and nearly full; keep it to the small sets.
