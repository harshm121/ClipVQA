#SBATCH --job-name=clip-vqa
#SBATCH --output=logs.out
#SBATCH --error=logs.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --partition=short
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
set -x
echo "Run starting"
srun python appendModelGetResults.py
echo "Run complete"
