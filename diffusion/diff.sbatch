#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=diffucolor            # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=output_diff.log   # Log-file (important!)
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=12G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1               # Number of GPU's
#SBATCH --time=24:00:00              # Max execution time
#SBATCH --partition=a5000
#


# 2. Activate your Anaconda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/ygeurts/anaconda3/envs/deep_learning

# 3. Use an ABSOLUTE PATH to navigate to your project
cd /home/ygeurts/DiffuColor/diffusion

# 4. Run your Pythonscript
python train.py --epochs 100 --lr 1e_4 --batch_size 16 --drop_prob 0.15 --guidance_scale 3.0

