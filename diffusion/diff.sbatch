#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=diffucolor            # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=output.log   # Log-file (important!)
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=12G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1               # Number of GPU's
#SBATCH --time=14:00:00              # Max execution time
#SBATCH --partition=a5000
#


# 2. Activate your Anaconda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/ygeurts/anaconda3/envs/deep_learning

# 3. Use an ABSOLUTE PATH to navigate to your project
cd /home/ygeurts/DiffuColor/diffusion

# 4. Run your Pythonscript
python train.py --epochs 50 --checkpoint checkpoints/h6plwk9v/diffusion_epoch60.pt --drop_prob 0.15 --guidance_scale 2.0

