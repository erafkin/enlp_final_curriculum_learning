#!/bin/bash
#SBATCH --job-name="rafkin_curriculum_learning"
#SBATCH --nodes=1
#SBATCH --partition=base
#SBATCH --output="%x.o%j"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=60:00:00
#SBATCH --mail-user=epr41@georgetown.edu
#SBATCH --mail-type=END,FAIL

module load cuda/12.5

module load gcc/11.4.0
 
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 ./scripts/train_pipeline.py