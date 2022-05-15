#!/bin/sh

#SBATCH -A m3443
#SBATCH -J ctd
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -c 10
#SBATCH -t 4:00:00
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=a.akram@gsi.de
#SBATCH -q regular

mkdir -p logs

conda activate exatrkx
module load cgpu

srun --pty /bin/bash -l traintrack configs/pipeline_quickstart.yaml
