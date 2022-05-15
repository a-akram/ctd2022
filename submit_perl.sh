#!/bin/sh

#!/bin/bash

# *** 1 node, 1 task, 1 GPU ***

#SBATCH -A m3443_g                   # --account=<account>, account name. cpu: m3443, gpu: m3443_g
#SBATCH -J ctd                       # --job-name=<jobname>, job name
#SBATCH -N 1                         # --nodes=<minnodes>[-maxnodes], absent: resource satisfy -n, -c, -G
#SBATCH -G 1                         # --gpus=[type:]<number>, gpus required for this job
#SBATCH -C gpu                       # --constraint=<list>, assign "slurm feature(s)" 
#SBATCH -q regular                   # --qos=<qos>, request quality of service
#SBATCH -t 4:00:00                   # --time=<time>, request job time
#SBATCH -n 1                         # --ntasks=<number>, max task for an allocated resource
#SBATCH --ntasks-per-node=1          # --ntasks-per-node=<ntasks>, --ntasks=<number> takes precedence
#SBATCH -c 128                       # --cpus-per-task=<ncpus>, cores per task
#SBATCH --gpus-per-task=1            # --gpus-per-task=[type:]<number>, number of gpus for each task
#SBATCH --gpus-per-node=1            # --gpus-per-node=[type:]<number>, number of gpus on each node

#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=a.akram@gsi.de


export SLURM_CPU_BIND="cores"
conda activate exatrkx-gpu
srun --pty /bin/bash -l traintrack configs/pipeline_quickstart.yaml
