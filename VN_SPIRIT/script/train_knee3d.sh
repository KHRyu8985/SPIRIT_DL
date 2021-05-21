#!/bin/bash
#SBATCH -J subeen_test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --exclusive
#SBATCH -o ./%x_%u_%j.out
#SBATCH -e ./%x_%u_%j.err
#SBATCH --gres=gpu:volta:1

# -J: job name
# -n: number of tasks (processes/workers)
# -N: number of nodes
# -c: cores per task (maximum 40)
# --exclusive: exclusive node
# --gres=gpu:volta: number of gpus
# %(specifiers): https://slurm.schedmd.com/sbatch.html
#       ex) %x: job name, %u: user name, %j: job id number 

# chmod a+x
# Run the script

python train_knee3d.py --verbose --loss l1
echo "Finished"

