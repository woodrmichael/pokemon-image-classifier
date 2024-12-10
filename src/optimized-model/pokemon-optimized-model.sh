#!/bin/bash

################################################################################
#
# Submit file for a batch job on Rosie.
#
# To submit your job, run 'sbatch <jobfile>'
# To view your jobs in the Slurm queue, run 'squeue -l -u <your_username>'
# To view details of a running job, run 'scontrol show jobid -d <jobid>'
# To cancel a job, run 'scancel <jobid>'
#
# See the manpages for salloc, srun, sbatch, squeue, scontrol, and scancel
# for more information or read the Slurm docs online: https://slurm.schedmd.com
#
################################################################################


# You _must_ specify the partition. Rosie's default is the 'teaching'
# partition for interactive nodes.  Another option is the 'batch' partition.
#SBATCH --partition=dgx

# The number of nodes to request
#SBATCH --nodes=1

# The number of GPUs to request
#SBATCH --gpus=8

# The number of CPUs to request per GPU
#SBATCH --cpus-per-gpu=9

# The error file to write to
#SBATCH --error='sbatcherrorfile.out'

# Kill the job if it takes longer than the specified time
# format: <days>-<hours>:<minutes>
#SBATCH --time=0-3:0

# Uncomment the following lines and update them if you want to get an email
# SBATCH --mail-type=ALL
# SBATCH --mail-user=yourusernamehere@msoe.edu


####
#
# Here's the actual job code.
# Note: You need to make sure that you execute this from the directory that
# your python file is located in OR provide an absolute path.
#
####

# Path to container
container="/data/containers/msoe-horovod-tf1.sif "

# Command to run inside container
command="horovodrun -np 8 -H localhost:8 python optimized-model.py --data /home/woodm/CSC2611/pokemon-image-classifier/stratified-data --batch_size 16 --epochs 10 --main_dir /home/woodm/CSC2611/pokemon-image-classifier/src/optimized-model --augment_data true --fine_tune true"

# Execute singularity container on node.
singularity exec --nv -B /data:/data ${container} /usr/local/bin/nvidia_entrypoint.sh ${command}
