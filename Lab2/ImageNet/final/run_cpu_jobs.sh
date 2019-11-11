#!/bin/bash

#module load python/3.6-conda5.2
#module load cuda/9.2.88
#module load mvapich2/2.3.2
#source activate mxnet_lab2

cd /users/PAS1588/srander/lab2/Lab2/ImageNet/final

#mpirun -np $((${nodes} * 28)) -hostfile $PBS_NODEFILE python common.py ${model_name} ${batch} ${gpu_enabled}
mpirun -np $((${nodes} * ${ppn})) -ppn ${ppn} python common.py ${model_name} ${batch} ${gpu_enabled} ${nodes} ${ppn}
