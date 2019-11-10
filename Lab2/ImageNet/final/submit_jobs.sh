#!/bin/bash

qsub -V -l nodes=4:ppn=28 -l walltime=4:00:00 -A PAS1588 -v gpu_enabled=False,batch=64,model_name=ResNet50_v2,nodes=4 ./run_gpu_jobs.sh

qsub -V -l nodes=2:ppn=8:gpus=1 -l walltime=4:00:00 -A PAS1588 -v gpu_enabled=True,batch=64,model_name=ResNet50_v2,nodes=2 ./run_gpu_jobs.sh
