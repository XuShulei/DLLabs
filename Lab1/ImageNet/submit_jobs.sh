#!/bin/bash

qsub -V -l nodes=1:ppn=28 -l walltime=4:00:00 -A PZS0622 ./run_cpu_jobs.sh

qsub -V -l nodes=1:ppn=8:gpus=1 -l walltime=4:00:00 -A PZS0622 ./run_gpu_jobs.sh
