#!/bin/bash
module load python/3.6-conda5.2
module load cuda/9.2.88
cd /users/PAS1421/osu10534/HPDL/labs/HPDL_labs/Lab1/ImageNet

export KMP_AFFINITY=granularity=fine,compact,1,0
#export OMP_NUM_THREADS=8
#python train_imagenet_cpu.py 'ResNet50_v2' 16 |& tee output/cpu_8_resnet
#python train_imagenet_cpu.py 'VGG16' 16 |& tee output/cpu_8_vgg

export OMP_NUM_THREADS=16
python train_imagenet_cpu.py 'ResNet50_v2' 16 |& tee output/cpu_16_resnet
python train_imagenet_cpu.py 'VGG16' 16 |& tee output/cpu_16_vgg

export OMP_NUM_THREADS=28
python train_imagenet_cpu.py 'ResNet50_v2' 16 |& tee output/cpu_28_resnet
python train_imagenet_cpu.py 'VGG16' 16 |& tee output/cpu_28_vgg
