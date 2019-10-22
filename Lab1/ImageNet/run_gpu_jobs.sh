#!/bin/bash
module load python/3.6-conda5.2
module load cuda/9.2.88

cd /users/PAS1421/osu10534/HPDL/labs/HPDL_labs/Lab1/ImageNet

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
#python train_imagenet_gpu.py 'ResNet50_v2' 16 |& tee output/gpu_16_resnet
#python train_imagenet_gpu.py 'VGG16' 16 |& tee output/gpu_16_vgg

#python train_imagenet_gpu.py 'ResNet50_v2' 32 |& tee output/gpu_32_resnet
#python train_imagenet_gpu.py 'VGG16' 32 |& tee output/gpu_32_vgg

python train_imagenet_gpu.py 'ResNet50_v2' 64 |& tee output/gpu_64_resnet
python train_imagenet_gpu.py 'VGG16' 64 |& tee output/gpu_64_vgg
