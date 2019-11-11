import sys
import mxnet as mx
import numpy as np
import os, time, shutil
import horovod.mxnet as hvd

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs
from gluoncv.model_zoo import get_model
from gluoncv.data.sampler import SplitSampler
import math

import socket

if (len(sys.argv) < 6):
    print('please input both model name and batch size')
    sys.exit()

model_name = sys.argv[1]
per_device_batch_size = int(sys.argv[2])
is_gpu_enabled = sys.argv[3] == 'True'
num_nodes = int(sys.argv[4])
ppn = int(sys.argv[5])

hvd.init()

ctx = [mx.gpu(hvd.local_rank())] if is_gpu_enabled else [mx.cpu(hvd.local_rank())]

print("----")
print(ctx)
print(is_gpu_enabled)
print(socket.gethostname())
print("----")

classes = 23

epochs = 10
lr = 0.001

momentum = 0.9
wd = 0.0001

lr_factor = 0.75
lr_steps = [10, 20, 30, np.inf]

num_gpus = 1
num_workers = hvd.size()

batch_size = per_device_batch_size * max(num_gpus, 1)

# Data Augmentation
jitter_param = 0.4
lighting_param = 0.1

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# With the data augmentation functions, we can define our data loaders:
# Change this:
# path = './tiny-imagenet-200'
path = './minc-2500-tiny'
train_path = os.path.join(path, 'train')
val_path = os.path.join(path, 'val')
test_path = os.path.join(path, 'test')

train_data_size = 115

train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
    batch_size=batch_size,
    sampler=SplitSampler(train_data_size, num_workers, hvd.rank()))

val_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(val_path).transform_first(transform_test),
    batch_size=batch_size, shuffle=False)

test_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(test_path).transform_first(transform_test),
    batch_size=batch_size, shuffle=False)

# Model and Trainer

finetune_net = get_model(model_name, pretrained=True)
with finetune_net.name_scope():
    finetune_net.output = nn.Dense(classes)
finetune_net.output.initialize(init.Xavier(), ctx=ctx)
finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()

optimizer_parameters = {'learning_rate': lr, 'wd': wd}
optimizer = mx.optimizer.NAG(momentum=momentum, **optimizer_parameters)

model_parameters = finetune_net.collect_params()
if model_parameters is not None:
    hvd.broadcast_parameters(model_parameters, root_rank=0)

trainer = hvd.DistributedTrainer(model_parameters, optimizer)

metric = mx.metric.Accuracy()
L = gluon.loss.SoftmaxCrossEntropyLoss()


# we define a evaluation function for validation and testing
def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    return metric.get()


# Training Loop
lr_counter = 0
num_batch = len(train_data)

for epoch in range(epochs):
    if epoch == lr_steps[lr_counter]:
        trainer.set_learning_rate(trainer.learning_rate * lr_factor)
        lr_counter += 1

    tic = time.time()
    train_loss = 0
    metric.reset()

    for i, batch in enumerate(train_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        with ag.record():
            outputs = [finetune_net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        for l in loss:
            l.backward()

        trainer.step(batch_size)
        train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

        metric.update(label, outputs)

    _, train_acc = metric.get()
    train_loss /= num_batch

    _, val_acc = test(finetune_net, val_data, ctx)

    print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f' %
          (epoch, train_acc, train_loss, val_acc, time.time() - tic))

_, test_acc = test(finetune_net, test_data, ctx)
print('[Finished] Test-acc: %.3f' % (test_acc))
