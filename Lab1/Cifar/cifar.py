import numpy as np
import mxnet as mx


from LossRecordHandler import LossRecordHandler
from mxnet import gluon

from gluoncv.model_zoo import get_model

import os
import _pickle as cPickle
from mxnet.gluon.contrib.estimator import estimator
from mxnet.gluon.contrib.estimator.event_handler import CheckpointHandler


def extract_images_and_labels(path, file):
    path = os.path.abspath(path + file)
    f = open(path, 'rb')
    dictionary = cPickle.load(f, encoding="latin1")
    images = dictionary['data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dictionary['labels']
    image_array = mx.nd.array(images)
    label_array = mx.nd.array(labels)
    return image_array, label_array


def build_training_set(path):
    _training_data = []
    _training_label = []
    for f in ("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"):
        image_array, label_array = extract_images_and_labels(path, f)
        if len(_training_data) == 0:
            _training_data = image_array
            _training_label = label_array
        else:
            _training_data = mx.nd.concatenate([_training_data, image_array])
            _training_label = mx.nd.concatenate([_training_label, label_array])
    return _training_data, _training_label


def build_estimator(_ctx):
    _net = get_model('cifar_resnet20_v1', classes=10)
    _net.initialize(mx.init.Xavier(), ctx=ctx)
    _loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

    _optimizer = 'nag'
    _optimizer_params = {'learning_rate': 0.1, 'wd': 0.0001, 'momentum': 0.9}
    _trainer = gluon.Trainer(_net.collect_params(), _optimizer, _optimizer_params)

    train_acc = mx.metric.Accuracy()
    _est = estimator.Estimator(net=_net,
                               loss=_loss_fn,
                               metrics=train_acc,
                               trainer=_trainer,
                               context=_ctx)

    return _est


cifar_folder_path = "./cifar-10-batches-py/"
batch = 128

training_data, training_label = build_training_set(cifar_folder_path)
train_iter = mx.io.NDArrayIter(
    data=training_data, label=training_label, batch_size=batch, shuffle=True)

valid_data, valid_label = extract_images_and_labels(cifar_folder_path, "test_batch")
valid_iter = mx.io.NDArrayIter(
    data=valid_data, label=valid_label, batch_size=batch, shuffle=True)

final_val_data = mx.gluon.data.dataset.ArrayDataset(valid_data, valid_label)
final_train_data = mx.gluon.data.dataset.ArrayDataset(training_data, training_label)

train_data_loader = gluon.data.DataLoader(final_train_data, batch_size=batch,
                                          shuffle=True)
val_data_loader = gluon.data.DataLoader(final_val_data, batch_size=batch,
                                        shuffle=False)

val_acc = mx.metric.Accuracy()
loss_record_handler = LossRecordHandler()
checkpoint_handler = CheckpointHandler(model_dir='./',
                                       monitor=val_acc,
                                       save_best=True)
# number of GPUs to use
ctx = [mx.context.cpu()]
epochs = 10
est = build_estimator(ctx)
est.fit(train_data_loader, val_data_loader, epochs=epochs, event_handlers=[])

print(loss_record_handler.loss_history)
