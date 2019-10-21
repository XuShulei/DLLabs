import mxnet as mx
from mxnet.gluon.contrib.estimator.event_handler import TrainBegin, TrainEnd, EpochEnd


class LossRecordHandler(TrainBegin, TrainEnd, EpochEnd):
    def __init__(self):
        super(LossRecordHandler, self).__init__()
        self.loss_history = {}

    def train_begin(self, estimator, *args, **kwargs):
        print("Training begin")

    def train_end(self, estimator, *args, **kwargs):
        # Print all the losses at the end of training
        print("Training ended")
        for loss_name in self.loss_history:
            for i, loss_val in enumerate(self.loss_history[loss_name]):
                print("Epoch: {}, Loss name: {}, Loss value: {}".format(i, loss_name, loss_val))

    def epoch_end(self, estimator, *args, **kwargs):
        for metric in estimator.train_metrics:
            # look for train Loss in training metrics
            # we wrapped loss value as a metric to record it
            if isinstance(metric, mx.metric.Loss):
                loss_name, loss_val = metric.get()
                # append loss value for this epoch
                self.loss_history.setdefault(loss_name, []).append(loss_val)
