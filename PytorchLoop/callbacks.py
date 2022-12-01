import torch
from torch.utils.tensorboard.writer import SummaryWriter


class Callback():
    def __init__(self): pass
    def on_train_begin(self): pass
    def on_train_end(self): pass
    def on_epoch_begin(self): pass
    def on_epoch_end(self): pass
    def on_batch_begin(self): pass
    def on_batch_end(self): pass
    def on_loss_begin(self): pass
    def on_loss_end(self): pass
    def on_step_begin(self): pass
    def on_step_end(self): pass


class TensorboardHandler(Callback):

    def __init__(self, log_dir, write_graph=True, write_images=False ,**kwargs):
        self.log_dir = log_dir
        self.write_graph = write_graph
        self.write_images = write_images
        
    def on_train_begin(self, trainLoader, model):
        self.writer = SummaryWriter(self.log_dir)
        if self.write_graph:
            self.writer.add_graph(model, next(iter(trainLoader))[0].cuda())

    def on_train_end(self):
        self.writer.close()

    def on_epoch_end(self, epoch, metrics):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, epoch)

    def on_batch_end(self, batch, metrics):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, batch)