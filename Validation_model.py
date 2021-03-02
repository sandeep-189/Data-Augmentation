import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import f1
import torch
import torch.nn as nn

class Net(pl.LightningModule):
    def __init__(self, model, num_classes, classes_weight = None, lr = 0.0001, monitor = "val_f1_score"):
        super(Net,self).__init__()
        self.model = model
        self.lr = lr
        self.monitor = monitor
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss(weight = classes_weight);
        
    def forward(self, input_seq):
        return self.model(input_seq)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience = 3, verbose = True)
        return {
            "optimizer":optimizer,
            "scheduler":scheduler,
            "monitor":self.monitor,
               }
    
    def training_step(self, batch, batch_idx):
        y_pred = self(batch["data"])
        loss = self.criterion(y_pred, batch["label"])
        self.log('train_loss', loss, on_step = False, on_epoch = True, prog_bar = False, logger = True)
        self.log('train_f1_score', f1(y_pred, batch["label"], num_classes = self.num_classes, average = "micro"), on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_pred = self(batch["data"])
        loss = self.criterion(y_pred, batch["label"])
        self.log('val_loss', loss, on_step = False, on_epoch = True, prog_bar = False, logger = True)
        self.log('val_f1_score', f1(y_pred, batch["label"], num_classes = self.num_classes, average = "micro"), on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return loss
    
    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr',type = float,default = 0.0001)
        parser.add_argument('--monitor',type = str,default = "val_f1_score")
        return parser