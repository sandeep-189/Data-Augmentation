import pytorch_lightning as pl

class Net(pl.LightningModule):
    def __init__(self, model, classes_weight = None):
        super(Net,self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss(weight = classes_weight);
        
    def forward(self, input_seq):
        return self.model(input_seq)
    
    def configure_optimizers(self):
        return torch.optim.Adamax(self.parameters(), lr=0.0001)
    
    def training_step(self, batch, batch_idx):
        y_pred = self(batch["data"])
        loss = self.criterion(y_pred, batch["label"])
        self.log('train_loss', loss, on_step = False, on_epoch = True, prog_bar = False, logger = True)
        self.log('train_f1_score', f1_score(y_pred, batch["label"], class_reduction = "micro"), on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_pred = self(batch["data"])
        loss = self.criterion(y_pred, batch["label"])
        self.log('val_loss', loss, on_step = False, on_epoch = True, prog_bar = False, logger = True)
        self.log('val_f1_score', f1_score(y_pred, batch["label"], class_reduction = "micro"), on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return loss