import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from encoder_cnn import IMG_ENCODE_SIZE, EncoderCNN
from torch.optim import AdamW
from torchmetrics import Accuracy


class ERM(pl.LightningModule):
    def __init__(self, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.ecnn = EncoderCNN()
        self.classifier = nn.Linear(IMG_ENCODE_SIZE, 1)
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_acc = Accuracy('binary')
        self.test_acc = Accuracy('binary')

    def forward(self, x, y, e, p, c):
        x = self.ecnn(x)
        y_pred = self.classifier(x).view(-1)
        return y_pred, y

    def training_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        loss = F.binary_cross_entropy_with_logits(y_pred, y.float())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        y_pred, y = self(*batch)
        if dataloader_idx == 0:
            self.val_acc(y_pred, y)
        else:
            assert dataloader_idx == 1
            self.test_acc(y_pred, y)

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc)
        self.log('test_acc', self.test_acc)

    def test_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        self.test_acc(y_pred, y)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)