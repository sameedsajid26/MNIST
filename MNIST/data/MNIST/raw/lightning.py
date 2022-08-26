import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


import pytorch_lightning as pl

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001



class NeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

    def configure_optimizers(self):
        params = [encoder.parameters(), decoder.parameters()]
        optimizer = torch.optim.Adam(
        self.
        parameters, lr=1e-3)
        optimizer = torch.optim.Adam(params, lr=1e-3)
        return optimizer
        return optimizer
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.cuda(0)
        x = x.view(x.size(0), -1)
        z =self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log(‘train_loss’, loss)
        return loss
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.cuda(0)
        x = x.view(x.size(0), -1)
        z =self.encoder(x)
        x_hat =self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        val_loss.append(loss)
        self.log(‘val_loss’, loss)
