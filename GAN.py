import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import random
from pytorch_lightning.metrics.functional.classification import f1_score

# Set random seed for reproducibility
# manualSeed = 3942 # 7595 # 6161 # 8037
manualSeed = random.randint(1, 10000) # use if you want new results
torch.manual_seed(manualSeed)        
        
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
#     if classname.find('LayerNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
    
class GAN(pl.LightningModule):
    
    def __init__(self, val_model, generator, discriminator, noise_len, val_expected_output, init_weight = True):
        super(GAN, self).__init__()
        self.noise_len = noise_len
        self.criterion = nn.BCELoss()
        self.generator = generator
        self.discriminator = discriminator
        self.val_model = val_model
        self.val_expected_output = val_expected_output
        
        # init weights
        if init_weight:
            init_weights(self.generator)
            init_weights(self.discriminator)
        
        self.save_hyperparameters()
    
    def forward(self, x):
        if self.training :
            return self.discriminator(x)
        else:
            return self.generator(x)
    
    # get a random number from a gaussian distribution with a decaying standard deviation from ground truth
    # This is used to give noisy labels so that the network will not learn very fast (Discriminator loss will reduce to 0 quickly which is a failure mode)
    def gen_randn(self, *size, value = 0, decay = 0):
        if decay != 0:
            stddev = (0.2 * np.exp(-0.01 * self.current_epoch)) # standard deviation exponentially reduces to 0 from 0.1
            if value <= 0.4:
                mean = value + stddev
            elif value >= 0.6:
                mean = value - stddev
            else :
                mean = value
                stddev = 0.1 * stddev
            return mean + (stddev * torch.randn(size, dtype = torch.float, device = self.device))
        else:
            return value * torch.ones(size, dtype = torch.float, device = self.device)
        
    def discriminator_step(self, batch):
        self.discriminator.zero_grad()
        batch_size = batch.shape[0]
        
        # Train discriminator with real images
        label = self.gen_randn(batch_size, value = 1, decay = 1) # 1 is real label. It gives a noisy label 
        output = self(batch).view(-1)
        D_x = output.mean().item()
        self.log("D(x)", D_x, on_step = False, on_epoch = True, prog_bar = False, logger = True)
        errD_real = self.criterion(output, label)
        
        # Train discriminator with fake images
        noise = torch.randn(batch_size, self.noise_len, dtype = torch.float, device = self.device)
        fake_data = self.generator(noise)
        label = self.gen_randn(batch_size, value = 0, decay = 0) # 0 is fake label
        fake_output = self(fake_data.detach()).view(-1)
        D_G_z1 = fake_output.mean().item()
        self.log("D(G(x))_1", D_G_z1, on_step = False, on_epoch = True, prog_bar = False, logger = True)
        errD_fake = self.criterion(fake_output, label)
        
        errD = errD_real + errD_fake
        self.log("train_loss_d", errD, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return errD
        
    def generator_step(self, batch):
         # Train Generator 
        self.generator.zero_grad()
        batch_size = batch.shape[0]
        noise = torch.randn(batch_size, self.noise_len, dtype = torch.float, device = self.device)
        fake_data = self.generator(noise)
        label = self.gen_randn(batch_size, value = 1, decay = 0)  # since fake data should become real data ideally for generator, no noise added
        fake_output = self(fake_data).view(-1)
        D_G_z2 = fake_output.mean().item()
        self.log("D(G(x))_2", D_G_z2, on_step = False, on_epoch = True, prog_bar = False, logger = True)
        
        errG = self.criterion(fake_output, label)
        self.log("train_loss_g", errG, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return errG
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch["data"]

        # train generator
        result = None
        if optimizer_idx == 0:
            result = self.generator_step(x)

        # train discriminator
        if optimizer_idx == 1:
            result = self.discriminator_step(x)

        return result
        
    def validation_step(self, batch, batch_idx):
        noise = torch.randn(batch.shape[0], self.noise_len, dtype = torch.float, device = self.device)
        fake_data = self(noise)
        fake_outputs = self.val_model(fake_data)
        _, index = torch.max(fake_outputs, dim = 1)
        V_G_x = torch.mean(index.to(torch.float)).item()
        self.log("V(G(x))", V_G_x, on_step = False, on_epoch = True, prog_bar = False, logger = True)
        self.log("Accurate_count", torch.count_nonzero(index == self.val_expected_output), on_step = False, on_epoch = True, prog_bar = False, logger = True)
        label = torch.full((batch.shape[0],), self.val_expected_output, dtype = torch.long, device = self.device)
        val_loss = nn.CrossEntropyLoss()(fake_outputs, label)
        self.log("val_loss", val_loss, on_step = False, on_epoch = True, prog_bar = False, logger = True)
        self.log('val_f1_score', f1_score(fake_outputs, label, class_reduction = "micro"), on_step = False, on_epoch = True, prog_bar = True, logger = True)    
        return val_loss
    
    def configure_optimizers(self):
        optiG = torch.optim.Adam(self.generator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
        optiD = torch.optim.Adam(self.discriminator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
        return (optiG, optiD)