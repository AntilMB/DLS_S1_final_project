import torch.nn as nn
import torch

from cycle_gan.model.generators import Generator
from cycle_gan.model.discriminators import Discriminator
from cycle_gan.loss.loss_funtions import ArticleGanLossMSE, CycleLoss


class CycleGan(nn.Module):
    def __init__(self, device='cpu', n_residual=9, lambda_=10, lr=0.0002):
        super().__init__()
        
        self.lambda_ = lambda_
        self.device = device

        # generators
        self.G = Generator(n_residual=n_residual).to(self.device)
        self.F = Generator(n_residual=n_residual).to(self.device)

        # discriminators
        self.D_x = Discriminator().to(self.device)
        self.D_y = Discriminator().to(self.device)

        # loss for generators
        self.gan_loss = ArticleGanLossMSE().to(self.device)
        self.cycle_loss = CycleLoss().to(self.device)

        # criterion
        self.criterion_gen = torch.optim.Adam(list(self.G.parameters()) + list(self.F.parameters()), 
                                              lr=lr)

        self.criterion_dis = torch.optim.Adam(list(self.D_x.parameters()) + list(self.D_y.parameters()), 
                                              lr=lr)
    
    
    def _generators_step(self):
        self._set_discriminators_grad(False)
        # self._set_generator_grad(True)
        self.criterion_gen.zero_grad()
        
        # loss for generators
        loss_G = self.gan_loss(self.D_y(self.fake_y_from_real_x), 1)
        loss_F = self.gan_loss(self.D_x(self.fake_x_from_real_y), 1)
        
        loss_rec_x = self.cycle_loss(self.rec_x_from_fake_y, self.real_x)
        loss_rec_y = self.cycle_loss(self.rec_y_from_fake_x, self.real_y)
        
        gen_loss = loss_G + loss_F + self.lambda_ * (loss_rec_x + loss_rec_y)
        gen_loss.backward()
        
        self.criterion_gen.step()
        
    def _discriminators_step(self):
        self._set_discriminators_grad(True)
        # self._set_generator_grad(False)
        self.criterion_dis.zero_grad()
        
        # loss for discriminators
        loss_x_real = self.gan_loss(self.D_x(self.real_x), 1)
        loss_x_fake = self.gan_loss(self.D_x(self.fake_x_from_real_y.detach()), 0)
        
        loss_y_real = self.gan_loss(self.D_y(self.real_y), 1)
        loss_y_fake = self.gan_loss(self.D_y(self.fake_y_from_real_x.detach()), 0)
        
        loss = loss_x_real + loss_x_fake + loss_y_real + loss_y_fake
        loss.backward()
        
        self.criterion_dis.step()

    def _set_discriminators_grad(self, grad):
        self._set_grad([self.D_x, self.D_y], grad)
    def _set_generator_grad(self, grad):
        self._set_grad([self.F, self.G], grad)

    def _set_grad(self, modules, grad):
        for module in modules:
            for param in module.parameters():
                param.requires_grad = grad

    def forward(self, input_data):
        self.real_x = input_data['x'].to(self.device)
        self.real_y = input_data['y'].to(self.device)
        
        # generate
        self.fake_y_from_real_x = self.G(self.real_x)
        self.rec_x_from_fake_y = self.F(self.fake_y_from_real_x)
        
        self.fake_x_from_real_y = self.F(self.real_y)
        self.rec_y_from_fake_x = self.G(self.fake_x_from_real_y)
        
    def train_step(self, input_data):
        self.forward(input_data)

        # generators training step
        self._generators_step()
        
        # discriminators training step
        self._discriminators_step()

    @torch.no_grad()    
    def test_step(self, input_data):
        self.forward(input_data)
        dct = {
            'real_x': self.real_x,
            'fake_y': self.fake_y_from_real_x,
            'rec_x': self.rec_x_from_fake_y,
            'real_y': self.real_y,
            'fake_x': self.fake_x_from_real_y,
            'rec_y': self.rec_y_from_fake_x
        }
        return dct

