import torch
from torch import nn, optim
from torchsummary import summary

#Generator
#define convolution layer
def con_v_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding = 0):
    layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_dim),
            nn.ReLU(True))
    return layer

#define transpose convolution layer
def dcon_v_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding = 0, output_padding = 0):
    layer = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, output_padding),
            nn.InstanceNorm2d(out_dim),
            nn.ReLU(True))
    return layer

class ResidualBlock(nn.Module):
    def __init__(self, dim, use_dropout):
        super(ResidualBlock, self).__init__()
        res_block = [nn.ReflectionPad2d(1),
                        con_v_norm_relu(dim, dim, kernel_size = 3)]
        if use_dropout:
            res_block += [nn.Dropout(0.5)]
        res_block += [nn.ReflectionPad2d(1),
                        nn.Conv2d(dim, dim, kernel_size = 3, padding = 0),
                        nn.InstanceNorm2d(dim)]
        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)
    
class Generator(nn.Module):
    #input->RGB, output->RGB
    def __init__(self, input = 3, output = 3, filters = 64, use_dropout = True, n_blocks = 2):
        super(Generator, self).__init__()
        # 256 + 3*2 = 262        
        # 262 - 7 + 0 + 1 = 256         
        # ( 256 + 2 - 3 + 1 / 2 = 128        
        # 128 + 2 - 3 + 1 / 2 = 64
        #downsample( shape + 2 * padding - kernel + 1 ) / stride
        model = [nn.ReflectionPad2d(3),
                    con_v_norm_relu(input, filters * 1, 7),
                    con_v_norm_relu(filters * 1, filters * 2, 3, 2, 1),
                    con_v_norm_relu(filters * 2, filters * 4, 3, 2, 1)]
        #neck
        for i in range(n_blocks):
            model += [ResidualBlock(filters * 4, use_dropout)]

        #upsample (input-1)*stride + kernel - 2*padding + output_padding
        # (64 - 1) * 2 + 3 - 2 * 1 + 1 = 128
        # (128 - 1) * 2 + 3 - 2 * 1 + 1 = 256
        # 256 + 6 = 262
        # 262 - 7 + 1 = 256
        model += [dcon_v_norm_relu(filters * 4, filters * 2, 3, 2, 1, 1),
                    dcon_v_norm_relu(filters * 2, filters * 1, 3, 2, 1, 1),
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(filters, output, 7),
                    nn.Tanh()]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

#Discriminator
def conv_norm_leakyrelu(in_dim, out_dim, kernel_size, stride = 1, padding = 0, output_padding = 0):
    layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_dim),
            nn.LeakyReLU(0.2, True))
    return layer

class Discriminator(nn.Module):
    def __init__(self, input = 3, filters = 64, n_layer = 3):
        super(Discriminator, self).__init__()
        #layer1 don't batchNorm
        # 256 -1 +1 = 256
        model = [nn.Conv2d(input, filters, kernel_size = 1, stride = 1, padding = 0),
                    nn.LeakyReLU(0.2, True)]
        #layer2&3
        # 256 +2 -4 +1 / 2 = 128
        for i in range(1, n_layer):
            n_filters_prev = 2 ** (i - 1)
            n_filters = 2 ** i
            model += [conv_norm_leakyrelu(filters * n_filters_prev , filters * n_filters, kernel_size=4, stride=2, padding=1)]
        #layer4 = 1
        n_filters_prev = 2 ** (n_layer - 1)
        n_filters = 2 ** n_layer
        model += [conv_norm_leakyrelu(filters * n_filters_prev , filters * n_filters, kernel_size=4, stride=1, padding=1)]
        #outputlayer
        model += [nn.Conv2d(filters * n_filters, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    

class LamdaLR1():
    def __init__(self, epochs, offset, decay_epoch):
        self.epoch = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch
        
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch) / (self.epoch - self.decay_epoch)
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator().to(device)
    D = Discriminator().to(device)
    summary(G, (3, 256, 256), device = 'cuda' if torch.cuda.is_available() else 'cpu')
    summary(D, (3, 256, 256), device = 'cuda' if torch.cuda.is_available() else 'cpu')