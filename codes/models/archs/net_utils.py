import torch.nn as nn
from .last import last_model

def init_weights(w, init_type):

    if init_type == 'w_init_relu':
        nn.init.kaiming_uniform_(w, nonlinearity = 'relu')
    elif init_type == 'w_init_leaky':
        nn.init.kaiming_uniform_(w, nonlinearity = 'leaky_relu')
    elif init_type == 'w_init':
        nn.init.uniform_(w)

def activation(activation):

    if activation == 'relu':
        return nn.ReLU(inplace = True)
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope = 0.1 ,inplace = True )
    elif activation == 'selu':
        return nn.SELU(inplace = True)
    elif activation == 'linear':
        return nn.Linear()


# ---------------------------------功能函数------------------------------------
def conv_activation(in_ch, out_ch , kernel_size = 3, stride = 1, padding = 1, activation = 'relu', init_type = 'w_init_relu'):


    if activation == 'relu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.ReLU(inplace = True))

    elif activation == 'leaky_relu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.LeakyReLU(negative_slope = 0.1 ,inplace = True ))

    elif activation == 'selu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.SELU(inplace = True))

    elif activation == 'linear':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding))



def upsample(in_ch, out_ch):

    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True)



def leaky_deconv(in_ch, out_ch):

    return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.LeakyReLU(0.1,inplace=True)
                        )

def deconv_activation(in_ch, out_ch ,activation = 'relu' ):

    if activation == 'relu':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.ReLU(inplace = True))

    elif activation == 'leaky_relu':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.LeakyReLU(negative_slope = 0.1 ,inplace = True ))

    elif activation == 'selu':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.SELU(inplace = True))

    elif activation == 'linear':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True))

class Encoder(nn.Module):

    def __init__(self,in_ch, nf):
        super(Encoder, self).__init__()
        self.nf = nf
        embed_dim = 64
        if self.nf == 16:
            embed_dim = 16
        self.last = last_model(embed_dim = embed_dim, number=4, out_number=self.nf)
        
    def forward(self,x):
        conv1 = self.last(x)
        return conv1
    
class Encoder_LR(nn.Module):

    def __init__(self, in_ch, nf):
        super(Encoder_LR, self).__init__()
        self.nf = nf
        self.last = last_model(img_size=120, number=1, type_num=2, out_number=self.nf, window_size=5)
        
    def forward(self, x):
        st = self.last(x)
        return st


class ResBlock(nn.Module):

    def __init__(self, n_filters=64):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x) + x


