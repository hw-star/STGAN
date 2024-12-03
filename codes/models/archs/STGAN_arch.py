import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.net_utils import *

class STAGN(nn.Module):
    def __init__(self, ngf=64, n_blocks=16):
        super(STAGN, self).__init__()
        self.get_g_nopadding = Get_gradient_nopadding()
        self.Encoder = Encoder(3, nf=ngf)
        self.Encoder_grad = Encoder(3, nf=int(ngf/4))
        self.gpcd_align = Merge()
        self.content_extractor = ContentExtractor(ngf * 2, n_blocks)
        self.Reconstruction = Reconstruction()
        self.LR_to_threeChannel = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.last = nn.Sequential(
            nn.Conv2d(3, 6, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(6, 3, 3, 1, 1, bias=True)
        )

    def forward(self, LR, LR_UX4, Ref,Ref_DUX4, weights=None):
        LR_UX4_grad = self.get_g_nopadding(LR_UX4)
        Ref_DUX4_grad = self.get_g_nopadding(Ref_DUX4)
        Ref_grad = self.get_g_nopadding(Ref)
        LR_grad = self.get_g_nopadding(LR)

        upscale_plain, content_feat = self.content_extractor(LR) # content_feat-LR:4 128 120 120
        upscale_plain, content_feat_LR = self.content_extractor(LR_grad) # content_feat-LR:4 128 120 120

        sthw_LR_UX4 = self.Encoder(LR_UX4)        
        sthw_Ref_DUX4 = self.Encoder(Ref_DUX4)
        sthw_Ref = self.Encoder(Ref)
        
        #grad
        sthw_LR_UX4_grad = self.Encoder_grad(LR_UX4_grad)
        sthw_Ref_DUX4_grad = self.Encoder_grad(Ref_DUX4_grad)
        sthw_Ref_grad = self.Encoder_grad(Ref_grad)

        maps = [sthw_LR_UX4, sthw_Ref_DUX4, sthw_Ref]
        maps_grad = [sthw_LR_UX4_grad, sthw_Ref_DUX4_grad, sthw_Ref_grad]

        merge = self.gpcd_align(maps, maps_grad, content_feat, content_feat_LR) # 128 120 120
        return self.Reconstruction(merge, content_feat)
    
class Merge(nn.Module):
    def __init__(self):
        super(Merge, self).__init__()
        self.conv_LR_UX4 = nn.Conv2d(80, 80, 3, 1, 1, bias=True)
        self.conv_Ref_DUX4 = nn.Conv2d(80, 80, 3, 1, 1, bias=True)
        self.conv_Ref = nn.Conv2d(80, 80, 3, 1, 1, bias=True)

        self.conv = nn.Conv2d(80 * 3, 128, 3, 1, 1, bias=True)  
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.RAM_LR = RAM(nf=64)
        self.RAM_LR_grad = RAM(nf=16)

    def forward(self, maps, maps_grad, LR_simple,LR_grad_simple):
        x = self.RAM_LR(maps[0], LR_simple)
        y = self.RAM_LR(maps[1], LR_simple)
        z = self.RAM_LR(maps[2], LR_simple)

        x_grad = self.RAM_LR_grad(maps_grad[0], LR_grad_simple)
        y_grad = self.RAM_LR_grad(maps_grad[1], LR_grad_simple)
        z_grad = self.RAM_LR_grad(maps_grad[2], LR_grad_simple)

        cat_LR_UX4 = torch.cat([x,x_grad], dim=1)
        cat_Ref_DUX4 = torch.cat([y,y_grad], dim=1)
        cat_Ref = torch.cat([z,z_grad], dim=1)

        last_x = self.conv_LR_UX4(cat_LR_UX4)
        last_y = self.conv_Ref_DUX4(cat_Ref_DUX4)
        last_z = self.conv_Ref(cat_Ref)

        conv = self.lrelu(self.conv(torch.cat([last_x, last_y, last_z], dim=1)))
        return conv


class ContentExtractor(nn.Module):
    def __init__(self, ngf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.body = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
        )

        self.tail = nn.Sequential(
            nn.Conv2d(ngf, ngf // 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf // 8, 3, kernel_size=3, stride=1, padding=1),
            # nn.Tanh()
        )
        
    def forward(self, x):
        h = self.head(x)
        h = self.body(h) + h
        upscale = self.tail(h)
        return upscale, h
    
class Reconstruction(nn.Module):
    def __init__(self):
        super(Reconstruction, self).__init__()
        self.conv1 = nn.Conv2d(128 * 2, 128, 3, 1, 1, bias=True)
        self.conv1_30 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(8, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, merge, content_feat):
        x = merge + content_feat
        x = self.lrelu(self.conv1_30(x))
        x = self.pixel_shuffle(x)
        x = self.lrelu(self.conv2(x))

        return x

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


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[-3, -10, -3],
                    [0, 0, 0],
                    [3, 10, 3]]
        kernel_h = [[-3, 0, 3],
                    [-10, 0, 10],
                    [-3, 0, 3]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)

        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)


    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim = 1)

        return x

class RAM(nn.Module):
    def __init__(self, nf=64):
        super(RAM, self).__init__()
        self.mul_conv1 = nn.Conv2d(nf + 128, 128, kernel_size=3, stride=1, padding=1)
        self.mul_conv2 = nn.Conv2d(128, nf, kernel_size=3, stride=1, padding=1)
        self.add_conv1 = nn.Conv2d(nf + 128, 128, kernel_size=3, stride=1, padding=1)
        self.add_conv2 = nn.Conv2d(128, nf, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, features, conditions): # conditions是LR
        print("RAM cat :",features.shape, conditions.shape)
        cat_input = torch.cat((features, conditions), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.lrelu(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.lrelu(self.add_conv1(cat_input)))
        return features * mul + add