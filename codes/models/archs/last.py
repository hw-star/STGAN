import torch
import torch.nn as nn 
from .swin_transformer import SwinTransformer_HW, PatchEmbed

class last_model(nn.Module):
    def __init__(self,img_size=480, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=64, depths=[2,2,4], num_heads=[4,8,16],
                 window_size=10, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, out_number=64, type_num=1,number = 1):
        super(last_model, self).__init__()
        data = {
        "img_size": img_size,
        "patch_size": patch_size,
        "in_chans": in_chans,
        "num_classes": num_classes,
        "embed_dim": embed_dim,
        "depths": depths,
        "num_heads": num_heads,
        "window_size": window_size,
        "mlp_ratio": mlp_ratio,
        "qkv_bias": qkv_bias,
        "qk_scale": qk_scale,
        "drop_rate": drop_rate,
        "attn_drop_rate": attn_drop_rate,
        "drop_path_rate": drop_path_rate,
        "norm_layer": norm_layer,
        "ape": ape,
        "patch_norm": patch_norm,
        "use_checkpoint": use_checkpoint,
        "out_number": out_number,
        "type_num": type_num,
        "number": number
        }
        self.patch_embed = PatchEmbed(
            img_size=data['img_size'], patch_size=data['patch_size'], in_chans=data['in_chans'], embed_dim=data['embed_dim'],
        norm_layer=data['norm_layer'] if data['patch_norm'] else None
        )
        self.norm = data['norm_layer'](data['embed_dim'])

        self.xian = nn.Conv2d(3, data['embed_dim'], kernel_size=4, stride=4, padding=0)

        self.content_extractor = ContentExtractor(input=data['embed_dim'], ngf=data['embed_dim'])

        self.content_extractor2 = ContentExtractor(input=data['embed_dim'], ngf=data['embed_dim'])

        self.concat1 = nn.Conv2d(data['embed_dim'] * 2, data['embed_dim'], kernel_size=3, stride=1, padding=1)
        self.concat2 = nn.Conv2d(data['embed_dim'] * 2, data['embed_dim'], kernel_size=3, stride=1, padding=1)
        

        self.st1 = SwinTransformer_HW(img_size=120, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=data['embed_dim'], depths=data["depths"], num_heads=data["num_heads"],
                 window_size=data["window_size"], mlp_ratio=data["mlp_ratio"], qkv_bias=data["qkv_bias"], qk_scale=data["qk_scale"],
                 drop_rate=data["drop_rate"], attn_drop_rate=data["attn_drop_rate"], drop_path_rate=data["drop_path_rate"],
                 norm_layer=data["norm_layer"], ape=data["ape"], patch_norm=data["patch_norm"],
                 use_checkpoint=data["use_checkpoint"], out_number = data["out_number"], number=data["number"], type_num=data['type_num'], patch_embed= self.patch_embed, a_num_patches= self.patch_embed.num_patches, b_patches_resolution=self.patch_embed.patches_resolution)
        
        self.st2 = SwinTransformer_HW(img_size=120, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=data['embed_dim'], depths=data["depths"], num_heads=data["num_heads"],
                 window_size=data["window_size"], mlp_ratio=data["mlp_ratio"], qkv_bias=data["qkv_bias"], qk_scale=data["qk_scale"],
                 drop_rate=data["drop_rate"], attn_drop_rate=data["attn_drop_rate"], drop_path_rate=data["drop_path_rate"],
                 norm_layer=data["norm_layer"], ape=data["ape"], patch_norm=data["patch_norm"],
                 use_checkpoint=data["use_checkpoint"], out_number = data["out_number"], number=data["number"], type_num=data['type_num'],patch_embed= self.patch_embed, a_num_patches= self.patch_embed.num_patches, b_patches_resolution=self.patch_embed.patches_resolution)

    def forward(self, x):
        h = self.xian(x)
        h = self.content_extractor(h)
        x = self.patch_embed(x)
        x = self.st1(x) # 64 120 120 / 16 120 120
        x = self.concat1(torch.cat((x, h), dim=1)) # 64/16 120 120
        h = self.content_extractor2(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.st2(x) # 64/16 120 120
        x = self.concat2(torch.cat((x, h), dim=1))
        return x


class ContentExtractor(nn.Module):
    def __init__(self,input = 3, ngf=64, n_blocks=15):
        super(ContentExtractor, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(input, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.body = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
        )
        
    def forward(self, x):
        h = self.head(x)
        h = self.body(h) + h
        return h


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