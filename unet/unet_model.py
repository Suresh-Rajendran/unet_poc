""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .activation import *
import torch
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import FeaturePyramidNetwork, misc_nn_ops


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, act_func = 'esh'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, af = act_func))
        self.down1 = (Down(64, 128, af = act_func))
        self.down2 = (Down(128, 256, af = act_func))
        self.down3 = (Down(256, 512, af = act_func))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, af = act_func))
        self.up1 = (Up(1024, 512 // factor, bilinear, af = act_func))
        self.up2 = (Up(512, 256 // factor, bilinear, af = act_func))
        self.up3 = (Up(256, 128 // factor, bilinear, af = act_func))
        self.up4 = (Up(128, 64, bilinear, af = act_func))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



class CustomFPN(FeaturePyramidNetwork):
    def __init__(self, act_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_func = Activation(act_func)

        def _replace_relu(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, name, self.act_func)
                _replace_relu(child)

        _replace_relu(self)


# Create a custom ResNet model with the custom activation function
class CustomResNet(resnet.ResNet):
    def __init__(self, act_func = 'esh', *args, **kwargs):
        super().__init__(*args, **kwargs)

        def _replace_relu(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, name, Activation(act_func))
                _replace_relu(child)

        _replace_relu(self)


# Function to generate the custom ResNet backbone with FPN
def custom_resnet_fpn_backbone(arch, pretrained, act_func='esh', **kwargs):
    backbone = CustomResNet(resnet.resnet._resnet(arch, resnet.Bottleneck, [3, 4, 6, 3], pretrained, **kwargs), act_func=act_func)
    fpn = CustomFPN(act_func, in_channels_list=[256, 512, 1024, 2048], out_channels=256)
    return BackboneWithFPN(backbone, return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}, fpn=fpn)

# Function to create a custom Mask R-CNN model with the custom backbone
def create_custom_mask_rcnn_model(num_classes, act_func = 'esh'):
    # Create the custom backbone with FPN
    backbone = custom_resnet_fpn_backbone("resnet50", pretrained=True, act_func = act_func)

    # Create the Mask R-CNN model with the custom backbone
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model
