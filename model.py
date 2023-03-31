# ### Model
# - UNET
#  - Encoder-Decoder architecture with skip connections
#     - Encoder reduces the size of feature maps by using convolutions + max pooling
#     - Decoder reconstructs segmentation mask by using Upsampling + Convolutions
#     - Skip-Connections allow information flow from encoder to decoder in all intermediate lvls of UNet
#        -  reduces vanishing gradient problem

import torch

# 2 convolutions between each down or upconvolution step
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):  # no of in and out channels for each convolution
        super().__init__()
        # 2 convolutions with Relu
        self.step = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                        # 3 is kernel size and padding 1 , so it doesnt reduce the shape of the input
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d (out_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU())
    def forward(self, X):  # forward function that implement the steps
        return self.step(X)
    

# compine DoubleConv with maxpooling for Encoder layers ,, DoubleConv with Upsampling for Decoder layers

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder layers:
        self.layer1 = DoubleConv(1, 64)  # the single slices have only 1 channel as input ..
        self.layer2 = DoubleConv(64, 128)
        self.layer3 = DoubleConv(128, 256)
        self.layer4 = DoubleConv(256, 512)
        # Decoder layers:
        self.layer5 = DoubleConv(512+256, 256)
        #512 from layer4 . because of skip connections: o/p of layer3 (256) directly forwarded to the current layer5
        self.layer6 = DoubleConv(256+128, 128)
        self.layer7 = DoubleConv(128+64, 64)   # 64 from layer2
        
        self.layer8 = torch.nn.Conv2d(64, 1, 1)  # 64 i/p , 1 o/p , 1 kernel size , final segmentation
        
        self.maxpool = torch.nn.MaxPool2d(2)  # reduce shape of features by 50% 
        
    def forward (self, x):
        x1 = self.layer1(x)
        x1m= self.maxpool(x1)
        
        x2 = self.layer2(x1m)
        x2m= self.maxpool(x2)
        
        x3 = self.layer3(x2m)
        x3m= self.maxpool(x3)
        
        x4 = self.layer4(x3m)
        
        x5 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x4)  # upsample current feature map(x4) with factor 2
        x5 = torch.cat([x5, x3], dim=1)  # skip connection, concat upsampling features from Decoder with matching from Encoder
        x5 = self.layer5(x5)  # compute o/p of corresponding Decoder layer based on the concatinated i/p
        
        x6 = torch.nn.Upsample(scale_factor=2, mode= "bilinear")(x5)
        x6 = torch.cat([x6, x2], dim=1)  # x5 and x3 has same o/p dim . x6 and x2 has same o/p dim
        x6 = self.layer6(x6)
        
        x7 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.layer7(x7)
        
        ret = self.layer8(x7)
        return ret