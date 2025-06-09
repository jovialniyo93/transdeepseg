import torch.nn as nn
import torch
import torch.nn.functional as F


'''UNet model'''

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, stride=1, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Padding in case the input sizes are not compatible
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate and apply double conv
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.outc_edges = nn.Conv2d(64, 10, kernel_size=1)  # Assume edge output is 1 channel

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
        edges = self.outc_edges(x)  # Generate edges output
        return logits, edges



'''StarDist'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.double_conv(x)
        x2 = self.conv1d(x)
        x = x1 + x2
        return F.relu(x)

class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.bilinear:
            x1 = self.conv(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x



class StarDistModel(nn.Module):
    def __init__(self, n_channels, n_classes, n_rays=32, bilinear=True):
        super(StarDistModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_rays = n_rays
        self.bilinear = bilinear

        self.res1 = ResBlock(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.res2 = ResBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.res3 = ResBlock(128, 256)
        self.up1 = Upsampling(256, 128)
        self.res4 = ResBlock(384, 128)
        self.up2 = Upsampling(128, 64)
        self.res5 = ResBlock(192, 64)
        self.output_conv = nn.Conv2d(64, n_classes + n_rays, kernel_size=1)

    def forward(self, x):
        x1 = self.res1(x)
        x2 = self.down1(x1)
        x3 = self.res2(x2)
        x4 = self.down2(x3)
        x5 = self.res3(x4)
        x6 = self.up1(x5, x3)
        x7 = self.res4(x6)
        x8 = self.up2(x7, x1)
        x9 = self.res5(x8)
        output = self.output_conv(x9)
        logits = output[:, :self.n_classes, :, :]
        distances = output[:, self.n_classes:, :, :]
        return logits, distances


'''FCN8s'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN8s(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(FCN8s, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Convolutional layers setup
        self.conv1_1 = nn.Conv2d(n_channels, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Subsequent conv layers up to conv5_3
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Fully connected layers adapted for segmentation
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # Output layers for segmentation
        self.score_fr = nn.Conv2d(4096, n_classes, 1)
        self.score_edges = nn.Conv2d(4096, n_classes, 1)  # For edges or additional features

        # Upsampling
        self.upscore2 = nn.ConvTranspose2d(n_classes, n_classes, 4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(n_classes, n_classes, 4, stride=2, bias=False)
        self.upscore_final = nn.ConvTranspose2d(n_classes, n_classes, 16, stride=8, bias=False)

    def forward(self, x):
        h = self.relu1_1(self.conv1_1(x))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        logits = self.score_fr(h)
        edges = self.score_edges(h)

        # Upsample logits to the final image size
        logits = self.upscore2(logits)
        logits = self.upscore_pool4(logits)
        logits = self.upscore_final(logits)
        logits = F.interpolate(logits, size=(576, 576), mode='bilinear', align_corners=False)

        # Similarly, upsample edges to the final image size
        edges = self.upscore2(edges)
        edges = self.upscore_pool4(edges)
        edges = self.upscore_final(edges)
        edges = F.interpolate(edges, size=(576, 576), mode='bilinear', align_corners=False)

        return logits, edges

# Example instantiation and test of the network
#if __name__ == '__main__':
    #model = FCN8s(n_channels=1, n_classes=2)
    #dummy_input = torch.rand((1, 1, 576, 576))  # Example input tensor
    #logits, edges = model(dummy_input)
    #print("Output shapes - Logits: {}, Edges: {}".format(logits.shape, edges.shape))



'''Cellpose'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def batchconv(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )

def batchconv0(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.double_conv(x)
        x2 = self.conv1d(x)
        x = x1 + x2
        x = self.bn(x)
        x = self.relu(x)
        return x

class Upsampling(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

class Cellpose(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(Cellpose, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.res1 = ResBlock(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.res2 = ResBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.res3 = ResBlock(128, 256)
        self.up1 = Upsampling(256, bilinear=bilinear)
        self.res4 = ResBlock(384, 128)
        self.up2 = Upsampling(128, bilinear=bilinear)
        self.res5 = ResBlock(192, 64)
        self.conv3 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.res1(x)
        x2 = self.down1(x1)
        x2 = self.res2(x2)
        x3 = self.down2(x2)
        x3 = self.res3(x3)
        x4 = self.up1(x3, x2)
        x4 = self.res4(x4)
        x5 = self.up2(x4, x1)
        x6 = self.res5(x5)
        logits = self.conv3(x6)
        edges = self.conv4(x6)
        return logits, edges

# Example instantiation and forwarding of the network
#model = Cellpose(n_channels=1, n_classes=2, bilinear=True)
#dummy_input = torch.rand((1, 1, 576, 576))  # Dummy input to test the model
#logits, edges = model(dummy_input)  # Forward pass to get logits and edge outputs
# Print the shapes of the outputs to confirm they are as expected
#print("Shape of logits:", logits.shape)
#print("Shape of edges:", edges.shape)


'''DeepSeg Model'''

class Upsampling(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1=self.double_conv(x)
        x2 = self.conv1d(x)
        x=x1+x2
        x=self.bn(x)
        x=self.relu(x)
        return x

class DeepSeg(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DeepSeg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.res1=ResBlock(n_channels,64)
        self.down1 = nn.MaxPool2d(2)
        self.res2 = ResBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.res3 = ResBlock(128, 256)
        self.up1 = Upsampling(256, 128)
        self.res4 = ResBlock(384, 128)
        self.up2 = Upsampling(128, 64)
        self.res5 = ResBlock(192, 64)
        self.conv3 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)


    def forward(self, x):
        x1=self.res1(x)
        x2=self.down1(x1)
        x2 = self.res2(x2)
        x3 = self.down2(x2)
        x3 = self.res3(x3)
        x4=self.up1(x3,x2)
        x4 = self.res4(x4)
        x5 = self.up2(x4,x1)
        x6 = self.res5(x5)
        logits=self.conv3(x6)
        edges = self.conv4(x6)
        return logits,edges

'''TransformerDeepSeg Model'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock2D(nn.Module):
    def __init__(self, channels, num_heads=8, dk=64, kernel_size=3, stride=1, padding=1):
        super(TransformerBlock2D, self).__init__()
        self.conv_q = nn.Conv2d(channels, dk * num_heads, kernel_size, stride, padding)
        self.conv_k = nn.Conv2d(channels, dk * num_heads, kernel_size, stride, padding)
        self.conv_v = nn.Conv2d(channels, dk * num_heads, kernel_size, stride, padding)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=dk * num_heads, num_heads=num_heads)
        self.fc = nn.Linear(dk * num_heads, channels)
        self.layer_norm = nn.LayerNorm([channels])

    def forward(self, x):
        batch, channels, height, width = x.size()
        q = self.conv_q(x).view(batch, -1, height * width).permute(2, 0, 1)
        k = self.conv_k(x).view(batch, -1, height * width).permute(2, 0, 1)
        v = self.conv_v(x).view(batch, -1, height * width).permute(2, 0, 1)
        
        att_output, _ = self.multihead_attention(q, k, v)
        att_output = att_output.permute(1, 2, 0).contiguous().view(batch, channels, height, width)
        
        out = self.fc(att_output.transpose(1, 3)).transpose(1, 3)
        return self.layer_norm(out + x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ResBlock, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Upsampling, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class TransformerDeepSeg(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(TransformerDeepSeg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.res1 = ResBlock(n_channels, 64)
        self.transformer1 = TransformerBlock2D(64)
        self.down1 = nn.MaxPool2d(2)
        
        self.res2 = ResBlock(64, 128)
        self.transformer2 = TransformerBlock2D(128)
        self.down2 = nn.MaxPool2d(2)
        
        self.res3 = ResBlock(128, 256)
        self.transformer3 = TransformerBlock2D(256)
        self.up1 = Upsampling(256, 128, bilinear)
        
        self.res4 = ResBlock(128, 128)
        self.transformer4 = TransformerBlock2D(128)
        self.up2 = Upsampling(128, 64, bilinear)
        
        self.res5 = ResBlock(64, 64)
        self.transformer5 = TransformerBlock2D(64)
        
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        self.edge_conv = nn.Conv2d(64, n_classes, kernel_size=1)  # Additional output for edges

    def forward(self, x):
        x = self.res1(x)
        x = self.transformer1(x)
        x = self.down1(x)
        
        x = self.res2(x)
        x = self.transformer2(x)
        x = self.down2(x)
        
        x = self.res3(x)
        x = self.transformer3(x)
        x = self.up1(x, x)
        
        x = self.res4(x)
        x = self.transformer4(x)
        x = self.up2(x, x)
        
        x = self.res5(x)
        x = self.transformer5(x)
        
        logits = self.final_conv(x)
        edges = self.edge_conv(x)  # Compute edges
        return logits, edges


'''UNet Segmentation'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat((x, skip), dim=1)  # Concatenate on the channel axis
        x = self.conv(x)
        return x

class UnetSegmentation(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UnetSegmentation, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder blocks
        self.e1 = encoder_block(n_channels, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        # Bottleneck using conv_block instead of en_conv_block
        self.b = conv_block(512, 1024)  # Adjusted to use conv_block

        # Decoder blocks
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        # Output layers for segmentation and edges
        self.outputs = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)
        self.outputs_edges = nn.Conv2d(64, 10, kernel_size=1)  # Edge detection output layer, adjust channels as needed

    def forward(self, inputs):
        # Encoding path
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # Bottleneck (bridge)
        b = self.b(p4)

        # Decoding path
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        # Output for segmentation and edges
        logits = self.outputs(d4)
        edges = self.outputs_edges(d4)

        return logits, edges









