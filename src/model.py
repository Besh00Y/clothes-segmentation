import torch
import torch.nn as nn
import torchvision.models as models


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1 )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d( out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):

        x = nn.functional.interpolate( x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x

class UNetResNet34(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        weights = (
            models.ResNet34_Weights.DEFAULT
            if pretrained
            else None
        )

        resnet = models.resnet34(weights=weights)

        self.encoder0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        self.pool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(in_channels=512,skip_channels=256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):

        # Encoder
        e0 = self.encoder0(x)
        e1 = self.encoder1(self.pool(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)

        # Final segmentation mask
        out = self.final_conv(d1)

        out = nn.functional.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out


if __name__ == "__main__":

    model = UNetResNet34(pretrained=True)

    x = torch.randn(2, 3, 256, 256)

    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)