import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class doubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(doubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
        

class unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(unet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(doubleConv(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(
                feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(doubleConv(feature * 2, feature))

        self.middle = doubleConv(features[-1], features[-1]*2)
        self.output_conv = nn.Conv2d(
            features[0], out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        skip_connections = []

        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.middle(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip_conn = skip_connections[i//2]

            if x.shape != skip_conn.shape:
                x = TF.resize(x, size=skip_conn.shape[2:])

            cat_skip = torch.cat([skip_conn, x], dim=1)
            x = self.decoder[i+1](cat_skip)

        x = self.output_conv(x)
        return x
