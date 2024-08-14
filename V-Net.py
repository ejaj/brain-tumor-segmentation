import torch
import torch.nn as nn
import torch.nn.functional as F


class VNet(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=32):
        super(VNet, self).__init__()

        self.encoder1 = self.create_encoder_block(in_channels, init_features)
        self.encoder2 = self.create_encoder_block(init_features, init_features * 2)
        self.encoder3 = self.create_encoder_block(init_features * 2, init_features * 4)
        self.encoder4 = self.create_encoder_block(init_features * 4, init_features * 8)

        self.bottleneck = self.create_encoder_block(init_features * 8, init_features * 16)

        self.upconv4 = self.create_upconv_block(init_features * 16, init_features * 8)
        self.decoder4 = self.create_decoder_block(init_features * 16, init_features * 8)

        self.upconv3 = self.create_upconv_block(init_features * 8, init_features * 4)
        self.decoder3 = self.create_decoder_block(init_features * 8, init_features * 4)

        self.upconv2 = self.create_upconv_block(init_features * 4, init_features * 2)
        self.decoder2 = self.create_decoder_block(init_features * 4, init_features * 2)

        self.upconv1 = self.create_upconv_block(init_features * 2, init_features)
        self.decoder1 = self.create_decoder_block(init_features * 2, init_features)

        self.final_conv = nn.Conv3d(init_features, out_channels, kernel_size=1)

    def create_encoder_block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )

    def create_upconv_block(self, in_channels, features):
        return nn.ConvTranspose3d(in_channels, features, kernel_size=2, stride=2)

    def create_decoder_block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Check and adjust dimensions
        if x.size(2) < 4 or x.size(3) < 4 or x.size(4) < 4:
            raise ValueError("Input dimensions are too small. Consider resizing your input.")

        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool3d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool3d(enc3, kernel_size=2, stride=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool3d(enc4, kernel_size=2, stride=2))

        # Decoder path
        dec4 = self.decode_block(self.upconv4, self.decoder4, bottleneck, enc4)
        dec3 = self.decode_block(self.upconv3, self.decoder3, dec4, enc3)
        dec2 = self.decode_block(self.upconv2, self.decoder2, dec3, enc2)
        dec1 = self.decode_block(self.upconv1, self.decoder1, dec2, enc1)

        # Final output
        return torch.sigmoid(self.final_conv(dec1))

    def decode_block(self, upconv, decoder, x, enc_features):
        x = upconv(x)
        # Use F.interpolate to ensure correct size
        x = F.interpolate(x, size=enc_features.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat((x, enc_features), dim=1)
        x = decoder(x)
        return x


# Example usage
if __name__ == "__main__":
    model = VNet(in_channels=4, out_channels=3, init_features=16)
    x = torch.randn(4, 4, 32, 240, 240)
    output = model(x)
    print(output.shape)
