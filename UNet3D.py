import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=32):
        super(UNet3D, self).__init__()

        self.encoder1, self.pool1 = self.create_encoder_block(in_channels, init_features)
        self.encoder2, self.pool2 = self.create_encoder_block(init_features, init_features * 2)
        self.encoder3, self.pool3 = self.create_encoder_block(init_features * 2, init_features * 4)
        self.encoder4, self.pool4 = self.create_encoder_block(init_features * 4, init_features * 8)

        self.bottleneck = self.create_conv_block(init_features * 8, init_features * 16)

        self.upconv4 = self.create_upconv_block(init_features * 16, init_features * 8)
        self.decoder4 = self.create_decoder_block(init_features * 8, init_features * 8)

        self.upconv3 = self.create_upconv_block(init_features * 8, init_features * 4)
        self.decoder3 = self.create_decoder_block(init_features * 4, init_features * 4)

        self.upconv2 = self.create_upconv_block(init_features * 4, init_features * 2)
        self.decoder2 = self.create_decoder_block(init_features * 2, init_features * 2)

        self.upconv1 = self.create_upconv_block(init_features * 2, init_features)
        self.decoder1 = self.create_decoder_block(init_features, init_features)

        self.final_conv = nn.Conv3d(in_channels=init_features, out_channels=out_channels, kernel_size=1)

    def create_encoder_block(self, in_channels, features):
        encoder = self.create_conv_block(in_channels, features)
        pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # Pooling only in height and width
        return encoder, pool

    def create_conv_block(self, in_channels, features):
        block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True)
        )
        return block

    def create_upconv_block(self, in_channels, features):
        return nn.ConvTranspose3d(in_channels, features, kernel_size=2, stride=2)

    def create_decoder_block(self, in_channels, features):
        return self.create_conv_block(in_channels * 2, features)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.decode_block(self.upconv4, self.decoder4, bottleneck, enc4)
        dec3 = self.decode_block(self.upconv3, self.decoder3, dec4, enc3)
        dec2 = self.decode_block(self.upconv2, self.decoder2, dec3, enc2)
        dec1 = self.decode_block(self.upconv1, self.decoder1, dec2, enc1)

        return torch.sigmoid(self.final_conv(dec1))

    def decode_block(self, upconv, decoder, x, enc_features):
        x = upconv(x)

        # Resize the upsampled tensor to match the encoder feature map size
        x = F.interpolate(x, size=enc_features.shape[2:], mode='trilinear', align_corners=True)

        x = torch.cat((x, enc_features), dim=1)
        x = decoder(x)
        return x
