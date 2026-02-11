import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import wandb

# from fae.models.feature_extractor import Extractor
# from fae.utils.pytorch_ssim import SSIMLoss
from models.feature_extractor import Extractor
from utils.pytorch_ssim import SSIMLoss
import pytorch_msssim


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        raise NotImplementedError

    def predict_anomaly(self, x: Tensor):
        raise NotImplementedError

    def load(self, config, epoch: int = None):
        """
        Load model from W&B
        :param path: Path to the model <entity>/<project>/<run_id>/<model_name>
        """
        # name = os.path.basename(path)
        # run_path = os.path.dirname(path)
        # weights = wandb.restore(name, run_path=run_path, root="/tmp", replace=True)
        # if weights is None:
        #     raise RuntimeError(f"Model {name} not found under {run_path}")
        # self.load_state_dict(torch.load(weights.name)['model_state_dict'])
        # os.remove(weights.name)
        save_dir = os.path.join("output", "checkpoints")
        # model_name = os.path.join(save_dir, config.train_dataset+".pt")
        model_name = os.path.join(save_dir, "fae_" + config.train_dataset + "_" + str(epoch) + ".pt")
        # weight = torch.load(model_name)['model_state_dict']
        weight = torch.load(model_name, weights_only=False)['model_state_dict']
        self.load_state_dict(weight, strict=False)

    def save(self, config, name: str):
        save_dir = os.path.join("output", "checkpoints")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = os.path.join(save_dir, config.train_dataset+".pt")
        # torch.save({
        #     'model_state_dict': self.state_dict(),
        #     'config': config
        # }, os.path.join(wandb.run.dir, name))
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': config
        }, os.path.join(save_dir, name))
        # torch.save({
        #     'model_state_dict': self.state_dict(),
        #     'config': config
        # }, save_name)


def vanilla_feature_encoder(in_channels: int, hidden_dims: List[int],
                            norm_layer: str = None, dropout: float = 0.0,
                            bias: bool = False):
    """
    Vanilla feature encoder.
    Args:
        in_channels (int): Number of input channels
        hidden_dims (List[int]): List of hidden channel dimensions
        norm_layer (str): Normalization layer to use
        dropout (float): Dropout rate
        bias (bool): Whether to use bias
    Returns:
        encoder (nn.Module): The encoder
    """
    ks = 5  # Kernel size
    pad = ks // 2  # Padding

    # Build encoder
    enc = nn.Sequential()
    # for i, hidden_dim in enumerate(hidden_dims):
    for i in range(len(hidden_dims)):
        # Add a new layer
        layer = nn.Sequential()

        # Convolution
        layer.add_module(f"encoder_conv_{i}",
                         nn.Conv2d(in_channels, hidden_dims[i], ks, stride=2,
                                   padding=pad, bias=bias))

        # If not last layer
        # if i < len(hidden_dims) - 1:
        # Normalization
        if norm_layer is not None:
            layer.add_module(f"encoder_norm_{i}",
                             eval(norm_layer)(hidden_dims[i]))

        # LeakyReLU
        layer.add_module(f"encoder_relu_{i}", nn.LeakyReLU())

        # Dropout
        if dropout > 0:
            layer.add_module(f"encoder_dropout_{i}", nn.Dropout2d(dropout))

        # Add the layer to the encoder
        enc.add_module(f"encoder_layer_{i}", layer)

        in_channels = hidden_dims[i]

    # Final layer
    enc.add_module("encoder_conv_final",
                   nn.Conv2d(in_channels, in_channels, ks, stride=1, padding=pad,
                             bias=bias))

    return enc


def vanilla_feature_decoder(out_channels: int, hidden_dims: List[int],
                            norm_layer: str = None, dropout: float = 0.0,
                            bias: bool = False):
    """
    Vanilla feature decoder.
    Args:
        out_channels (int): Number of output channels
        hidden_dims (List[int]): List of hidden channel dimensions
        norm_layer (str): Normalization layer to use
        dropout (float): Dropout rate
        bias (bool): Whether to use bias
    Returns:
        decoder (nn.Module): The decoder
    """
    ks = 6  # Kernel size
    pad = 2  # Padding

    hidden_dims = [out_channels] + hidden_dims

    # Build decoder
    dec = nn.Sequential()
    for i in range(len(hidden_dims) - 1, 0, -1):
        # Add a new layer
        layer = nn.Sequential()

        # Transposed convolution
        layer.add_module(f"decoder_tconv_{i}",
                         nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i - 1],
                                            ks, stride=2, padding=pad,
                                            bias=bias))

        # Normalization
        if norm_layer is not None:
            layer.add_module(f"decoder_norm_{i}",
                             eval(norm_layer)(hidden_dims[i - 1]))

        # LeakyReLU
        layer.add_module(f"decoder_relu_{i}", nn.LeakyReLU())

        # Dropout
        if dropout > 0:
            layer.add_module(f"decoder_dropout_{i}", nn.Dropout2d(dropout))

        # Add the layer to the decoder
        dec.add_module(f"decoder_layer_{i}", layer)

    # Final layer
    dec.add_module("decoder_conv_final",
                   nn.Conv2d(hidden_dims[0], out_channels, 1, bias=False))

    return dec


# class AE(nn.Module):
#     # def __init__(self, in_channels=1, latent_size=16, multiplier=1, unc=False, img_size=64):
#     def __init__(self, config):
#         super(AE, self).__init__()
#         in_channels = config.in_channels
#         out_channels = config.in_channels
#         img_size = 32  # feat_size = 128 // 4
#         latent_size = 256
#         self.fm = img_size // 16
#
#         multiplier = 16
#         self.mp = multiplier
#
#         self.en_layer1 = nn.Sequential(
#             nn.Conv2d(in_channels, int(16 * multiplier), 4, 2, 1, bias=False),
#             nn.BatchNorm2d(int(16 * multiplier)),
#             nn.ReLU(True)
#         )
#
#         self.en_layer2 = nn.Sequential(
#             nn.Conv2d(int(16 * multiplier), int(32 * multiplier), 4, 2, 1, bias=False),
#             nn.BatchNorm2d(int(32 * multiplier)),
#             nn.ReLU(True)
#         )
#
#         self.en_layer3 = nn.Sequential(
#             nn.Conv2d(int(32 * multiplier), int(64 * multiplier), 4, 2, 1, bias=False),
#             nn.BatchNorm2d(int(64 * multiplier)),
#             nn.ReLU(True)
#         )
#
#         self.en_layer4 = nn.Sequential(
#             nn.Conv2d(int(64 * multiplier), int(64 * multiplier), 4, 2, 1, bias=False),
#             nn.BatchNorm2d(int(64 * multiplier)),
#             nn.ReLU(True)
#         )
#
#         mid_num = 2048
#         self.linear_enc = nn.Sequential(
#             nn.Linear(int(64 * multiplier) * self.fm * self.fm, mid_num),
#             nn.BatchNorm1d(mid_num),
#             nn.ReLU(True),
#         )
#
#         self.bottle_enc = nn.Linear(mid_num, latent_size)
#         self.bottle_dec = nn.Linear(latent_size, mid_num)
#
#         # ------------------------------------------- Decoder ------------------------------------------- #
#         self.linear_dec = nn.Sequential(
#             nn.BatchNorm1d(mid_num),
#             nn.ReLU(True),
#             nn.Linear(mid_num, int(64 * multiplier) * self.fm * self.fm))
#
#         self.de_layer1 = nn.Sequential(
#             nn.ConvTranspose2d(int(64 * multiplier), int(64 * multiplier), 4, 2, 1, bias=False),
#             nn.BatchNorm2d(int(64 * multiplier)),
#             nn.ReLU(True),
#         )
#         self.de_layer2 = nn.Sequential(
#             nn.ConvTranspose2d(int(64 * multiplier), int(32 * multiplier), 4, 2, 1, bias=False),
#             nn.BatchNorm2d(int(32 * multiplier)),
#             nn.ReLU(True),
#         )
#         self.de_layer3 = nn.Sequential(
#             nn.ConvTranspose2d(int(32 * multiplier), int(16 * multiplier), 4, 2, 1, bias=False),
#             nn.BatchNorm2d(int(16 * multiplier)),
#             nn.ReLU(True),
#         )
#
#         self.de_layer4 = nn.Sequential(
#             nn.ConvTranspose2d(int(16 * multiplier), out_channels, 4, 2, 1, bias=False)
#         )
#
#     def forward(self, x):
#         en_feat = self.encode(x)
#         z = self.bottle_enc(en_feat)
#
#         de_feat = self.bottle_dec(z)
#         out = self.decode(de_feat)
#
#         # return {'en_feat': en_feat, 'z': z, 'x_hat': out}
#         return out
#
#     def encode(self, x):
#         e1 = self.en_layer1(x)  # 16
#         e2 = self.en_layer2(e1)  # 32
#         e3 = self.en_layer3(e2)  # 64
#         e4 = self.en_layer4(e3)  # 64
#         en_feat = e4.view(e4.size(0), -1)
#         en_feat = self.linear_enc(en_feat)
#
#         return en_feat
#
#     def decode(self, de_feat):
#         d4 = self.linear_dec(de_feat)
#         d4 = d4.view(d4.size(0), int(64 * self.mp), self.fm, self.fm)  # 64
#         d3 = self.de_layer1(d4)  # 64
#         d2 = self.de_layer2(d3)  # 32
#         d1 = self.de_layer3(d2)  # 16
#         out = self.de_layer4(d1)  # 1
#
#         return out


class FeatureAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.enc = vanilla_feature_encoder(config.in_channels,
                                           config.hidden_dims,
                                           norm_layer="nn.BatchNorm2d",
                                           dropout=config.dropout,
                                           bias=False)
        self.dec = vanilla_feature_decoder(config.in_channels,
                                           config.hidden_dims,
                                           norm_layer="nn.BatchNorm2d",
                                           dropout=config.dropout,
                                           bias=False)

    def forward(self, x: Tensor) -> Tensor:
        z = self.enc(x)
        rec = self.dec(z)
        return rec


class FeatureReconstructor(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.extractor = Extractor(inp_size=config.image_size,
                                   cnn_layers=config.extractor_cnn_layers,
                                   keep_feature_prop=config.keep_feature_prop,
                                   pretrained=not config.random_extractor)

        config.in_channels = self.extractor.c_feats
        self.ae = FeatureAutoencoder(config)
        # self.ae = AE(config)

        if config.loss_fn == 'ssim':
            self.loss_fn = SSIMLoss(window_size=5, size_average=False)
            # self.loss_fn = pytorch_msssim.SSIM(win_size=5, size_average=False, channel=256, data_range=1)
        elif config.loss_fn == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss function: {config.loss_fn}")

    def forward(self, x: Tensor):
        with torch.no_grad():
            feats = self.extractor(x)
            # print(feats.shape)
        return feats, self.ae(feats)

    def get_feats(self, x: Tensor) -> Tensor:
        return self.extractor(x)

    def get_rec(self, feats: Tensor) -> Tensor:
        return self.ae(feats)

    def loss(self, x: Tensor):
        feats, rec = self(x)
        loss = self.loss_fn(rec, feats).mean()
        return {'rec_loss': loss}

    def predict_anomaly(self, x: Tensor):
        """Returns per image anomaly maps and anomaly scores"""
        # Extract features
        feats, rec = self(x)

        # Compute anomaly map
        anomaly_map = self.loss_fn(rec, feats).mean(1, keepdim=True)
        anomaly_score = anomaly_map.mean(-1).mean(-1).mean(-1)
        # anomaly_score = anomaly_map
        # anomaly_score = -self.loss_fn(rec, feats)
        anomaly_map = F.interpolate(anomaly_map, x.shape[-2:], mode='bilinear',
                                    align_corners=True)
        #
        # # Anomaly score only where object in the image, i.e. at x > 0
        # anomaly_score = []
        # for i in range(x.shape[0]):
        #     roi = anomaly_map[i][x[i] > 0]
        #     roi = roi[roi > torch.quantile(roi, 0.9)]
        #     anomaly_score.append(roi.mean())
        # anomaly_score = torch.stack(anomaly_score)
        # return anomaly_map, anomaly_score
        return anomaly_map, anomaly_score


if __name__ == '__main__':
    # Config
    from argparse import Namespace
    config = Namespace()
    config.image_size = 128
    config.hidden_dims = [100, 150, 200, 300]
    config.generator_hidden_dims = [300, 200, 150, 100]
    config.discriminator_hidden_dims = [100, 150, 200, 300]
    config.dropout = 0.2
    config.extractor_cnn_layers = ['layer1', 'layer2']
    config.keep_feature_prop = 1.0
    device = "cpu"

    # Model
    fae = FeatureReconstructor(config).to(device)
    print(fae.ae.enc)
    print("")
    print(fae.ae.dec)

    # Data
    x = torch.randn(32, 1, *[config.image_size] * 2).to(device)

    # Forward
    feats, rec = fae(x)
    print(feats.shape, rec.shape)
    anomaly_map, anomaly_score = fae.predict_anomaly(x)
    # loss = fae.loss(x)
