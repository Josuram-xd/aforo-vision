"""One-off: download OSNet-AIN weights and export them to ONNX for src/identity/body_reid.py.

Torchreid has no CPython 3.14 wheel, so the pipeline runs OSNet through onnxruntime (like ArcFace).
torch is only needed here, to turn the checkpoint into data/osnet/<variant>.onnx.

The network below is the OSNet-AIN architecture from Torchreid (MIT, Zhou et al., ICCV 2019 / TPAMI 2021),
as ported by LibreYOLO (MIT). Module names mirror Torchreid so the checkpoint loads strictly.
Weights: https://huggingface.co/LibreYOLO/LibreReID-osnet (Torchreid multi-source MS+D+C checkpoints, MIT;
trained on MSMT17, DukeMTMC-reID and CUHK03, which carry research-oriented terms).

Downloading is a network call to a third party. Run it on purpose:
    python -m tools.export_osnet_onnx
"""

import argparse
import hashlib
import urllib.request
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

OUTPUT_DIR = Path("data/osnet")
WEIGHT_URL = "https://huggingface.co/LibreYOLO/LibreReID-osnet/resolve/main/{name}.pt"
VARIANTS = {  # name -> (stage widths, sha256 of the checkpoint)
    "osnet_ain_x0_25": ([16, 64, 96, 128], "ce171fe160b3608f5e4c19489774991419be965b1d6f4bdccc4b4cfd2ef95347"),
    "osnet_ain_x0_5": ([32, 128, 192, 256], "510bcebae21bd0c0fcc7df388e97d2f687a9ee4befa4394d6fb1fb19aac0bce2"),
    "osnet_ain_x0_75": ([48, 192, 288, 384], "57b31d7f806edac586540e08e98c589f0010ad7876dacf4af013deaa284dd26d"),
    "osnet_ain_x1_0": ([64, 256, 384, 512], "34c24e98b6b70c8b62480f846fd0d581aa2fd1535bc0276aecf1f10430b731d1"),
}
DEFAULT_VARIANT = "osnet_ain_x0_25"  # ~1 MB, fastest on CPU
INPUT_HEIGHT, INPUT_WIDTH = 256, 128


class _ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, instance_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.InstanceNorm2d(out_ch, affine=True) if instance_norm else nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Conv1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Conv1x1Linear(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if bn else None

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x) if self.bn is not None else x


class _LightConv3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False, groups=out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class _LightConvStream(nn.Module):
    def __init__(self, in_ch, out_ch, depth):
        super().__init__()
        layers = [_LightConv3x3(in_ch, out_ch)] + [_LightConv3x3(out_ch, out_ch) for _ in range(depth - 1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _ChannelGate(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, in_ch // reduction, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_ch // reduction, in_ch, kernel_size=1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        gate = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * gate


class _OSBlock(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=4, streams=4, instance_norm=False):
        super().__init__()
        mid = out_ch // reduction
        self.conv1 = _Conv1x1(in_ch, mid)
        self.conv2 = nn.ModuleList([_LightConvStream(mid, mid, t) for t in range(1, streams + 1)])
        self.gate = _ChannelGate(mid)
        self.conv3 = _Conv1x1Linear(mid, out_ch, bn=not instance_norm)
        self.downsample = _Conv1x1Linear(in_ch, out_ch) if in_ch != out_ch else None
        self.IN = nn.InstanceNorm2d(out_ch, affine=True) if instance_norm else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = sum(self.gate(stream(x1)) for stream in self.conv2)
        x3 = self.conv3(x2)
        if self.IN is not None:
            x3 = self.IN(x3)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


def _stage(in_ch, out_ch, instance_norm_flags):
    layers = []
    for index, instance_norm in enumerate(instance_norm_flags):
        layers.append(_OSBlock(in_ch if index == 0 else out_ch, out_ch, instance_norm=instance_norm))
    return nn.Sequential(*layers)


class OSNetAIN(nn.Module):
    """OSNet-AIN feature extractor: 512-d embedding, classifier head dropped."""

    def __init__(self, channels, feature_dim=512):
        super().__init__()
        self.conv1 = _ConvLayer(3, channels[0], 7, stride=2, padding=3, instance_norm=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = _stage(channels[0], channels[1], [True, True])
        self.pool2 = nn.Sequential(_Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2))
        self.conv3 = _stage(channels[1], channels[2], [False, True])
        self.pool3 = nn.Sequential(_Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2))
        self.conv4 = _stage(channels[2], channels[3], [True, False])
        self.conv5 = _Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU())

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv5(self.conv4(x))
        return self.fc(self.global_avgpool(x).flatten(1))


def download_weights(variant: str, root: Path = OUTPUT_DIR) -> Path:
    destination = root / f"{variant}.pt"
    if not destination.is_file():
        root.mkdir(parents=True, exist_ok=True)
        print(f"Descargando {variant}.pt de Hugging Face (LibreYOLO/LibreReID-osnet) ...")
        urllib.request.urlretrieve(WEIGHT_URL.format(name=variant), destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != VARIANTS[variant][1]:
        destination.unlink()
        raise RuntimeError(f"SHA-256 de {variant}.pt no coincide ({digest}); archivo borrado.")
    return destination


def export(variant: str = DEFAULT_VARIANT, root: Path = OUTPUT_DIR) -> Path:
    checkpoint = torch.load(download_weights(variant, root), map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in checkpoint.get("state_dict", checkpoint).items() if not k.startswith("classifier.")}
    model = OSNetAIN(VARIANTS[variant][0])
    model.load_state_dict(state)  # strict: fails loudly if the architecture drifts from the checkpoint
    model.eval()

    destination = root / f"{variant}.onnx"
    dummy = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    torch.onnx.export(
        model, dummy, str(destination), input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17, dynamo=False,
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga pesos OSNet-AIN y los exporta a ONNX para body_reid.py.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default=DEFAULT_VARIANT)
    args = parser.parse_args()
    print(f"Listo: {export(args.variant)}")
