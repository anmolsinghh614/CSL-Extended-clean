import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch


def _adapt_resnet_for_cifar(model):
    """
    Adapt an ImageNet-style ResNet for small images (32×32 or 64×64).

    ImageNet ResNets use a 7×7 conv (stride 2) + maxpool (stride 2) as the stem,
    which immediately reduces 224×224 → 56×56. On 32×32 inputs, this produces
    tiny 8×8 feature maps that destroy spatial information.

    Standard CIFAR practice: replace with a 3×3 conv (stride 1) + no maxpool,
    preserving full resolution into the residual blocks.
    """
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class ResNet18(nn.Module):
    """ResNet-18 — used for Tiny ImageNet (200 classes, 64×64)."""
    def __init__(self, num_classes=200, pretrained=False, image_size=64):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        if image_size <= 64:
            _adapt_resnet_for_cifar(self.model)
        self._update_num_classes(num_classes)

    def _update_num_classes(self, num_classes):
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x, return_features=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)

        if return_features:
            return self.model.fc(features), features
        return self.model.fc(features)

    def get_feature_dim(self):
        return self.model.fc.in_features


class _LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(_LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class _CifarBasicBlock(nn.Module):
    """
    Basic residual block of the CIFAR ResNets.

    Downsampling shortcuts use He et al.'s option A — subsample spatially and zero-pad the
    new channels — rather than a 1×1 projection. Option A adds no parameters, which is what
    keeps ResNet-32 at 0.46M, and it is what the long-tailed CIFAR literature builds on.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(_CifarBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            pad = (planes - in_planes) // 2
            self.shortcut = _LambdaLayer(
                lambda x: F.pad(x[:, :, ::stride, ::stride], (0, 0, 0, 0, pad, pad))
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet32(nn.Module):
    """
    ResNet-32 for CIFAR (He et al.) — the backbone the long-tailed CIFAR benchmarks report
    against.

    Three stages of five basic blocks at 16/32/64 channels, from a 3×3 stem, giving 6n+2 = 32
    layers and ~0.46M parameters. This is a different network from a torchvision ResNet with a
    CIFAR stem: ImageNet ResNet-34 is ~21M parameters and produces 512-dim features, so
    accuracies obtained with it are not comparable to published CIFAR-10-LT/CIFAR-100-LT
    numbers. Use `ResNet34` if the larger model is wanted deliberately.
    """
    def __init__(self, num_classes=10, pretrained=False, image_size=32, blocks_per_stage=5):
        super(ResNet32, self).__init__()
        if pretrained:
            raise ValueError(
                "No pretrained weights exist for the CIFAR ResNet-32; it is trained from "
                "random initialization. Pass pretrained=False, or use ResNet34/ResNet50 for "
                "an ImageNet-pretrained backbone."
            )

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, blocks_per_stage, stride=1)
        self.layer2 = self._make_layer(32, blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(64, blocks_per_stage, stride=2)
        self.fc = nn.Linear(64, num_classes)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(_CifarBasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _update_num_classes(self, num_classes):
        self.fc = nn.Linear(self.fc.in_features, num_classes)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Adaptive pooling rather than a fixed 8×8 kernel, so the same backbone still works if
        # it is ever handed inputs at a resolution other than 32×32.
        out = F.adaptive_avg_pool2d(out, 1)
        features = torch.flatten(out, 1)

        if return_features:
            return self.fc(features), features
        return self.fc(features)

    def get_feature_dim(self):
        return self.fc.in_features


class ResNet34(nn.Module):
    """ResNet-34 with a CIFAR stem — larger alternative to the benchmark ResNet-32."""
    def __init__(self, num_classes=1000, pretrained=False, image_size=32):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(pretrained=pretrained)
        if image_size <= 64:
            _adapt_resnet_for_cifar(self.model)
        self._update_num_classes(num_classes)

    def _update_num_classes(self, num_classes):
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x, return_features=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)

        if return_features:
            return self.model.fc(features), features
        return self.model.fc(features)

    def get_feature_dim(self):
        return self.model.fc.in_features


class ResNet50(nn.Module):
    """ResNet-50 — used for ImageNet-LT and iNaturalist (224×224)."""
    def __init__(self, num_classes=1000, pretrained=False, image_size=224):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        if image_size <= 64:
            _adapt_resnet_for_cifar(self.model)
        self._update_num_classes(num_classes)

    def _update_num_classes(self, num_classes):
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x, return_features=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)

        if return_features:
            return self.model.fc(features), features
        return self.model.fc(features)

    def get_feature_dim(self):
        return self.model.fc.in_features


class ResNet101(nn.Module):
    """ResNet-101."""
    def __init__(self, num_classes=1000, pretrained=False, image_size=224):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(pretrained=pretrained)
        if image_size <= 64:
            _adapt_resnet_for_cifar(self.model)
        self._update_num_classes(num_classes)

    def _update_num_classes(self, num_classes):
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x, return_features=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)

        if return_features:
            return self.model.fc(features), features
        return self.model.fc(features)

    def get_feature_dim(self):
        return self.model.fc.in_features
