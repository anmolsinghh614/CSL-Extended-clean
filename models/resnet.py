import torch.nn as nn
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


class ResNet32(nn.Module):
    """ResNet-32 (wraps resnet34) — used for CIFAR-10/100 (32×32)."""
    def __init__(self, num_classes=1000, pretrained=False, image_size=32):
        super(ResNet32, self).__init__()
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
