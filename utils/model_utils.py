import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vit_b_16, ViT_B_16_Weights, vgg16_bn, VGG16_BN_Weights

# ============== LeNet Model Definitions ==============
class LeNetClient(nn.Module):
    """Client-side feature extractor (convolutional layers) for LeNet."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        return self.features(x)


class LeNetServer(nn.Module):
    """Server-side classifier (fully connected layers) for LeNet."""

    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class LeNetFull(nn.Module):
    """Full LeNet model."""

    def __init__(self, num_classes):
        super().__init__()
        self.client_model = LeNetClient()
        self.server_model = LeNetServer(num_classes)

    def forward(self, x):
        x = self.client_model(x)
        return self.server_model(x)


# ============== ResNet18 Model Definitions ==============
class ResNet18Client(nn.Module):
    """Client-side ResNet18 (layers 1-3) with ImageNet pretrained weights."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Adapt conv1 for CIFAR-10 (32x32 images)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(resnet.conv1.weight[:, :, 2:5, 2:5])

        self.bn1 = resnet.bn1
        self.relu = resnet.relu

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class ResNet18Server(nn.Module):
    """Server-side ResNet18 (layer 4 + classifier)."""

    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNet18Full(nn.Module):
    """Full ResNet18 model adapted for CIFAR-10."""

    def __init__(self, num_classes):
        super().__init__()
        self.client_model = ResNet18Client()
        self.server_model = ResNet18Server(num_classes)

    def forward(self, x):
        x = self.client_model(x)
        return self.server_model(x)


class ResidualAdapter(nn.Module):
    """Residual adapter module for feature refinement."""

    def __init__(self, channels=256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        nn.init.zeros_(self.adapter[-1].weight)

    def forward(self, x):
        return x + self.adapter(x)


# ============== VGG16 Model Definitions (Tiny-ImageNet) ==============
class VGG16Client(nn.Module):
    """Client-side VGG16 feature extractor."""

    def __init__(self):
        super().__init__()
        original_vgg = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        self.features = original_vgg.features

    def forward(self, x):
        return self.features(x)


class VGG16Server(nn.Module):
    """Server-side VGG16 classifier with pretrained weights."""

    def __init__(self, num_classes):
        super().__init__()
        original_vgg = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)

        # Inherit pooling layer
        self.avgpool = original_vgg.avgpool

        # Inherit classifier (with pretrained weights)
        self.classifier = original_vgg.classifier

        # Replace the last layer to match the target number of classes
        in_features = self.classifier[6].in_features
        self.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG16Full(nn.Module):
    """Full VGG16 model."""

    def __init__(self, num_classes):
        super().__init__()
        self.client_model = VGG16Client()
        self.server_model = VGG16Server(num_classes)

    def forward(self, x):
        x = self.client_model(x)
        return self.server_model(x)


# ============== ViT-B/16 Model Definitions ==============
class ViTClient(nn.Module):
    """Client-side Vision Transformer (first N encoder layers)."""

    def __init__(self, cut_layer=6):
        super().__init__()
        vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        self.patch_embed = vit.conv_proj
        self.class_token = vit.class_token
        self.positional_encoding = vit.encoder.pos_embedding
        self.dropout = vit.encoder.dropout
        self.encoder_layers = nn.ModuleList([vit.encoder.layers[i] for i in range(cut_layer)])

    def forward(self, x):
        # x shape: [B, C, H, W]
        x = self.patch_embed(x)  # [B, 768, 14, 14]
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 768]
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x = x + self.positional_encoding
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x)
        return x


class ViTServer(nn.Module):
    """Server-side Vision Transformer (remaining encoder layers + head)."""

    def __init__(self, num_classes, cut_layer=6):
        super().__init__()
        vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        self.encoder_layers = nn.ModuleList(
            [vit.encoder.layers[i] for i in range(cut_layer, len(vit.encoder.layers))]
        )
        self.ln = vit.encoder.ln
        self.heads = nn.Sequential()
        self.heads.add_module("head", nn.Linear(vit.hidden_dim, num_classes))

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.ln(x)
        return self.heads(x[:, 0])


class ViTFull(nn.Module):
    """Full Vision Transformer model."""

    def __init__(self, num_classes, cut_layer=6):
        super().__init__()
        self.client_model = ViTClient(cut_layer=cut_layer)
        self.server_model = ViTServer(num_classes, cut_layer=cut_layer)

    def forward(self, x):
        x = self.client_model(x)
        return self.server_model(x)


# ========== Model Factory ==========
def get_models(model_name):
    """
    Get model classes by name.

    Args:
        model_name: Name of the model architecture.
            Supported: 'lenet', 'resnet18', 'vgg16', 'vit-b'

    Returns:
        Tuple of (ClientModel, ServerModel, FullModel) classes.
    """
    if model_name == "lenet":
        return LeNetClient, LeNetServer, LeNetFull
    elif model_name == "resnet18":
        return ResNet18Client, ResNet18Server, ResNet18Full
    elif model_name == "vgg16":
        return VGG16Client, VGG16Server, VGG16Full
    elif model_name == "vit-b":
        return ViTClient, ViTServer, ViTFull
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from ['lenet', 'resnet18', 'vgg16', 'vit-b']"
        )
