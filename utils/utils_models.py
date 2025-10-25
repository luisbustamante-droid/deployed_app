# utils/utils_models.py
import torch
import torch.nn as nn
import torchvision.models as tvm

def _set_classifier_out(in_features: int, num_classes: int) -> nn.Linear:
    layer = nn.Linear(in_features, num_classes)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer

def build_model(arch: str, num_classes: int = 5, pretrained: bool = False, dropout_rate: float = 0.3) -> nn.Module:
    arch = arch.lower()

    if arch == "resnet50":
        try:
            weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
            model = tvm.resnet50(weights=weights)
        except Exception:
            model = tvm.resnet50(pretrained=pretrained)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_f, num_classes))
        return model

    if arch in {"mobilenet_v3_large", "mobilenetv3-large", "mobilenetv3_large"}:
        try:
            weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            model = tvm.mobilenet_v3_large(weights=weights)
        except Exception:
            model = tvm.mobilenet_v3_large(pretrained=pretrained)
        model.classifier[2].p = dropout_rate
        model.classifier[-1] = nn.Linear(1280, num_classes)
        return model

    if arch in {"efficientnetv2_b0", "efficientnet-v2-b0", "efficientnet_v2_b0"}:
        # Preferir timm también en inferencia para mantener compatibilidad exacta
        try:
            import timm
            model = timm.create_model(
                "efficientnetv2_b0",
                pretrained=False,  # False, pero misma arquitectura timm
                num_classes=num_classes,
                drop_rate=dropout_rate
            )
            return model
        except Exception:
            # Fallback torchvision
            try:
                weights = tvm.EfficientNet_V2_B0_Weights.DEFAULT if pretrained else None
                model = tvm.efficientnet_v2_b0(weights=weights)
            except Exception:
                model = tvm.efficientnet_b0(pretrained=pretrained)
        in_f = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_f, num_classes))
        return model

    raise ValueError(f"Arquitectura no soportada: {arch}")
