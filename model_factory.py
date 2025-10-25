# ============================================================
# model_factory.py
# Factoría de modelos (ResNet50 / MobileNetV3-Large / EfficientNetV2-B0)
# ============================================================

import torch
import torch.nn as nn
import torchvision.models as tvm


def _set_classifier_out(in_features: int, num_classes: int) -> nn.Linear:
    """Devuelve una capa lineal inicializada para el clasificador final."""
    layer = nn.Linear(in_features, num_classes)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def build_model(arch: str, num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """Crea un modelo CNN con pesos preentrenados y clasificador adaptado."""
    arch = arch.lower()

    # ---------- ResNet-50 ----------
    if arch == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.resnet50(weights=weights)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            _set_classifier_out(in_f, num_classes)
        )
        return model

    # ---------- MobileNet V3-Large ----------
    if arch in {"mobilenet_v3_large", "mobilenetv3-large", "mobilenetv3_large"}:
        weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.mobilenet_v3_large(weights=weights)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = _set_classifier_out(in_f, num_classes)
        return model

    # ---------- EfficientNet V2-B0 ----------
    if arch in {"efficientnetv2_b0", "efficientnet-v2-b0", "efficientnet_v2_b0"}:
        try:
            import timm
            model = timm.create_model(
                "efficientnetv2_b0",
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=0.3
            )
            return model
        except ImportError:
            # Fallback a torchvision si no hay timm
            weights = tvm.EfficientNet_V2_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = tvm.efficientnet_v2_b0(weights=weights)
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = _set_classifier_out(in_f, num_classes)
            return model

    # ---------- Arquitectura no reconocida ----------
    raise ValueError(f"Arquitectura no soportada: {arch}")


# ---------- Inicialización del dispositivo ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device activo: {device}")
