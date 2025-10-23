import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.nn.quantized.functional as qF
except ImportError:  # pragma: no cover - quantization backend missing
    qF = None  # type: ignore[assignment]

try:
    from torch.ao.quantization.fake_quantize import (
        FakeQuantize,
        default_fake_quant,
        default_weight_fake_quant,
    )

    _HAS_FAKE_QUANT = True
except (ImportError, AttributeError):  # pragma: no cover - optional dependency
    FakeQuantize = None  # type: ignore[assignment]

    def default_fake_quant(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("torch.ao.quantization.fake_quantize is required for QAT.")

    def default_weight_fake_quant(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("torch.ao.quantization.fake_quantize is required for QAT.")

    _HAS_FAKE_QUANT = False


class KANLinear(nn.Module):
    """
    Linear layer that augments standard activations with learnable spline bases.

    This version supports quantization-aware training (QAT) via ``enable_qat`` and can
    export an ``QuantizedKANLinear`` module with int8 weights for inference.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: nn.Module = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.grid_range = grid_range

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        else:
            self.register_parameter("spline_scaler", None)

        self._spline_feature_dim = in_features * (grid_size + spline_order)
        self._total_feature_dim = in_features + self._spline_feature_dim

        self.qat_enabled = False
        self.activation_fake_quant: Optional[FakeQuantize] = None
        self.feature_fake_quant: Optional[FakeQuantize] = None
        self.weight_fake_quant: Optional[FakeQuantize] = None
        self.output_fake_quant: Optional[FakeQuantize] = None

        self.reset_parameters()

    # --------------------------------------------------------------------- Utils
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline and self.spline_scaler is not None:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if self.enable_standalone_scale_spline and self.spline_scaler is not None:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight * self.scale_spline

    def _combined_weight(self) -> torch.Tensor:
        base = self.base_weight
        spline = self.scaled_spline_weight.view(self.out_features, -1)
        return torch.cat([base, spline], dim=1)

    def _combined_features(self, x: torch.Tensor) -> torch.Tensor:
        base_features = self.base_activation(x)
        spline_features = self.b_splines(x).view(x.size(0), -1)
        return torch.cat([base_features, spline_features], dim=1)

    # -------------------------------------------------------------------- QAT API
    def enable_qat(
        self,
        activation_fake_quant: Optional[FakeQuantize] = None,
        feature_fake_quant: Optional[FakeQuantize] = None,
        weight_fake_quant: Optional[FakeQuantize] = None,
        output_fake_quant: Optional[FakeQuantize] = None,
    ) -> "KANLinear":
        if not _HAS_FAKE_QUANT:
            raise RuntimeError("Fake-quantization support is not available in this PyTorch build.")
        if self.qat_enabled:
            return self
        self.activation_fake_quant = activation_fake_quant or default_fake_quant()
        self.feature_fake_quant = feature_fake_quant or default_fake_quant()
        self.weight_fake_quant = weight_fake_quant or default_weight_fake_quant()
        self.output_fake_quant = output_fake_quant or default_fake_quant()
        self.qat_enabled = True
        return self

    def disable_qat(self) -> "KANLinear":
        self.qat_enabled = False
        self.activation_fake_quant = None
        self.feature_fake_quant = None
        self.weight_fake_quant = None
        self.output_fake_quant = None
        return self

    def is_qat_active(self) -> bool:
        return self.qat_enabled

    # -------------------------------------------------------------------- Forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        if self.qat_enabled and self.activation_fake_quant is not None:
            x = self.activation_fake_quant(x)

        features = self._combined_features(x)
        if self.qat_enabled and self.feature_fake_quant is not None:
            features = self.feature_fake_quant(features)

        weight = self._combined_weight()
        if self.qat_enabled and self.weight_fake_quant is not None:
            weight = self.weight_fake_quant(weight)

        output = F.linear(features, weight)
        if self.qat_enabled and self.output_fake_quant is not None:
            output = self.output_fake_quant(output)

        return output.reshape(*original_shape[:-1], self.out_features)

    # ----------------------------------------------------------------- Splines IO
    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    # -------------------------------------------------------------------- Export
    def export_int8(self) -> "QuantizedKANLinear":
        if not self.qat_enabled:
            raise RuntimeError(
                "QAT must be enabled before exporting. Call enable_qat() and train/calibrate."
            )
        if not _HAS_FAKE_QUANT:
            raise RuntimeError("Fake-quantization support is not available in this PyTorch build.")
        if qF is None:
            raise RuntimeError("torch.nn.quantized.functional is required for int8 export.")

        if any(
            module is None
            for module in (
                self.feature_fake_quant,
                self.weight_fake_quant,
                self.output_fake_quant,
            )
        ):
            raise RuntimeError("Fake-quant modules are not initialised; run enable_qat() first.")

        weight = self._combined_weight().detach()
        weight_scale, weight_zero_point = self.weight_fake_quant.calculate_qparams()
        weight_scale_f = float(weight_scale.item())
        weight_zero_point_i = int(weight_zero_point.item())

        weight_q = torch.quantize_per_tensor(weight, weight_scale_f, weight_zero_point_i, torch.qint8)

        feature_scale, feature_zero_point = self.feature_fake_quant.calculate_qparams()
        output_scale, output_zero_point = self.output_fake_quant.calculate_qparams()

        activation_scale = activation_zero_point = None
        if self.activation_fake_quant is not None:
            activation_scale, activation_zero_point = self.activation_fake_quant.calculate_qparams()

        return QuantizedKANLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            grid=self.grid,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
            base_activation=copy.deepcopy(self.base_activation),
            weight_q=weight_q,
            feature_scale=float(feature_scale.item()),
            feature_zero_point=int(feature_zero_point.item()),
            output_scale=float(output_scale.item()),
            output_zero_point=int(output_zero_point.item()),
            activation_scale=(
                float(activation_scale.item()) if activation_scale is not None else None
            ),
            activation_zero_point=(
                int(activation_zero_point.item()) if activation_zero_point is not None else None
            ),
        )

    # ----------------------------------------------------------- Grid management
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    # --------------------------------------------------------------- Regularizer
    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> torch.Tensor:
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class QuantizedKANLinear(nn.Module):
    """
    Quantized inference module produced by :meth:`KANLinear.export_int8`.

    The heavy lifting (B-spline feature construction) is still performed in float,
    but the linear projection uses int8 weights and activations.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        grid: torch.Tensor,
        grid_size: int,
        spline_order: int,
        base_activation: nn.Module,
        weight_q: torch.Tensor,
        feature_scale: float,
        feature_zero_point: int,
        output_scale: float,
        output_zero_point: int,
        activation_scale: Optional[float] = None,
        activation_zero_point: Optional[int] = None,
    ):
        super().__init__()
        if qF is None:
            raise RuntimeError("torch.nn.quantized.functional is required for quantized inference.")
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.register_buffer("grid", grid.clone())
        self.base_activation = base_activation
        self.register_buffer("weight", weight_q)
        self.feature_scale = float(feature_scale)
        self.feature_zero_point = int(feature_zero_point)
        self.output_scale = float(output_scale)
        self.output_zero_point = int(output_zero_point)
        self.activation_scale = float(activation_scale) if activation_scale is not None else None
        self.activation_zero_point = int(activation_zero_point) if activation_zero_point is not None else None
        self._spline_feature_dim = in_features * (grid_size + spline_order)
        self._total_feature_dim = in_features + self._spline_feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        if self.activation_scale is not None and self.activation_zero_point is not None:
            x = torch.quantize_per_tensor(
                x, self.activation_scale, self.activation_zero_point, torch.quint8
            ).dequantize()

        features = self._combined_features(x)
        features_q = torch.quantize_per_tensor(
            features, self.feature_scale, self.feature_zero_point, torch.quint8
        )
        output_q = qF.linear(features_q, self.weight, bias=None, scale=self.output_scale, zero_point=self.output_zero_point)
        output = output_q.dequantize()
        return output.reshape(*original_shape[:-1], self.out_features)

    def _combined_features(self, x: torch.Tensor) -> torch.Tensor:
        base_features = self.base_activation(x)
        spline_features = self.b_splines(x).view(x.size(0), -1)
        return torch.cat([base_features, spline_features], dim=1)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        return bases.contiguous()

    def update_grid(self, *_args, **_kwargs) -> None:
        raise RuntimeError("QuantizedKANLinear does not support update_grid.")

    def regularization_loss(self, *_args, **_kwargs) -> torch.Tensor:
        raise RuntimeError("QuantizedKANLinear does not support regularization_loss.")


class KAN(nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid: bool = False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

    def enable_qat(self) -> "KAN":
        enable_kan_qat(self)
        return self

    def disable_qat(self) -> "KAN":
        disable_kan_qat(self)
        return self

    def convert_to_int8(self) -> "KAN":
        convert_kan_to_int8(self)
        return self


def enable_kan_qat(module: nn.Module) -> nn.Module:
    """
    Recursively enable quantization-aware training for every ``KANLinear`` module in ``module``.
    """
    for m in module.modules():
        if isinstance(m, KANLinear):
            m.enable_qat()
    return module


def disable_kan_qat(module: nn.Module) -> nn.Module:
    """
    Recursively disable quantization-aware training for every ``KANLinear`` module in ``module``.
    """
    for m in module.modules():
        if isinstance(m, KANLinear):
            m.disable_qat()
    return module


def convert_kan_to_int8(module: nn.Module) -> nn.Module:
    """
    Recursively replace every ``KANLinear`` module in ``module`` with its quantized counterpart.
    """
    if isinstance(module, nn.ModuleList):
        for idx, child in enumerate(module):
            if isinstance(child, KANLinear):
                module[idx] = child.export_int8()
            else:
                convert_kan_to_int8(child)
        return module

    if isinstance(module, nn.Sequential):
        for idx, child in enumerate(module):
            if isinstance(child, KANLinear):
                module[idx] = child.export_int8()
            else:
                convert_kan_to_int8(child)
        return module

    for name, child in list(module.named_children()):
        if isinstance(child, KANLinear):
            setattr(module, name, child.export_int8())
        else:
            convert_kan_to_int8(child)
    return module
