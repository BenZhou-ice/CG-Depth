import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    Spatial attention branch in CGRU:
        (F1 + F2)
          -> channel-wise MaxPool / AvgPool
          -> Concat
          -> 7x7 Conv
        Output: [B, 1, H, W]
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        avg_map = torch.mean(x, dim=1, keepdim=True)
        return self.conv(torch.cat([max_map, avg_map], dim=1))


class ChannelAttention(nn.Module):
    """
    Channel attention branch in CGRU:
        (F1 + F2)
          -> Global Average Pooling
          -> 1x1 Conv
          -> activation
          -> 1x1 Conv
        Output: [B, C, 1, 1]
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ContentGuidedFusion(nn.Module):
    """
    Content-guided attentive feature fusion corresponding to Eqs. (14)-(16):

        F_sum = F1 + F2
        Ws = SA(F_sum)
        Wc = CA(F_sum)

        A = broadcast(Ws + Wc)
        W = sigmoid(
              GroupConv7x7(
                concat(A, F_sum)
              )
            )

        Fw = F1 + F2 + F1 * W + F2 * (1 - W)

    The 7x7 fusion convolution uses groups=C, so each output channel
    combines the corresponding attention/context channel pair.
    """
    def __init__(
        self,
        channels: int,
        ca_reduction: int = 16,
        kernel_size: int = 7,
    ):
        super().__init__()

        self.channels = channels
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)
        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction=ca_reduction
        )

        self.weight_conv = nn.Conv2d(
            in_channels=2 * channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        features_1: torch.Tensor,
        features_2: torch.Tensor
    ):
        if features_1.shape != features_2.shape:
            raise ValueError(
                f"features_1 and features_2 must have identical shapes, "
                f"got {tuple(features_1.shape)} and {tuple(features_2.shape)}."
            )

        f_sum = features_1 + features_2

        # Eq. (14)
        ws = self.spatial_attention(f_sum)     # [B, 1, H, W]
        wc = self.channel_attention(f_sum)     # [B, C, 1, 1]

        # Standard broadcasting -> [B, C, H, W]
        attention = ws + wc

        # Eq. (15)
        weight_input = torch.cat([attention, f_sum], dim=1)
        w = self.sigmoid(self.weight_conv(weight_input))

        # Eq. (16)
        fw = (
            features_1
            + features_2
            + features_1 * w
            + features_2 * (1.0 - w)
        )

        return fw, w


class DepthwiseConv(nn.Module):
    """Depthwise convolution used by the gated multi-scale branches."""
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GatedMultiScaleAggregation(nn.Module):
    """
    Bottom half of the CGRU diagram.

    Gate branch:
        Fw
          -> 5x5 DWConv
          -> Split [1/2, 3/8, 1/8]
             * branch 1: identity
             * branch 2: 5x5 DWConv
             * branch 3: 7x7 DWConv
          -> Concat
          -> 1x1 Conv
          -> SiLU
          -> Gate

    Value branch:
        Fw
          -> 1x1 Conv
          -> SiLU
          -> Value

    Output:
        Fo = Conv1x1(Value * Gate)
    """
    def __init__(self, channels: int):
        super().__init__()

        if channels % 8 != 0:
            raise ValueError(
                f"CGRU requires channels divisible by 8 for the "
                f"1/2, 3/8, 1/8 split, got {channels}."
            )

        self.channels = channels
        self.c_direct = channels // 2
        self.c_5x5 = 3 * channels // 8
        self.c_7x7 = channels // 8

        # Pre-split depthwise convolution
        self.pre_dwconv = DepthwiseConv(channels, kernel_size=5)

        # Multi-scale branches
        self.branch_5x5 = DepthwiseConv(self.c_5x5, kernel_size=5)
        self.branch_7x7 = DepthwiseConv(self.c_7x7, kernel_size=7)

        # Gate projection
        self.gate_proj = nn.Conv2d(
            channels, channels, kernel_size=1, bias=True
        )
        self.gate_act = nn.SiLU()

        # Value branch
        self.value_proj = nn.Conv2d(
            channels, channels, kernel_size=1, bias=True
        )
        self.value_act = nn.SiLU()

        # Final output projection
        self.out_proj = nn.Conv2d(
            channels, channels, kernel_size=1, bias=True
        )

    def forward(self, fw: torch.Tensor):
        # Gate branch
        x = self.pre_dwconv(fw)

        f_direct, f_5x5, f_7x7 = torch.split(
            x,
            [self.c_direct, self.c_5x5, self.c_7x7],
            dim=1
        )

        f_5x5 = self.branch_5x5(f_5x5)
        f_7x7 = self.branch_7x7(f_7x7)

        multi_scale = torch.cat(
            [f_direct, f_5x5, f_7x7],
            dim=1
        )

        gate = self.gate_act(self.gate_proj(multi_scale))

        # Value branch
        value = self.value_act(self.value_proj(fw))

        # Eq. (20)
        out = self.out_proj(value * gate)

        return out, gate, value


class CGRU(nn.Module):
    """
    Content-Guided Gated Recurrent Unit core shown in the supplied paper figure.
    """
    def __init__(
        self,
        channels: int,
        ca_reduction: int = 16,
        attention_kernel_size: int = 7,
    ):
        super().__init__()

        self.fusion = ContentGuidedFusion(
            channels=channels,
            ca_reduction=ca_reduction,
            kernel_size=attention_kernel_size,
        )

        self.aggregation = GatedMultiScaleAggregation(
            channels=channels
        )

    def forward(
        self,
        features_1: torch.Tensor,
        features_2: torch.Tensor,
        return_intermediates: bool = False,
    ):
        fw, w = self.fusion(features_1, features_2)
        out, gate, value = self.aggregation(fw)

        if return_intermediates:
            return {
                "output": out,
                "fusion_weight": w,
                "fused_feature": fw,
                "gate": gate,
                "value": value,
            }

        return out


if __name__ == "__main__":
    # Paper channel allocation requires C % 8 == 0.
    f1 = torch.randn(2, 64, 24, 80, requires_grad=True)
    f2 = torch.randn(2, 64, 24, 80, requires_grad=True)

    model = CGRU(
        channels=64,
        ca_reduction=16,
        attention_kernel_size=7,
    )

    result = model(
        f1,
        f2,
        return_intermediates=True
    )

    print("features_1   :", tuple(f1.shape))
    print("features_2   :", tuple(f2.shape))
    print("fusion weight:", tuple(result["fusion_weight"].shape))
    print("fused feature:", tuple(result["fused_feature"].shape))
    print("gate          :", tuple(result["gate"].shape))
    print("value         :", tuple(result["value"].shape))
    print("output        :", tuple(result["output"].shape))

    # Gradient check
    loss = result["output"].mean()
    loss.backward()

    grad = model.fusion.weight_conv.weight.grad
    print("fusion weight-conv grad mean:", grad.abs().mean().item())
