import torch
import torch.nn as nn
import torch.nn.functional as F


class StraightThroughThreshold(nn.Module):
    """Hard threshold in forward pass; straight-through gradient in backward pass."""
    def __init__(self, threshold: float = 0.3):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hard = (x >= self.threshold).to(x.dtype)
        return hard.detach() - x.detach() + x


class SobelEdgeResponse(nn.Module):
    """Channel-wise Sobel edge response, reduced to a single spatial map."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        kx = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]]
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]]
        ).view(1, 1, 3, 3)

        self.register_buffer("kx", kx, persistent=False)
        self.register_buffer("ky", ky, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        kx = self.kx.to(dtype=x.dtype).expand(c, 1, 3, 3)
        ky = self.ky.to(dtype=x.dtype).expand(c, 1, 3, 3)

        gx = F.conv2d(x, kx, padding=1, groups=c)
        gy = F.conv2d(x, ky, padding=1, groups=c)

        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + self.eps)
        return mag.mean(dim=1, keepdim=True)


class DSRU(nn.Module):
    """
    Depth-Aware Split-Reconstruct Unit from the supplied DSR-Conv diagram.
    """
    def __init__(self, channels: int, threshold: float = 0.3):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(
                f"DSRU requires an even number of channels, got {channels}."
            )

        self.edge_response = SobelEdgeResponse()
        self.importance_head = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.threshold = StraightThroughThreshold(threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FE: feature energy
        feature_energy = x.abs().mean(dim=1, keepdim=True)

        # ER: Sobel edge response
        edge_response = self.edge_response(x)

        # learned importance map
        w = self.sigmoid(
            self.importance_head(
                torch.cat([edge_response, feature_energy], dim=1)
            )
        )

        # threshold -> complementary masks W1/W2
        w1 = self.threshold(w)
        w2 = 1.0 - w1

        # weighted spatial separation
        x1w = x * w1
        x2w = x * w2

        # 1:1 channel split
        x11w, x12w = torch.chunk(x1w, 2, dim=1)
        x21w, x22w = torch.chunk(x2w, 2, dim=1)

        # cross reconstruction exactly as shown in the diagram
        y1 = x11w + x22w
        y2 = x12w + x21w

        return torch.cat([y1, y2], dim=1)


class ChannelAttention(nn.Module):
    """
    Diagram branch:
        Input -> Average Pooling -> FC -> FC -> channel weights
    The weights are applied to the DSRU output.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg_pool(x)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        return self.sigmoid(w)


class CRU(nn.Module):
    """
    Channel Reduction Unit, kept consistent with the supplied original ScConv code:
        split -> squeeze
        upper: GWConv + PWConv
        lower: PWConv + residual concat
        global pooling -> SoftMax -> weighted fusion
    """
    def __init__(
        self,
        op_channel: int,
        alpha: float = 1 / 2,
        squeeze_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
    ):
        super().__init__()

        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1).")

        self.up_channel = int(alpha * op_channel)
        self.low_channel = op_channel - self.up_channel

        up_s = self.up_channel // squeeze_ratio
        low_s = self.low_channel // squeeze_ratio

        if up_s < 1 or low_s < 1:
            raise ValueError("Channels are too small for squeeze_ratio.")

        self.squeeze1 = nn.Conv2d(
            self.up_channel, up_s, kernel_size=1, bias=False
        )
        self.squeeze2 = nn.Conv2d(
            self.low_channel, low_s, kernel_size=1, bias=False
        )

        self.gwconv = nn.Conv2d(
            up_s,
            op_channel,
            kernel_size=group_kernel_size,
            stride=1,
            padding=group_kernel_size // 2,
            groups=group_size,
            bias=True,
        )
        self.pwconv1 = nn.Conv2d(
            up_s, op_channel, kernel_size=1, bias=False
        )

        self.pwconv2 = nn.Conv2d(
            low_s,
            op_channel - low_s,
            kernel_size=1,
            bias=False,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        upper, lower = torch.split(
            x, [self.up_channel, self.low_channel], dim=1
        )

        upper = self.squeeze1(upper)
        lower = self.squeeze2(lower)

        y1 = self.gwconv(upper) + self.pwconv1(upper)
        y2 = torch.cat([self.pwconv2(lower), lower], dim=1)

        out = torch.cat([y1, y2], dim=1)

        weights = F.softmax(self.pool(out), dim=1)
        out = weights * out

        out1, out2 = torch.chunk(out, 2, dim=1)
        return out1 + out2


class SpatialAttention(nn.Module):
    """
    Diagram branch:
        F_t -> Max Pooling / Average Pooling -> concat -> Conv -> spatial weights
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        x = torch.cat([max_map, avg_map], dim=1)
        return self.sigmoid(self.conv(x))


class DSRConv(nn.Module):
    def __init__(
        self,
        channels: int,
        threshold: float = 0.3,
        ca_reduction: int = 16,
        alpha: float = 1 / 2,
        squeeze_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
        spatial_kernel_size: int = 7,
        shortcut: bool = False,
    ):
        super().__init__()

        self.shortcut = shortcut

        self.dsru = DSRU(
            channels=channels,
            threshold=threshold,
        )

        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction=ca_reduction,
        )

        self.cru = CRU(
            op_channel=channels,
            alpha=alpha,
            squeeze_ratio=squeeze_ratio,
            group_size=group_size,
            group_kernel_size=group_kernel_size,
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # top DSRU branch
        fs = self.dsru(x)

        # lower channel-attention branch originates from Input
        wc = self.channel_attention(x)

        # multiplication after DSRU
        ft = fs * wc

        # CRU main branch
        fc = self.cru(ft)

        # spatial attention originates from F_t (the purple feature in diagram)
        ws = self.spatial_attention(ft)

        # final multiplication
        out = fc * ws

        if self.shortcut:
            out = out + identity

        return out


# Optional compatibility alias
DSR_Conv = DSRConv


if __name__ == "__main__":
    x = torch.randn(1, 32, 16, 16, requires_grad=True)

    model = DSRConv(
        channels=32,
        threshold=0.3,
        shortcut=False,
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

    # Verify STE lets gradients reach the learned importance head
    y.mean().backward()
    grad = model.dsru.importance_head.weight.grad
    print("Importance-head gradient mean:", grad.abs().mean().item())
