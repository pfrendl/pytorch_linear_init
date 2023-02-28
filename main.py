import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        gain = math.sqrt(2)
        return gain * x.relu()


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        gain = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(gain * torch.randn((out_features, in_features)))
        self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(input=x, weight=self.weight, bias=self.bias)


def fill_axes(
    data_arr: list[Tensor],
    ax: plt.Axes,
) -> None:
    data = torch.stack(data_arr, dim=0)
    means = data.mean(dim=1)
    stds = data.std(dim=1)

    ax.plot(means.tolist(), color="orangered", label="Mean")
    ax.fill_between(
        x=np.arange(len(means)),
        y1=(means - stds / 2).detach().numpy(),
        y2=(means + stds / 2).detach().numpy(),
        facecolor="dodgerblue",
        alpha=0.1,
        edgecolor="dodgerblue",
        label="One standard deviation",
    )
    ax.grid(True, which="major", linestyle="--")
    ax.legend()


def test(
    test_name: str,
    linearity_ctr: Callable[[int, int], nn.Module],
    nonlinearity_ctr: Callable[[], nn.Module],
    batch_size: int,
    num_features: int,
    depth: int,
    save_dir: Path,
) -> None:
    x = torch.randn((batch_size, num_features))

    layers = [
        linearity_ctr(num_features, num_features),
        *[
            nn.Sequential(nonlinearity_ctr(), linearity_ctr(num_features, num_features))
            for _ in range(depth - 1)
        ],
    ]

    dim = 0
    means_arr = [x.mean(dim=dim)]
    stds_arr = [x.std(dim=dim)]
    with torch.no_grad():
        for layer in layers:
            x = layer(x)
            means_arr.append(x.mean(dim=dim))
            stds_arr.append(x.std(dim=dim))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    ax1.set_xlabel("Layers applied")
    ax1.set_ylabel("Distribution of feature means")
    fill_axes(data_arr=means_arr, ax=ax1)

    ax2.set_xlabel("Layers applied")
    ax2.set_ylabel("Distribution of feature standard deviations")
    fill_axes(data_arr=stds_arr, ax=ax2)

    plt.savefig(save_dir / f"plot_{test_name}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    save_dir = Path("outputs/")
    save_dir.mkdir(parents=True, exist_ok=True)

    tests = [
        ("default", nn.Linear, nn.Identity),
        ("custom", Linear, nn.Identity),
        ("default_relu", nn.Linear, nn.ReLU),
        ("custom_relu", Linear, ReLU),
    ]

    for test_name, linearity, nonlinearity in tests:
        test(
            test_name=test_name,
            linearity_ctr=linearity,
            nonlinearity_ctr=nonlinearity,
            batch_size=128,
            num_features=1024,
            depth=50,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    main()
