from collections import OrderedDict

import torch
from torch import nn

from layers import ChannelIndependentPatchEncoder, RoutedContextFusion, TemporalIntervalDecoder


class CS60(nn.Module):
    def __init__(self, normalization):
        super().__init__()
        self.register_buffer(
            "delta_ratio",
            torch.tensor(normalization["delta_scale"], dtype=torch.float32)
            / normalization["target_std"],
        )
        self.encoder = ChannelIndependentPatchEncoder()
        self.head = nn.Linear(96, 8)
        with torch.no_grad():
            nn.init.normal_(self.head.weight[5:], std=0.01)
            self.head.bias[5:].zero_()
        with torch.random.fork_rng(devices=[]):
            self.split_head = nn.Linear(96, 1)
            nn.init.zeros_(self.split_head.weight)
            nn.init.zeros_(self.split_head.bias)
        with torch.random.fork_rng(devices=[]):
            self.decoder = TemporalIntervalDecoder(96)
        with torch.random.fork_rng(devices=[]):
            self.context = RoutedContextFusion(96)

    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != (60, 52):
            raise ValueError("Each forecast requires 60 completed seconds and 52 AE features")
        if history.dtype != torch.float32 or not torch.isfinite(history).all():
            raise ValueError("Use finite float32 training-standardized AE features")
        n = history.shape[0]
        observed = history.new_ones((n, 60), dtype=torch.bool)
        values = history.transpose(1, 2).contiguous()
        latent = self.encoder(values, observed)
        means = history.reshape(n, 6, 10, 52).double().mean(2).to(history.dtype)
        context = history.new_zeros(n, 52, 30)
        context[:, :, -6:] = means.transpose(1, 2)
        mask = history.new_zeros(n, 30)
        mask[:, -6:] = 1
        recent, corrections = self.context(
            {
                "context_history": context,
                "context_history_mask": mask,
                "context_history_coverage": mask,
            },
            latent,
            observed,
        )
        fused = recent + corrections.mean(1)
        direct = self.head(fused)
        fine = direct[:, :5]
        first = fine.mean(1, keepdim=True)
        coarse = torch.cat((first, first + direct[:, 5:] * self.delta_ratio), dim=1)
        split = self.split_head(fused).squeeze(1)
        short = torch.stack((coarse[:, 0], coarse[:, 1] - split, coarse[:, 1] + split), dim=1)
        far = coarse[:, 3:4] + self.decoder(fused, values, observed)
        delivered = torch.cat((coarse[:, :2], coarse[:, 2:3], far.mean(1, keepdim=True)), dim=1)
        return {
            "fine": fine,
            "delta": (delivered[:, 1:] - delivered[:, :1]) / self.delta_ratio,
            "short": short,
            "five": torch.cat((delivered[:, :3], far), dim=1),
        }


class Ensemble(nn.Module):
    def __init__(self, normalization):
        super().__init__()
        self.members = nn.ModuleList(CS60(normalization) for _ in range(3))

    def forward(self, history):
        return torch.stack([member(history)["five"] for member in self.members]).mean(0)


def read_archived_weights(model, path):
    mapping = (
        ("core.core.curve.base.encoder.core.", "encoder."),
        ("core.core.curve.base.", ""),
        ("core.temporal_decoder.", "decoder."),
        ("routed_context.", "context."),
    )
    original = torch.load(path, map_location="cpu", weights_only=True)
    renamed = OrderedDict()
    for name, parameter in original.items():
        member, index, suffix = name.split(".", 2)
        if member != "members" or index not in {"0", "1", "2"}:
            raise ValueError("Expected the three paper ensemble members")
        for old, new in mapping:
            if suffix.startswith(old):
                renamed[f"members.{index}.{new}{suffix[len(old):]}"] = parameter
                break
        else:
            raise ValueError(f"Unrecognized checkpoint parameter {name}")
    model.load_state_dict(renamed, strict=True)
