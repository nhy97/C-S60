from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ChannelReadout(nn.Module):

    def __init__(self, channels, width):
        super().__init__()
        self.channel_identity = nn.Parameter(torch.empty(1, channels, width))
        nn.init.normal_(self.channel_identity, std=0.02)
        self.norm = nn.LayerNorm(width)
        self.score = nn.Sequential(
            nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 1)
        )
        self.project = nn.Sequential(nn.Linear(width, width), nn.GELU())

    def forward(self, channel_latents):
        tokens = self.norm(channel_latents + self.channel_identity)
        weights = self.score(tokens).softmax(dim=1)
        return self.project((tokens * weights).sum(dim=1))


class PatchProjection(nn.Module):
    def __init__(self, length, history, width, dropout):
        super().__init__()
        self.length = int(length)
        self.left_padding = (-history) % self.length
        self.patch_count = (history + self.left_padding) // self.length
        self.value = nn.Linear(self.length, width)
        self.mask = nn.Linear(self.length, width, bias=False)
        self.position = nn.Parameter(torch.empty(1, self.patch_count, width))
        nn.init.normal_(self.position, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches, observed):
        token = self.value(patches) + self.mask(observed.to(patches.dtype)) + self.position
        valid = observed.any(dim=-1)
        token = self.dropout(token)
        return torch.where(valid[:, :, None], token, 0.0), valid


class TemporalTransformer(nn.Module):
    def __init__(self, width, heads, layers, multiplier, dropout):
        super().__init__()

        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                width,
                heads,
                dim_feedforward=multiplier * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            for _ in range(layers)
        )

    def forward(self, tokens, observed):

        safe = observed.clone()
        safe[:, 0] = safe[:, 0] | ~observed.any(-1)
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=~safe)
            tokens = torch.where(observed[:, :, None], tokens, 0.0)
        return tokens


class TemporalPool(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Linear(width, 1)
        self.project = nn.Sequential(nn.Linear(width * 2, width), nn.GELU())

    def forward(self, tokens, observed):
        scores = (
            self.score(tokens).squeeze(-1).masked_fill(~observed, -torch.finfo(tokens.dtype).max)
        )
        weights = scores.softmax(-1) * observed.to(tokens.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        attended = (tokens * weights[:, :, None]).sum(1)
        last_index = (torch.arange(tokens.shape[1], device=tokens.device)[None] * observed).amax(-1)
        last = tokens.gather(1, last_index[:, None, None].expand(-1, 1, tokens.shape[-1])).squeeze(
            1
        )
        result = self.project(torch.cat((attended, last), -1))
        return torch.where(observed.any(-1, keepdim=True), result, 0.0)


class ChannelIndependentPatchEncoder(nn.Module):
    def __init__(self, width=96, dropout=0.2):
        super().__init__()
        channels, history = 52, 60
        heads, layers, multiplier, lengths = 4, 2, 2, [15]
        self.channel_chunk_size = 4096
        self.activation_checkpointing = True
        self.channels, self.history, self.width = channels, history, width
        self.branches = nn.ModuleList(PatchProjection(p, history, width, dropout) for p in lengths)
        self.temporal_encoder = TemporalTransformer(width, heads, layers, multiplier, dropout)
        self.temporal_pool = TemporalPool(width)
        self.readout = ChannelReadout(channels, width)
        self.scale_fusion = nn.Sequential(
            nn.Linear(len(lengths) * width, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Dropout(dropout),
        )

    def _encode_chunk(self, branch, patches, patch_observed):
        tokens, valid = branch(patches, patch_observed)
        hidden = self.temporal_encoder(tokens, valid)
        return self.temporal_pool(hidden, valid)

    def channel_latents(self, values, observed, scale_index=0):

        branch = self.branches[scale_index]
        batch, channels, _ = values.shape
        values = torch.where(observed[:, None, :], values, 0.0)
        padded = F.pad(values, (branch.left_padding, 0))
        patches = padded.unfold(-1, branch.length, branch.length)
        patch_mask = F.pad(observed, (branch.left_padding, 0)).unfold(
            -1, branch.length, branch.length
        )
        patch_mask = patch_mask[:, None].expand(-1, channels, -1, -1)
        flat = patches.reshape(batch * channels, branch.patch_count, branch.length)
        flat_mask = patch_mask.reshape(batch * channels, branch.patch_count, branch.length)
        outputs = []
        function = partial(self._encode_chunk, branch)
        for start in range(0, len(flat), self.channel_chunk_size):
            x, valid = (
                flat[start : start + self.channel_chunk_size],
                flat_mask[start : start + self.channel_chunk_size],
            )
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                z = checkpoint(function, x, valid, use_reentrant=False, preserve_rng_state=True)
            else:
                z = function(x, valid)
            outputs.append(z)
        return torch.cat(outputs, dim=0).reshape(batch, channels, self.width)

    def forward(self, values, observed):
        scales = [
            self.readout(self.channel_latents(values, observed, i))
            for i in range(len(self.branches))
        ]
        return self.scale_fusion(torch.cat(scales, -1))


def _initialize_value_head(head):
    nn.init.normal_(head.weight, mean=0.0, std=0.001)
    nn.init.zeros_(head.bias)


class TemporalIntervalDecoder(nn.Module):

    def __init__(self, width, heads=4, dropout=0.1):
        super().__init__()
        self.width = width
        self.patch_projection = nn.Linear(52 * 5 + 5, width)
        self.age_embedding = nn.Linear(1, width)
        self.register_buffer(
            "patch_ages", torch.arange(-57.5, 0.0, 5.0, dtype=torch.float32).reshape(12, 1) / 60.0
        )
        self.register_buffer(
            "interval_descriptors", torch.tensor([[30.0, 45.0, 15.0], [45.0, 60.0, 15.0]]) / 60.0
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=2 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            layer, num_layers=1, norm=nn.LayerNorm(width), enable_nested_tensor=False
        )
        self.latent_projection = nn.Linear(width, width)
        self.interval_projection = nn.Linear(3, width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * width, width)
        )
        self.output_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.out_heads = nn.ModuleList(nn.Linear(width, 1) for _ in range(2))
        for head in self.out_heads:
            _initialize_value_head(head)

    def memory_tokens(self, values, observed):
        if (
            values.ndim != 3
            or tuple(values.shape[1:]) != (52, 60)
            or tuple(observed.shape) != (len(values), 60)
            or observed.dtype != torch.bool
        ):
            raise ValueError("C-S60 temporal memory requires AE[B,52,60] and bool observed[B,60]")

        clean = torch.where(observed[:, None], values, 0.0)
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("Nonfinite observed C-S60 temporal AE")
        patch_mask = observed.reshape(len(values), 12, 5)
        patches = clean.reshape(len(values), 52, 12, 5).permute(0, 2, 1, 3).flatten(2)
        tokens = self.patch_projection(torch.cat((patches, patch_mask.to(clean.dtype)), -1))
        tokens = tokens + self.age_embedding(self.patch_ages.to(clean.dtype))[None]
        valid = patch_mask.any(-1)
        tokens = torch.where(valid[:, :, None], tokens, 0.0)
        safe = valid.clone()
        safe[:, 0] = safe[:, 0] | ~valid.any(-1)
        memory = self.history_encoder(tokens, src_key_padding_mask=~safe)
        return torch.where(valid[:, :, None], memory, 0.0), valid

    def forward(self, latent, values, observed):
        if tuple(latent.shape) != (len(values), self.width):
            raise ValueError("C-S60 temporal queries require original pooled latent[B,width]")
        memory, valid = self.memory_tokens(values, observed)
        safe = valid.clone()
        safe[:, 0] = safe[:, 0] | ~valid.any(-1)
        available = valid.any(-1, keepdim=True)
        latent = torch.where(available, latent, 0.0)
        query = (
            self.latent_projection(latent)[:, None]
            + self.interval_projection(self.interval_descriptors.to(latent.dtype))[None]
        )
        attended, _ = self.attention(
            query, memory, memory, key_padding_mask=~safe, need_weights=False
        )
        hidden = self.attention_norm(query + self.dropout(attended))
        hidden = self.output_norm(hidden + self.dropout(self.ffn(hidden)))
        residuals = torch.cat(
            [head(hidden[:, index]) for index, head in enumerate(self.out_heads)], 1
        )
        return torch.where(available, residuals, 0.0)


class ContextTokenEncoder(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(53, width)
        self.position = nn.Linear(1, width)
        layer = nn.TransformerEncoderLayer(
            width, 4, width * 2, dropout=0.1, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.register_buffer(
            "relative_ages",
            (torch.arange(29, -1, -1, dtype=torch.float32) * 10.0 + 5.0)[:, None] / 300.0,
        )

    def forward(self, x, recent_latent):
        values, raw_mask, coverage = (
            x["context_history"],
            x["context_history_mask"],
            x["context_history_coverage"],
        )
        count = len(recent_latent)
        if (
            tuple(values.shape) != (count, 52, 30)
            or tuple(raw_mask.shape) != (count, 30)
            or tuple(coverage.shape) != (count, 30)
        ):
            raise ValueError("C-S60 context requires AE[B,52,30], mask[B,30] and coverage[B,30]")
        if not torch.is_floating_point(values) or not torch.is_floating_point(coverage):
            raise ValueError("C-S60 context AE and coverage must be floating point")
        if (
            values.device != recent_latent.device
            or raw_mask.device != values.device
            or coverage.device != values.device
        ):
            raise ValueError("C-S60 context and recent latent must share a device")
        if values.dtype != recent_latent.dtype or coverage.dtype != recent_latent.dtype:
            raise ValueError("C-S60 context and coverage must match recent latent floating dtype")
        allowed = torch.arange(30, device=values.device) >= 24

        raw_mask = torch.where(allowed[None], raw_mask, 0.0)
        coverage = torch.where(allowed[None], coverage, 0.0)
        if not bool(torch.isfinite(raw_mask).all()) or not bool(
            ((raw_mask == 0) | (raw_mask == 1)).all()
        ):
            raise ValueError("Finite binary C-S60 observed mask required")
        observed = raw_mask.bool()
        if bool((observed[:, :-1] & ~observed[:, 1:]).any()):
            raise ValueError("C-S60 context permits only left padding")
        if not bool(torch.isfinite(coverage).all()) or not bool(
            ((coverage >= 0.0) & (coverage <= 1.0)).all()
        ):
            raise ValueError("Finite C-S60 coverage in [0,1] required")
        if not bool(torch.equal(coverage > 0.0, observed)):
            raise ValueError("C-S60 positive coverage must match observed bins")
        clean = torch.where(observed[:, None], values, 0.0)
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("Nonfinite observed C-S60 context AE")
        tokens = self.project(torch.cat((clean.transpose(1, 2), coverage[..., None]), -1))
        tokens = tokens + self.position(self.relative_ages.to(dtype=values.dtype))[None]
        tokens = torch.where(observed[..., None], tokens, 0.0)
        safe = observed.clone()
        safe[~observed.any(1), 0] = True
        memory = self.encoder(tokens, src_key_padding_mask=~safe)
        return torch.where(observed[..., None], memory, 0.0), observed


class RoutedContextFusion(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.width = width
        self.context_encoder = ContextTokenEncoder(width)
        self.query_norm = nn.LayerNorm(width)
        self.interval_projection = nn.Linear(3, width)
        self.register_buffer(
            "interval_descriptors",
            torch.tensor(
                ((0.0, 15.0, 15.0), (15.0, 30.0, 15.0), (30.0, 60.0, 30.0)), dtype=torch.float32
            )
            / 60.0,
        )
        self.attention = nn.MultiheadAttention(width, 4, dropout=0.1, batch_first=True)
        self.residual_projection = nn.Linear(width, width)
        nn.init.normal_(self.residual_projection.weight, std=0.001)
        nn.init.zeros_(self.residual_projection.bias)
        self.gate = nn.Linear(width, width)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x, recent_latent, recent_observed):
        if tuple(recent_latent.shape) != (
            len(recent_latent),
            self.width,
        ) or not torch.is_floating_point(recent_latent):
            raise ValueError("C-S60 fusion requires floating recent latent[B,width]")
        if (
            tuple(recent_observed.shape) != (len(recent_latent), 60)
            or recent_observed.dtype != torch.bool
        ):
            raise ValueError("C-S60 fusion requires recent boolean observed[B,60]")
        if recent_observed.device != recent_latent.device:
            raise ValueError("C-S60 recent latent and mask must share a device")
        recent_available = recent_observed.any(1, keepdim=True)
        recent = torch.where(recent_available, recent_latent, 0.0)
        if not bool(torch.isfinite(recent).all()):
            raise ValueError("Nonfinite observed C-S60 recent latent")
        memory, observed = self.context_encoder(x, recent)
        safe = observed.clone()
        safe[~observed.any(1), 0] = True
        queries = (
            self.query_norm(recent)[:, None]
            + self.interval_projection(self.interval_descriptors.to(recent.dtype))[None]
        )
        attended, _ = self.attention(
            queries, memory, memory, key_padding_mask=~safe, need_weights=False
        )
        available = recent_available & observed.any(1, keepdim=True)
        residuals = torch.where(available[..., None], self.residual_projection(attended), 0.0)
        residuals = torch.sigmoid(self.gate(queries)) * residuals
        return recent, residuals
