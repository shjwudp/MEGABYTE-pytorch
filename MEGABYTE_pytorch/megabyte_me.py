from collections import namedtuple

import torch
from torch import nn
import einops
from einops import rearrange
from einops.layers.torch import Rearrange
from MEGABYTE_pytorch.attend import Attend

MegabyteConfig = namedtuple(
    "MegabyteConfig",
    [
        "V", "P", "D_G", "D_L", "T_MAX", "pad_token_id",
        "g_nheads", "g_nlayers",
        "l_nheads", "l_nlayers",
    ]
)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.attend = Attend(
            causal = True,
            flash = flash,
            dropout = dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, attn_bias=None):
        h = self.heads

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))
        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        out = self.attend(q, k, v, attn_bias=attn_bias)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim),
    )


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        flash_attn = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=flash_attn),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x):
        n = x.shape[-2]
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x) + x

        print(x.shape)
        return self.norm(x)


class Megabyte(nn.Module):
    """
    notation
    V - vocabulary size
    P - patch size
    D_G - global dimension
    D_L - local dimension
    T - sequence length
    """

    def __init__(
        self,
        config: MegabyteConfig,
    ):
        super().__init__()
        self.config = config
        V = config.V
        D_G = config.D_G

        self.g_embedder = nn.Embedding(V, D_G)
        self.g_pos_embedder = nn.Embedding(config.T_MAX, D_G)
        self.g_transformer = Transformer(
            dim=config.P*config.D_G,
            layers=config.g_nlayers,
            dim_head=(config.P*config.D_G)//config.g_nheads,
            heads=config.g_nheads,
        )
        self.gl_linear = nn.Sequential(
            Rearrange("... (P D_G) -> ... P D_G", P=P, D_G=D_G),
            nn.Linear(D_G, D_L),
            Rearrange("... P D_L -> (...) P D_L", P=P, D_L=D_L),
        )

        self.l_embedder = nn.Embedding(V, D_L)
        self.l_pos_embedder = nn.Embedding(config.P, D_L)
        self.l_transformer = Transformer(
            dim=config.D_L,
            layers=config.l_nlayers,
            dim_head=config.D_L//config.l_nheads,
            heads=config.l_nheads,
        )

    def patch_embed(self, *, ids, embedder, pos_embedder):
        b, s, u = ids.shape

        # embedding = tokens embedding + absolute position embedding
        tokens_embed = embedder(rearrange(ids, "... s u -> ... (s u)", s=s, u=u))
        pos = torch.cat([torch.arange(s*u) for _ in range(b)]).reshape(b, s*u)
        pos_embed = pos_embedder(pos)
        h = tokens_embed + pos_embed

        # add spatial tokens
        d = h.shape[-1]
        h = rearrange(h, "... (s u) d -> ... s (u d)", s=s, u=u, d=d)
        embed_pad = torch.ones((b, 1, u*d), dtype=torch.long) * self.config.pad_token_id
        h, _ = einops.pack([h, embed_pad], "b * d")

        return h[:, :s, :]

    def forward(self, ids):
        """
        ids - input ids, shape [B, T]
        """
        B, T = ids.shape
        P = self.config.P
        K = T//P
        D_G = self.config.D_G

        global_in = self.patch_embed(
            ids=rearrange(ids, "... (K P) -> ... K P", K=K, P=P),
            embedder=self.g_embedder, pos_embedder=self.g_pos_embedder
        )
        print(global_in.shape)
        global_out = self.g_transformer(global_in)

        local_in = self.gl_linear(global_out) + self.patch_embed(
            ids=rearrange(ids, "B (K P) -> (B K) P 1", B=B, K=K, P=P),
            embedder=self.l_embedder, pos_embedder=self.l_pos_embedder
        )
        local_out = self.l_transformer(local_in)

        return local_out


V = 256
P = 4
D_G = 512
D_L = 128
T = 1024
B = 2

config = MegabyteConfig(
    V=V,
    P=P,
    D_G=D_G,
    D_L=D_L,
    T_MAX=T,
    pad_token_id=0,
    g_nlayers=4,
    g_nheads=16,
    l_nlayers=2,
    l_nheads=8,
)

megabyte = Megabyte(config)
input_ids = torch.randint(0, 255, (B, T))
x = megabyte(input_ids)

print(x.shape, x.norm())
