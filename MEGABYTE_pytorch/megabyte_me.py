from collections import namedtuple

import torch
from torch import nn
import einops
from einops import rearrange
from einops.layers.torch import Rearrange

MegabyteConfig = namedtuple(
    "MegabyteConfig",
    ["V", "P", "D_G", "D_L", "T_MAX", "pad_token_id"]
)


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

        self.global_embedder = nn.Embedding(V, D_G)
        self.pos_embedder = nn.Embedding(config.T_MAX, D_G)
        # self.patch_embedder = nn.ModuleList([
        #     Rearrange("... P D_G -> ... (P D_G)"),
        # ])

    def forward(self, ids):
        B, T = ids.shape
        P = self.config.P
        K = T//P
        D_G = self.config.D_G

        tokens_embed = self.global_embedder(ids)
        position = torch.cat([torch.arange(T) for _ in range(B)]).reshape(B, T)
        position_embed = self.pos_embedder(position)
        h = tokens_embed + position_embed
        h = h.reshape((B, K, P, D_G))

        h = rearrange(h, "... P D_G -> ... (P D_G)", P=P, D_G=D_G)
        embed_global_pad = torch.ones((B, 1, P * D_G), dtype=torch.long) * self.config.pad_token_id
        h, ps = einops.pack([h, embed_global_pad], "B * d")

        return h


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
)

megabyte = Megabyte(config)
input_ids = torch.randint(0, 255, (B, T))
x = megabyte(input_ids)

print(x.shape, x.norm())
