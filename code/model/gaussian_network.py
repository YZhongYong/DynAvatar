from model.embedder import *
import torch.nn as nn


class GaussianNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            multires_pnts=0,
    ):
        super().__init__()

        dims = [d_in + feature_vector_size] + dims
        self.d_in = d_in
        self.embedview_fn = None
        self.embedpnts_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        if multires_pnts > 0:
            embedpnts_fn, input_ch_pnts = get_embedder(multires_pnts)
            self.embedpnts_fn = embedpnts_fn
            dims[0] += (input_ch_pnts - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus(beta=100)
        self.scaling_layer = nn.Sequential(nn.Linear(64, 64), self.relu,
                                           nn.Linear(64, 2))
        self.rotations_layer = nn.Sequential(nn.Linear(64, 64), self.relu,
                                             nn.Linear(64, 4))
        self.opacity_layer = nn.Sequential(nn.Linear(64, 64), self.relu,
                                           nn.Linear(64, 1))
        self.color_layer = nn.Sequential(nn.Linear(64, 64), self.relu,
                                           nn.Linear(64, 3))

    def forward(self, offset):
        x = offset

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)

        offset_s = self.scaling_layer(x)
        offset_r = self.rotations_layer(x)
        offset_o = self.opacity_layer(x)
        offset_c = self.color_layer(x)
        return offset_s, offset_r, offset_o, offset_c