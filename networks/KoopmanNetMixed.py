# Copyright (C) 2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Authors: Ankush Chakrabarty and Gordon Wichern

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedEncoderKoopmanNet(nn.Sequential):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_latent,
        dim_out_x,
        dim_out_y,
        isDecoderLinear=False,
        isStateTransitionLinear=False,
    ):
        super().__init__()

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out_x = dim_out_x
        self.dim_out_y = dim_out_y
        self.dim_in = dim_in
        self.dim_latent = dim_latent
        self.dim_hidden = dim_hidden
        self.n_u = 1
        self.n_y = 1

        self.actfn = nn.ELU(0.1, inplace=True)

        self.encoder = torch.nn.Sequential(
            nn.Linear(self.dim_in, self.dim_hidden),
            self.actfn,
            nn.Linear(self.dim_hidden, self.dim_hidden // 2),
            self.actfn,
            nn.Linear(self.dim_hidden // 2, self.dim_hidden // 4),
            self.actfn,
            nn.Linear(self.dim_hidden // 4, self.dim_hidden // 4),
            self.actfn,
            nn.Linear(self.dim_hidden // 4, self.dim_hidden // 8),
            self.actfn,
            nn.Linear(self.dim_hidden // 8, self.dim_latent),
        )

        if isStateTransitionLinear:
            self.A_psi = torch.nn.Linear(
                self.dim_latent, self.dim_latent, bias=False
            )
            self.B_psi = torch.nn.Linear(self.n_u, self.dim_latent, bias=False)
        else:
            self.A_psi = torch.nn.Sequential(
                nn.Linear(self.dim_latent, self.dim_hidden),
                self.actfn,
                nn.Linear(self.dim_hidden, self.dim_hidden),
                self.actfn,
                nn.Linear(self.dim_hidden, self.dim_hidden),
                self.actfn,
                nn.Linear(self.dim_hidden, self.dim_hidden),
                self.actfn,
                nn.Linear(self.dim_hidden, self.dim_hidden),
                self.actfn,
                nn.Linear(self.dim_hidden, self.dim_latent),
            )

            self.B_psi = torch.nn.Sequential(
                nn.Linear(self.n_u, self.dim_hidden),
                self.actfn,
                nn.Linear(self.dim_hidden, self.dim_hidden),
                self.actfn,
                nn.Linear(self.dim_hidden, self.dim_latent),
            )

        if isDecoderLinear:
            self.dec_x = torch.nn.Linear(
                self.dim_latent, self.dim_out_x, bias=False
            )
        else:
            self.dec_x = torch.nn.Sequential(
                nn.Linear(self.dim_latent, self.dim_hidden // 4),
                self.actfn,
                nn.Linear(self.dim_hidden // 4, self.dim_hidden // 4),
                self.actfn,
                nn.Linear(self.dim_hidden // 4, self.dim_hidden // 4),
                self.actfn,
                nn.Linear(self.dim_hidden // 4, self.dim_hidden // 2),
                self.actfn,
                nn.Linear(self.dim_hidden // 2, self.dim_hidden),
                self.actfn,
                nn.Linear(self.dim_hidden, self.dim_out_x),
            )

        if isDecoderLinear:
            self.dec_y = torch.nn.Linear(
                self.dim_latent, self.dim_out_y, bias=False
            )
        else:
            self.dec_y = torch.nn.Sequential(
                nn.Linear(self.dim_latent, self.dim_hidden // 4),
                self.actfn,
                nn.Linear(self.dim_hidden // 4, self.dim_hidden // 4),
                self.actfn,
                nn.Linear(self.dim_hidden // 4, self.dim_hidden // 2),
                self.actfn,
                nn.Linear(self.dim_hidden // 2, self.dim_hidden // 2),
                self.actfn,
                nn.Linear(self.dim_hidden // 2, self.dim_hidden),
                self.actfn,
                nn.Linear(self.dim_hidden, self.dim_hidden),
                self.actfn,
                nn.Linear(self.dim_hidden, self.dim_out_y),
            )

    def init_weights(self):
        if isinstance(self, torch.nn.Linear):
            torch.nn.init.xavier_uniform(self.weight)
            self.bias.data.fill_(0.01)

    def forward(self, inputs):
        psi = self.encoder(
            inputs[:, :-1]
        )  # compute latent with residual encoder
        psi_plus = self.update_x(psi, inputs[:, -1])  # updating psi
        x_plus = self.dec_x(psi_plus)  # decoding updated latent
        y = self.dec_y(psi)  # getting y from psi
        return x_plus, y  # returning y and updated x

    def update_x(self, psi, u):
        return self.A_psi(psi) + self.B_psi(u)

    def loss_fn(self, inputs, targets):

        x_true = inputs[:, :-1]
        x_bar = self.dec_x(self.encoder(inputs[:, :-1]))

        # Prediction loss
        x_plus_pred, y_pred = self.forward(inputs)
        loss = F.mse_loss(targets, y_pred) + 1e-4 * F.mse_loss(x_bar, x_true)

        return loss, x_bar
