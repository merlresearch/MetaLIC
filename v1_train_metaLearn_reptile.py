# Copyright (C) 2021-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Authors: Ankush Chakrabarty and Gordon Wichern

import scipy.io as sio
import torch

from networks.KoopmanNetMixedReptile import MixedEncoderKoopmanNet
from networks.utils import MetaLearnInputOutputDataset
from reptile.reptile import Reptile

test = False
filename = "saved_weights/competitors/reptile_metatrain.pth"

max_epochs = 10000 if not test else 10
batch_size = 16
dim_hidden = 256
dim_latent = 128
win_len = 20
inner_lr = 1e-4
outer_lr = 1e-4
inner_loop_n_shot = 40

test = False

temp = sio.loadmat(
    "data/train_data/BoucWen.MetaLearn.TrainTestDataset.Simple.mat"
)
n_ics = temp["y_hist_train"].shape[0]
y_data = torch.from_numpy(temp["y_hist_train"]).T.unsqueeze(2) * 1e6
u_data = torch.from_numpy(temp["u_final"])

n_trajLen = u_data.shape[0]
n_initCondn = y_data.shape[1]

dim_y = 1
dim_u = 1

dim_in = (dim_u + dim_y) * win_len
dim_out_decy = dim_y
dim_out_decx = dim_in

# %% Set up training data
meta_train_dataset = MetaLearnInputOutputDataset(
    data=(y_data, u_data), n_traj=n_trajLen, n_ics=n_initCondn, win_len=win_len
)
n_loss_epochs = min([100, len(y_data) // batch_size])
print(f"Number of loss epochs for REPTILE: {n_loss_epochs}")

# %% Set up network
model = MixedEncoderKoopmanNet(
    dim_in=dim_in,
    dim_hidden=dim_hidden,
    dim_latent=dim_latent,
    dim_out_x=dim_out_decx,
    dim_out_y=dim_out_decy,
    isDecoderLinear=False,
    isStateTransitionLinear=False,
)

# %% Training meta-learner
""" Meta-Learning Loop """
del temp, y_data, u_data

reptile = Reptile(
    model,
    lr_inner=inner_lr,
    lr_outer=outer_lr,
    num_inner_steps=inner_loop_n_shot,
    meta_batch_size=batch_size,
    n_loss_epochs=n_loss_epochs,
    num_meta_iterations=max_epochs,
)

reptile.meta_train(meta_train_dataset)
