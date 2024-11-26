# Copyright (C) 2021-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Authors: Ankush Chakrabarty and Gordon Wichern

import os

import numpy as np
import scipy.io as sio
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks.KoopmanNetMixedReptile import MixedEncoderKoopmanNet
from networks.utils import MetaTestInputOutputDataset
from reptile.reptile import Reptile

if os.name == "nt":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

test = False
first_order = True

max_epochs = 10000 if not test else 10
batch_size = 16
dim_hidden = 256
dim_latent = 128
win_len = 20
inner_lr = 1e-4
outer_lr = 1e-4
inner_loop_n_shot = 5

test = False

temp = sio.loadmat(
    "data/train_data/BoucWen.MetaLearn.TrainTestDataset.Simple.mat"
)
n_ics = temp["y_hist_test"].shape[0]
y_data = torch.from_numpy(temp["y_hist_test"]).T.unsqueeze(2) * 1e6
u_data = torch.from_numpy(temp["u_final"])

prcnt_traj = 0.5
ix_traj_offline = int(prcnt_traj * y_data.shape[0])

y_data_offline = y_data[:ix_traj_offline, ...]
u_data_offline = u_data[:ix_traj_offline, ...]
y_data_online = y_data[ix_traj_offline:, ...]
u_data_online = u_data[ix_traj_offline:, ...]

n_trajLen = u_data.shape[0]
n_ics = y_data.shape[1]
idx_ic = torch.randint(0, n_ics, size=(1,))[0]

dim_y = 1
dim_u = 1

dim_in = (dim_u + dim_y) * win_len
dim_out_decy = dim_y
dim_out_decx = dim_in

# %% Set up meta-inference context data
test_dataset = MetaTestInputOutputDataset(
    data=(y_data_offline, u_data_offline), idx_ic=idx_ic, win_len=win_len
)

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
model.load_state_dict(
    torch.load("saved_weights/competitors/reptile_metatrain.pth")
)
model.train()

reptile = Reptile(
    model,
    lr_inner=inner_lr,
    lr_outer=outer_lr,
    num_inner_steps=inner_loop_n_shot,
    meta_batch_size=batch_size,
    n_loss_epochs=None,
    num_meta_iterations=None,
)
reptile.meta_infer(test_dataset)

# %% Evaluation loop
test_dataset = MetaTestInputOutputDataset(
    data=(y_data_online, u_data_online), idx_ic=idx_ic, win_len=win_len
)

dataloader = DataLoader(
    test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=False
)

batch = next(iter(dataloader))
targets = batch[1].float()
predictions = torch.empty_like(targets)

for batch in dataloader:
    for k in tqdm(range(batch[0].shape[0])):
        inputs = batch[0][k].unsqueeze(0).float()
        pred_y = model(inputs)[1]
        predictions[k] = pred_y

y_hat = predictions.detach().numpy().flatten() * 1e-6
y_true = targets.detach().numpy().flatten() * 1e-6
t = np.arange(0, len(y_hat)) * (1 / 750.0)

plt.figure()
plt.plot(t, y_hat, label="Predictions")
plt.plot(t, y_true, label="True")
plt.plot(t, y_hat - y_true, label="Error")
plt.tight_layout()
plt.show()

rmse = np.sqrt(np.mean((y_hat - y_true) ** 2))
print(f"RMSE: {rmse:.4e}")
