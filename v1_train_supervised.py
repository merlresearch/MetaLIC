# Copyright (C) 2021-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Authors: Ankush Chakrabarty and Gordon Wichern

import scipy.io as sio
import torch
from tqdm import tqdm

from networks.KoopmanNetMixed import MixedEncoderKoopmanNet
from networks.utils import MetaLearnInputOutputDataset

test = False
first_order = True if test else False

dt = 1 / 750.0

max_epochs = 10000 if not test else 10
batch_size = 64
dim_hidden = 256
dim_latent = 128
win_len = 20

temp = sio.loadmat("data/test_data/BoucWen.BenchmarkDataset.mat")
n_ics = 1
y_data = torch.from_numpy(temp["data_train_y"]).T.unsqueeze(2) * 1e6
u_data = torch.from_numpy(temp["data_train_u"]).T

prcnt_traj = 0.8
ix_traj_offline = int(prcnt_traj * y_data.shape[0])

y_data = y_data[:ix_traj_offline, ...]
u_data = u_data[:ix_traj_offline, ...]

n_trajLen = u_data.shape[0]
n_initCondn = y_data.shape[1]

dim_y = 1
dim_u = 1

dim_in = (dim_u + dim_y) * win_len
dim_out_decy = dim_y
dim_out_decx = dim_in

# %% Set up dataloader
train_dataset = MetaLearnInputOutputDataset(
    data=(y_data, u_data), n_traj=n_trajLen, n_ics=n_initCondn, win_len=win_len
)

dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
)

# %% Set up network
lr = 1e-3

model = MixedEncoderKoopmanNet(
    dim_in=dim_in,
    dim_hidden=dim_hidden,
    dim_latent=dim_latent,
    dim_out_x=dim_out_decx,
    dim_out_y=dim_out_decy,
    isDecoderLinear=False,
    isStateTransitionLinear=False,
)
opt = torch.optim.Adamax(model.parameters(), lr)

pbar = tqdm(range(max_epochs))
support_loss_best = 1e8
model.train()
for _ in pbar:

    support_loss_total = 0
    for iteration, batch in enumerate(dataloader):  # num_tasks/batch_size

        context_inputs = batch[0][iteration].unsqueeze(0).float()
        context_targets = batch[2][iteration].unsqueeze(0).float()

        support_loss = model.loss_fn(context_inputs, context_targets)[0]

        opt.zero_grad()
        support_loss.backward()
        opt.step()

        support_loss_total += support_loss.item()

    if support_loss_best >= support_loss_total:
        support_loss_best = support_loss_total
        torch.save(
            model,
            f"saved_weights/supervised_learned_model.{int(prcnt_traj*100)}percent.pt",
        )
        print("Saved weights!\n")

    pbar.set_postfix_str(f"Training loss: {support_loss_best:.6e}")
