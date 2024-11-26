# Copyright (C) 2021-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Authors: Ankush Chakrabarty and Gordon Wichern

import scipy.io as sio
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from maml import maml
from networks.KoopmanNetMixed import MixedEncoderKoopmanNet
from networks.utils import MetaLearnInputOutputDataset

test = False
first_order = False

max_epochs = 10000 if not test else 10
batch_size = 16
dim_hidden = 256
dim_latent = 128
win_len = 20
inner_lr = 1e-3
outer_lr = 1e-3
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
train_dataset = MetaLearnInputOutputDataset(
    data=(y_data, u_data), n_traj=n_trajLen, n_ics=n_initCondn, win_len=win_len
)

dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
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

# %% Training meta-learner
""" Meta-Learning Loop """
del temp, y_data, u_data

maml = maml.MAML(model, lr=outer_lr, first_order=first_order)
opt = torch.optim.Adamax(model.parameters(), inner_lr)
outer_lr_scheduler = ReduceLROnPlateau(
    opt, mode="min", patience=1000, factor=0.1, verbose=True
)

model.train()
best_outer_loop_loss = 1e8
pbar = tqdm(range(max_epochs))
for epoch in pbar:
    meta_train_loss_epoch = 0.0

    # for each iteration
    for iteration, batch in enumerate(dataloader):  # num_tasks/batch_size
        meta_train_loss = 0.0

        # for each task in the batch
        effective_batch_size = batch[0].shape[0]
        for i in range(effective_batch_size):
            learner = maml.clone()

            # divide the data into support and query sets
            context_inputs = batch[0][i].unsqueeze(0).float()
            context_targets = batch[2][i].unsqueeze(0).float()

            query_inputs = batch[1][i].unsqueeze(0).float()
            query_targets = batch[3][i].unsqueeze(0).float()

            for _ in range(inner_loop_n_shot):  # adaptation_steps
                support_loss = model.loss_fn(context_inputs, context_targets)[
                    0
                ]
                learner.adapt(
                    support_loss, allow_nograd=True, allow_unused=True
                )

            query_loss = model.loss_fn(query_inputs, query_targets)[0]
            meta_train_loss += query_loss

        meta_train_loss = meta_train_loss / effective_batch_size
        meta_train_loss_epoch += meta_train_loss

        # outer-loop update
        opt.zero_grad()
        meta_train_loss.backward()
        opt.step()

    # Saving the best model
    pbar.set_postfix_str(
        "Meta-loss: %.3e|Best Meta-Loss: %.3e"
        % (meta_train_loss_epoch, best_outer_loop_loss)
    )
    if meta_train_loss_epoch <= best_outer_loop_loss:
        best_outer_loop_loss = meta_train_loss_epoch
        if first_order:
            torch.save(model, f"saved_weights/fomaml_v1_epoch{epoch:06d}.pt")
        else:
            torch.save(model, f"saved_weights/maml_v1_epoch{epoch:06d}.pt")
        print(
            "\nIteration: %d||Avg. Train-Loss: %.4e|"
            % (epoch, meta_train_loss_epoch)
        )
        print("Saved weights!\n")

    outer_lr_scheduler.step(best_outer_loop_loss)
