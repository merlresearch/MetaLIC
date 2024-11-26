# Copyright (C) 2021-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Authors: Ankush Chakrabarty and Gordon Wichern

import os

import numpy as np
import scipy.io as sio
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from networks.utils import MetaTestInputOutputDataset

if os.name == "nt":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

test = False
first_order = False

batch_size = 64
dim_hidden = 256
dim_latent = 128
win_len = 20
lr = 1e-3
inner_loop_n_shot = 40 if not first_order else 10

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

# %% Set up training data
test_dataset = MetaTestInputOutputDataset(
    data=(y_data_offline, u_data_offline), idx_ic=idx_ic, win_len=win_len
)

dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, drop_last=False
)

# %% Set up network
if first_order:
    model = torch.load("saved_weights/fomaml_v1_final.pt")
else:
    model = torch.load("saved_weights/maml_v1_final.pt")
model.train()

# %% Training meta-learner
""" Meta-Learning Loop """
del temp, y_data, u_data
opt = torch.optim.Adamax(model.parameters(), lr)

best_meta_inference_loss = 1e8
pbar = tqdm(range(inner_loop_n_shot))
for epoch in pbar:

    meta_train_loss_epoch = 0.0
    for iteration, batch in enumerate(dataloader):  # num_tasks/batch_size

        context_inputs = batch[0][0].unsqueeze(0).float()
        context_targets = batch[1][0].unsqueeze(0).float()

        support_loss = model.loss_fn(context_inputs, context_targets)[0]

        opt.zero_grad()
        support_loss.backward()
        opt.step()

        pbar.set_postfix_str(f"Meta-inference loss: {support_loss.item():.3e}")

        meta_train_loss_epoch += support_loss.item()

    if meta_train_loss_epoch <= best_meta_inference_loss:
        best_outer_loop_loss = meta_train_loss_epoch
        if first_order:
            torch.save(model, "saved_weights/metatest/fomaml_v1_test.pt")
        else:
            torch.save(model, "saved_weights/metatest/maml_v1_test.pt")
        print(
            "\nIteration: %d||Avg. Train-Loss: %.4e|"
            % (epoch, meta_train_loss_epoch)
        )
        print("Saved weights!\n")

# %% Testing on the remainder of the signal
model = torch.load("saved_weights/metatest/fomaml_v1_test.pt")
model.eval()

test_dataset = MetaTestInputOutputDataset(
    data=(y_data_online, u_data_online), idx_ic=idx_ic, win_len=win_len
)

dataloader = torch.utils.data.DataLoader(
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
