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
META_INFERENCE_TRAIN_MAML = True  # Set to True for meta-inference adaptation, False for quick plots once meta-inference is done once.
META_INFERENCE_TRAIN_REPTILE = True

batch_size_maml = 64
dim_hidden = 256
dim_latent = 128
win_len = 20
lr = 1e-3
inner_loop_n_shot = (
    100  # Number of inner-loop iterations for meta-inference training
)

batch_size_reptile = 64
inner_lr_reptile = 1e-4
outer_lr_reptile = 1e-4
inner_loop_n_shot_reptile = 100

first_order = False  # True for FO-MAML, False for MAML

temp = sio.loadmat("data/test_data/BoucWen.BenchmarkDataset.mat")
n_ics = 1
y_data = torch.from_numpy(temp["data_train_y"]).T.unsqueeze(2) * 1e6
u_data = torch.from_numpy(temp["data_train_u"]).T

prcnt_traj = 0.2
ix_traj_offline = int(prcnt_traj * y_data.shape[0])
y_data_offline = y_data[:ix_traj_offline, ...]
u_data_offline = u_data[:ix_traj_offline, ...]

y_data_online = torch.from_numpy(temp["data_test_y"]).T.unsqueeze(2) * 1e6
u_data_online = torch.from_numpy(temp["data_test_u"]).T

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

dataloader = DataLoader(
    test_dataset, batch_size=batch_size_maml, shuffle=True, drop_last=False
)

# %% Set up network
if META_INFERENCE_TRAIN_MAML:
    if first_order:
        model = torch.load(
            "saved_weights/fomaml_v1_final.pt", map_location="cpu"
        )
    else:
        model = torch.load(
            "saved_weights/maml_v1_final.pt", map_location="cpu"
        )
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
            effective_batch_size = batch[0].shape[0]

            context_inputs = batch[0][0].unsqueeze(0).float()
            context_targets = batch[1][0].unsqueeze(0).float()

            support_loss = model.loss_fn(context_inputs, context_targets)[0]

            opt.zero_grad()
            support_loss.backward()
            opt.step()

            pbar.set_postfix_str(
                f"Meta-inference loss: {support_loss.item():.3e}"
            )

            meta_train_loss_epoch += support_loss.item() / effective_batch_size

        if meta_train_loss_epoch <= best_meta_inference_loss:
            best_meta_inference_loss = meta_train_loss_epoch
            if first_order:
                torch.save(
                    model, "saved_weights/benchmark/fomaml_v1_benchmark.pt"
                )
            else:
                torch.save(
                    model, "saved_weights/benchmark/maml_v1_benchmark.pt"
                )
            print(
                "\nIteration: %d||Avg. Train-Loss: %.4e|"
                % (epoch, meta_train_loss_epoch)
            )
            print("Saved weights!\n")

# %% Reptile Meta-Inference Training
model_reptile = MixedEncoderKoopmanNet(
    dim_in=dim_in,
    dim_hidden=dim_hidden,
    dim_latent=dim_latent,
    dim_out_x=dim_out_decx,
    dim_out_y=dim_out_decy,
    isDecoderLinear=False,
    isStateTransitionLinear=False,
)
if META_INFERENCE_TRAIN_REPTILE:
    model_reptile.load_state_dict(
        torch.load("saved_weights/competitors/reptile_metatrain.pth")
    )
    model_reptile.train()

    reptile = Reptile(
        model_reptile,
        lr_inner=inner_lr_reptile,
        lr_outer=outer_lr_reptile,
        num_inner_steps=inner_loop_n_shot_reptile,
        meta_batch_size=batch_size_maml,
        n_loss_epochs=None,
        num_meta_iterations=None,
    )
    reptile.meta_infer(test_dataset, batch_size_reptile)

    # Save the model
    torch.save(
        reptile.model.state_dict(),
        "saved_weights/benchmark/reptile_v1_benchmark.pt",
    )

# %% Testing on the remainder of the signal
if first_order:
    model = torch.load("saved_weights/benchmark/fomaml_v1_benchmark.pt")
else:
    model = torch.load("saved_weights/benchmark/maml_v1_benchmark.pt")
model_reptile.load_state_dict(
    torch.load("saved_weights/benchmark/reptile_v1_benchmark.pt")
)
model_uni = torch.load(
    "saved_weights/competitors/universal_learned_model.pt"
).eval()
model_sup_20 = torch.load(
    "saved_weights/competitors/supervised_learned_model.20percent.pt"
).eval()
model_sup_80 = torch.load(
    "saved_weights/competitors/supervised_learned_model.80percent.pt"
).eval()

test_dataset = MetaTestInputOutputDataset(
    data=(y_data_online, u_data_online), idx_ic=idx_ic, win_len=win_len
)

dataloader = DataLoader(
    test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=False
)

batch = next(iter(dataloader))
targets = batch[1].float()

for batch in dataloader:
    inputs = batch[0].float()

    # Test meta-learner MAML
    psi = model.encoder(
        inputs[..., :-1]
    )  # compute latent with residual encoder
    predictions = model.dec_y(psi)  # getting y from psi

    # Test Reptile
    psi_reptile = model_reptile.encoder(
        inputs[..., :-1]
    )  # compute latent with residual encoder
    predictions_reptile = model_reptile.dec_y(
        psi_reptile
    )  # getting y from psi

    # Test universal learner
    psi_uni = model_uni.encoder(
        inputs[..., :-1]
    )  # compute latent with residual encoder
    predictions_uni = model_uni.dec_y(psi_uni)  # getting y from psi

    # Test supervised learner
    psi_sup_20 = model_sup_20.encoder(
        inputs[..., :-1]
    )  # compute latent with residual encoder
    predictions_sup_20 = model_sup_20.dec_y(psi_sup_20)  # getting y from psi
    psi_sup_80 = model_sup_80.encoder(
        inputs[..., :-1]
    )  # compute latent with residual encoder
    predictions_sup_80 = model_sup_80.dec_y(psi_sup_80)  # getting y from psi

u_true = u_data_online.detach().cpu().numpy()
y_hat = predictions.detach().numpy().flatten() * 1e-6
y_hat_uni = predictions_uni.detach().numpy().flatten() * 1e-6
y_hat_sup_20 = predictions_sup_20.detach().numpy().flatten() * 1e-6
y_hat_sup_80 = predictions_sup_80.detach().numpy().flatten() * 1e-6
y_hat_reptile = predictions_reptile.detach().numpy().flatten() * 1e-6
y_true = targets.detach().numpy().flatten() * 1e-6
t = np.arange(0, len(y_hat)) * (1 / 750.0)

rmse = np.sqrt(np.mean((y_hat - y_true) ** 2))
temp = np.sqrt(np.mean((y_hat - y_true.mean()) ** 2))
model_fit = 100 * (1 - (rmse / temp))

rmse_reptile = np.sqrt(np.mean((y_hat_reptile - y_true) ** 2))
temp_reptile = np.sqrt(np.mean((y_hat_reptile - y_true.mean()) ** 2))
model_fit_reptile = 100 * (1 - (rmse_reptile / temp_reptile))

rmse_uni = np.sqrt(np.mean((y_hat_uni - y_true) ** 2))
temp_uni = np.sqrt(np.mean((y_hat_uni - y_true.mean()) ** 2))
model_fit_uni = 100 * (1 - (rmse_uni / temp_uni))

rmse_sup_20 = np.sqrt(np.mean((y_hat_sup_20 - y_true) ** 2))
temp_sup_20 = np.sqrt(np.mean((y_hat_sup_20 - y_true.mean()) ** 2))
model_fit_sup_20 = 100 * (1 - (rmse_sup_20 / temp_sup_20))

rmse_sup_80 = np.sqrt(np.mean((y_hat_sup_80 - y_true) ** 2))
temp_sup_80 = np.sqrt(np.mean((y_hat_sup_80 - y_true.mean()) ** 2))
model_fit_sup_80 = 100 * (1 - (rmse_sup_80 / temp_sup_80))

# %% Plotting
plt.figure(dpi=150)
plt.subplot(221)
plt.plot(t, y_true * 1e3, "bo", lw=0.75, label="Measured", markersize=1)
plt.plot(t, y_hat * 1e3, lw=0.75, color="k", label="Predicted [Meta]")
plt.ylabel(r"Displacement $y$ [mm]", fontsize="medium")
plt.xlabel(r"Time $t$ [s]", fontsize="medium")
plt.xlim([3.8, 4.7])
plt.ylim([-2.2, 2.5])
plt.subplot(212)
plt.plot(
    t,
    (y_hat - y_hat_reptile) * 1e3,
    lw=0.75,
    color="m",
    label="Error [Reptile]",
)
plt.plot(t, (y_hat - y_hat_uni) * 1e3, lw=0.75, color="c", label="Error [Uni]")
plt.plot(
    t,
    (y_hat - y_hat_sup_20) * 1e3,
    lw=0.75,
    color="r",
    label="Error [Sup/Tr20]",
)
plt.plot(
    t,
    (y_hat - y_hat_sup_80) * 1e3,
    lw=0.75,
    color="g",
    label="Error [Sup/Tr80]",
)
plt.plot(t, (y_hat - y_true) * 1e3, lw=0.75, color="k", label="Error [Meta]")
plt.legend(fontsize="small", ncols=5)
plt.title(
    f"[Fit] Reptile: {model_fit_reptile:.1f}|Uni: {model_fit_uni:.1f}|Sup/Tr20: {model_fit_sup_20:.1f}|Sup/Tr80: {model_fit_sup_80:.1f}|Meta: {model_fit:.1f}|[1]: 97.2",
    fontsize="medium",
)
plt.ylabel(r"Error $\varepsilon_y$ [mm]", fontsize="small")
plt.xlabel(r"Time $t$ [s]", fontsize="small")
plt.autoscale(tight=True)
plt.subplot(222)
plt.hist(
    ((y_hat - y_true) * 1e3),
    100,
    histtype="step",
    lw=1.0,
    color="k",
    label="MAML",
    density=True,
)
plt.hist(
    ((y_hat_reptile - y_true) * 1e3),
    100,
    histtype="step",
    lw=1.0,
    color="m",
    label="Reptile",
    density=True,
)
plt.hist(
    ((y_hat_sup_20 - y_true) * 1e3),
    100,
    histtype="step",
    lw=1.0,
    color="r",
    label="Sup/Tr20",
    density=True,
)
plt.hist(
    ((y_hat_sup_80 - y_true) * 1e3),
    100,
    histtype="step",
    lw=1.0,
    color="g",
    label="Sup/Tr80",
    density=True,
)
plt.xlabel(r"$\varepsilon_y\triangleq\hat{y} - y$", fontsize="small")
plt.ylabel(r"PDF($\,\varepsilon_y\,$)", fontsize="small")
plt.legend(ncols=1, loc="best", fontsize="x-small")
plt.tight_layout()
# plt.savefig("figures/fomaml.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Meta|RMSE: {rmse:.4e}|Model Fit: {model_fit:.2f}")
print(f"Reptile|RMSE: {rmse_reptile:.4e}|Model Fit: {model_fit_reptile:.2f}")
print(f"Uni|RMSE: {rmse_uni:.4e}|Model Fit: {model_fit_uni:.2f}")
print(f"Sup/Tr20|RMSE: {rmse_sup_20:.4e}|Model Fit: {model_fit_sup_20:.2f}")
print(f"Sup/Tr80|RMSE: {rmse_sup_80:.4e}|Model Fit: {model_fit_sup_80:.2f}")
