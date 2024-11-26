# Copyright (C) 2021-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Authors: Ankush Chakrabarty and Gordon Wichern

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# Reptile meta-learning algorithm
class Reptile:
    def __init__(
        self,
        model,
        lr_inner,
        lr_outer,
        num_inner_steps,
        meta_batch_size,
        num_meta_iterations,
        n_loss_epochs,
    ):
        self.model = model
        self.task_model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.num_inner_steps = num_inner_steps
        self.meta_batch_size = meta_batch_size
        self.num_meta_iterations = num_meta_iterations
        self.n_loss_epochs = n_loss_epochs
        self.filename = "saved_weights/competitors/reptile_metatrain.pth"

    def _train_single_task(self, query_inputs, query_targets):
        self.task_model.load_state_dict(self.model.state_dict())
        optimizer = optim.Adam(self.task_model.parameters(), lr=self.lr_inner)

        # Perform inner-loop training on the given task
        self.task_model.train()
        for step in range(self.num_inner_steps):
            optimizer.zero_grad()
            loss, _ = self.task_model.loss_fn(
                query_inputs.float(), query_targets.float()
            )
            loss.backward()
            optimizer.step()

        return self.task_model, loss.item()

    def meta_train(self, meta_train_data):

        pbar = tqdm(range(self.num_meta_iterations))
        avg_loss = 0
        avg_loss_best = 1e9
        for epoch in pbar:
            meta_gradients = {
                name: torch.zeros_like(param)
                for name, param in self.model.named_parameters()
            }

            # Iterate over the meta-batch tasks
            for _ in range(self.meta_batch_size):
                task_data = DataLoader(
                    meta_train_data,
                    batch_size=32,
                    shuffle=True,
                    drop_last=False,
                )
                _, query_inputs, _, query_targets = next(iter(task_data))
                task_model, loss = self._train_single_task(
                    query_inputs, query_targets
                )
                avg_loss += loss / self.meta_batch_size

                # Accumulate weight updates
                for name, param in task_model.named_parameters():
                    meta_gradients[name] += (
                        (param.data - self.model.state_dict()[name])
                        / self.lr_inner
                        / self.meta_batch_size
                    )

            # Update meta model using averaged gradients
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.copy_(
                        self.model.state_dict()[name]
                        - self.lr_outer * meta_gradients[name]
                    )

            # Add average loss to tqdm via postfix
            pbar.set_postfix(loss=avg_loss)

            if epoch > 1 and epoch % self.n_loss_epochs == 0:
                if avg_loss_best > avg_loss:
                    avg_loss_best = avg_loss
                    torch.save(self.model.state_dict(), self.filename)
                    print(
                        f"Model saved at epoch {epoch} with loss {avg_loss_best / self.n_loss_epochs}"
                    )
                avg_loss = 0

    def meta_infer(self, meta_infer_data, batch_size=128):
        meta_gradients = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
        }

        task_data = DataLoader(
            meta_infer_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        for _ in tqdm(task_data):
            context_inputs, context_targets = next(iter(task_data))
            task_model, _ = self._train_single_task(
                context_inputs, context_targets
            )

            # Accumulate weight updates
            for name, param in task_model.named_parameters():
                meta_gradients[name] += (
                    param.data - self.model.state_dict()[name]
                ) / self.lr_inner

        # Update meta model using averaged gradients
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(
                    self.model.state_dict()[name]
                    - self.lr_outer * meta_gradients[name]
                )
