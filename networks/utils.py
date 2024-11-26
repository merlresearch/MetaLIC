# Copyright (C) 2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Authors: Ankush Chakrabarty and Gordon Wichern

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MetaLearnInputOutputDataset(Dataset):

    def __init__(self, data, n_traj, n_ics, win_len=10):
        super().__init__()

        self.y_data, self.u_data = data

        self.n_scenarios = n_ics
        self.traj_len = n_traj
        self.win_len = win_len

    def __len__(self):
        return self.n_scenarios

    def __getitem__(self, idx_ic):
        # Sampling query and context set indices randomly
        idx_query = np.random.randint(
            low=self.win_len, high=self.traj_len, size=1
        )[0]
        idx_context = np.random.randint(
            low=self.win_len, high=self.traj_len, size=1
        )[0]

        # Building context set of inputs (y[t-N:t-1], u[t-N:t-1], u[t])
        # and targets (y[t], y[t-N:t-1], u[t-N:t-1]) -> last two for reconstruction error in VAE
        context_y_hist = self.y_data[
            idx_context - self.win_len : idx_context, idx_ic, :
        ].flatten()
        context_u_hist = self.u_data[
            idx_context - self.win_len : idx_context
        ].flatten()
        context_u_curr = self.u_data[idx_context].flatten()
        context_inputs = torch.cat(
            (context_y_hist, context_u_hist, context_u_curr), dim=0
        )
        context_y_curr = self.y_data[idx_context, idx_ic, :].flatten()
        context_targets = context_y_curr

        # Building query set of inputs (y[t-N:t-1], u[t-N:t-1], u[t])
        # and targets (y[t], y[t-N:t-1], u[t-N:t-1])
        query_y_hist = self.y_data[
            idx_query - self.win_len : idx_query, idx_ic, :
        ].flatten()
        query_u_hist = self.u_data[
            idx_query - self.win_len : idx_query
        ].flatten()
        query_u_curr = self.u_data[idx_query].flatten()
        query_inputs = torch.cat(
            (query_y_hist, query_u_hist, query_u_curr), dim=0
        )
        query_y_curr = self.y_data[idx_query, idx_ic, :].flatten()
        query_targets = query_y_curr

        return context_inputs, query_inputs, context_targets, query_targets


class MetaTestInputOutputDataset(Dataset):

    def __init__(self, data, idx_ic, win_len=10):
        super().__init__()

        self.y_data, self.u_data = data
        self.idx_ic = idx_ic

        self.traj_len = data[0].shape[0]
        self.win_len = win_len

    def __len__(self):
        return self.traj_len - self.win_len

    def __getitem__(self, idx_context):

        # Building context set of inputs (y[t-N:t-1], u[t-N:t-1], u[t])
        # and targets (y[t], y[t-N:t-1], u[t-N:t-1]) -> last two for reconstruction error in VAE
        context_y_hist = self.y_data[
            idx_context : idx_context + self.win_len, self.idx_ic, :
        ].flatten()
        context_u_hist = self.u_data[
            idx_context : idx_context + self.win_len
        ].flatten()
        context_u_curr = self.u_data[idx_context + self.win_len].flatten()
        context_inputs = torch.cat(
            (context_y_hist, context_u_hist, context_u_curr), dim=0
        )
        context_y_curr = self.y_data[
            idx_context + self.win_len, self.idx_ic, :
        ].flatten()
        context_targets = context_y_curr

        return context_inputs, context_targets


def get_latest_file(directory):
    """Returns the path to the most recently modified '.pt' file in the given directory."""

    latest_pt_file = None
    latest_mtime = 0

    for filename in os.listdir(directory):
        if filename.endswith(".pt"):  # Check for '.pt' extension
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                mtime = os.path.getmtime(file_path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_pt_file = file_path

    return latest_pt_file
