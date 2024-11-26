# Copyright (C) 2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch

from networks.utils import MetaLearnInputOutputDataset


@pytest.fixture
def dataset():
    y_data = torch.randn(100, 10, 1)
    u_data = torch.randn(100, 1)
    return MetaLearnInputOutputDataset(
        data=(y_data, u_data), n_traj=100, n_ics=10, win_len=10
    )


def test_len(dataset):
    assert len(dataset) == 10


def test_getitem(dataset):
    context_inputs, query_inputs, context_targets, query_targets = dataset[0]
    assert context_inputs.shape[0] == 21
    assert query_inputs.shape[0] == 21
    assert context_targets.shape[0] == 1
    assert query_targets.shape[0] == 1
