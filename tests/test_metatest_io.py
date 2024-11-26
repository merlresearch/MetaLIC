# Copyright (C) 2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch

from networks.utils import MetaTestInputOutputDataset


@pytest.fixture
def dataset():
    y_data = torch.randn(100, 10, 1)
    u_data = torch.randn(100, 1)
    return MetaTestInputOutputDataset(
        data=(y_data, u_data), idx_ic=0, win_len=10
    )


def test_len(dataset):
    assert len(dataset) == 90


def test_getitem(dataset):
    context_inputs, context_targets = dataset[0]
    assert context_inputs.shape[0] == 21
    assert context_targets.shape[0] == 1
