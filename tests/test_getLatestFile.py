# Copyright (C) 2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import tempfile

import pytest

from networks.utils import get_latest_file


@pytest.fixture
def test_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file1.pt")
        file2 = os.path.join(tmpdir, "file2.pt")
        with open(file1, "w") as f:
            f.write("test")
        with open(file2, "w") as f:
            f.write("test")
        yield tmpdir


def test_get_latest_file(test_dir):
    latest_file = get_latest_file(test_dir)
    assert latest_file in [
        os.path.join(test_dir, "file1.pt"),
        os.path.join(test_dir, "file2.pt"),
    ]
