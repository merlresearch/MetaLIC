# Copyright (c) 2019 Debajyoti Datta, Ian Bunner, Praateek Mahajan, Sebastien Arnold
#
# SPDX-License-Identifier: MIT

from torch import nn


class BaseLearner(nn.Module):

    def __init__(self, module=None):
        super().__init__()
        self.module = module

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return getattr(self.__dict__["_modules"]["module"], attr)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
