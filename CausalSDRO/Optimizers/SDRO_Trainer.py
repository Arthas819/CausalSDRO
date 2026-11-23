"""
    This file solves SDRO by CM-SCSC with one group per joint observation.
"""

from Optimizers.CSDRO_Trainer import *


class SDRO_Trainer(CSDRO_Trainer):

    def train_scsc(self, X_train, Y_train, pro_params):
        sdro_params = dict(pro_params, causal_groups=False)
        return super().train_scsc(X_train, Y_train, sdro_params)
