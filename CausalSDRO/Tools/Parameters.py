"""
    This file includes fixed parameters for DRO problems.
"""

'''
    Fixed parameters for Newsvendor
'''

class Parameters_News:
    def __init__(self, ):
        # Batch size
        self.BATCH_SIZE = 64
        # Learning rate for SAA method
        self.LR_SAA = 0.05
        self.LR_KLDRO = 0.03
        # # SAA training sample size
        # self.N1_SAA = 100
        # self.N2_SAA = 100
        # self.N3_SAA = 100
        # Evaluation sample size during training (from training set)
        self.N1_EVAL = 40
        self.N2_EVAL = 20
        self.N3_EVAL = 20
        # Epoches
        self.EPOCHS_SAA = 1000
        self.EPOCHS_SCSC = 1000
        # Gradient Clip for SCSC
        self.grad_clip = 4
        # Use Adam optimizer to solve SCSC
        self.use_adam = False
        # Stable loss
        self.scalar = 50

    # Epochs
    def setEpochs(self, EPOCHS_SAA, EPOCHS_SCSC):
        self.EPOCHS_SAA = EPOCHS_SAA
        self.EPOCHS_SCSC = EPOCHS_SCSC

    # Learning Rates for SCSC
    def setLearningRate(self, LR_alpha, LR_beta):
        self.LR_alpha = LR_alpha
        self.LR_beta = LR_beta

    # Structure of 2NN and SRF
    def setModelStrcuture(self, hidden_layer_dim, SRF_depth, Tree_number, Use_Linear_Leaf, Use_Diverse_SRT):
        self.hidden_layer_dim = hidden_layer_dim
        self.SRF_depth = SRF_depth
        self.Tree_number = Tree_number
        self.Use_Linear_Leaf = Use_Linear_Leaf
        self.Use_Diverse_SRT = Use_Diverse_SRT

'''
    Fixed parameters for Inventory Substitute problem 
'''

class Parameters_Inv:
    def __init__(self, ):
        # Inventory loss and decision bounds
        self.mu = 0.01
        self.B = 50.0
        self.a = self.B / 50
        self.x_bound = 5.0
        self.z_bound = 30.0
        self.alpha_scale = 40.0
        self.beta_scale = 1200 ** 0.5
        # Plain SGD for the Inventory kernel-smoothing baseline.
        self.K_SMOOTH = 1200
        self.BATCH_SIZE_SMOOTH = 6
        self.LR_SMOOTH = 40.0 / self.K_SMOOTH ** 0.5
        # Batch size
        self.BATCH_SIZE = 60
        # Learning rate for SAA method
        self.LR_SAA = 0.05
        self.LR_KLDRO = 0.02
        self.LR_CWDRO = 0.09
        # SAA training sample size
        self.N1_SAA = 60
        self.N2_SAA = 60
        self.N3_SAA = 60
        # Evaluation sample size during training (from training set)
        self.N1_EVAL = 20
        self.N2_EVAL = 20
        self.N3_EVAL = 20
        # Epoches
        self.EPOCHS_SAA = 300
        self.EPOCHS_SCSC = 1200
        # Gradient Clip for SCSC
        self.grad_clip = None
        # Use Adam optimizer to solve SCSC
        self.use_adam = False
        # Stable loss
        self.scalar = 80

    # Epochs
    def setEpochs(self, EPOCHS_SAA, EPOCHS_SCSC):
        self.EPOCHS_SAA = EPOCHS_SAA
        self.EPOCHS_SCSC = EPOCHS_SCSC

    # Learning Rates for SCSC
    def setLearningRate(self, LR_alpha, LR_beta):
        self.LR_alpha = LR_alpha
        self.LR_beta = LR_beta

    # Structure of 2NN and SRF
    def setModelStrcuture(self, hidden_layer_dim, SRF_depth, Tree_number, Use_Linear_Leaf, Use_Diverse_SRT):
        self.hidden_layer_dim = hidden_layer_dim
        self.SRF_depth = SRF_depth
        self.Tree_number = Tree_number
        self.Use_Linear_Leaf = Use_Linear_Leaf
        self.Use_Diverse_SRT = Use_Diverse_SRT

    # Reuse default CV parameters at every problem scale.
    def setCSDROParameters(self, n_batches, row=None):
        if row is not None and 'K' in row:
            self.EPOCHS_SCSC = (int(row['K']) + n_batches - 1) // n_batches
        K = self.EPOCHS_SCSC * n_batches
        alpha_scale = self.alpha_scale if row is None else row.get('best_alpha_scale', self.alpha_scale)
        beta_scale = self.beta_scale if row is None else row.get('best_beta_scale', self.beta_scale)
        beta = beta_scale / K ** 0.5
        if alpha_scale <= 0 or not 0 < beta <= 1 + 1e-12:
            raise ValueError('Inventory requires alpha > 0 and 0 < beta <= 1.')
        self.setLearningRate(alpha_scale / K ** 0.5, 1.0 if abs(beta - 1.0) <= 1e-12 else beta)
        self.Q = int(K ** 0.25) + (int(K ** 0.25) ** 4 < K)

'''
    Fixed parameters for Portfolio
'''

class Parameters_Portfolio:
    def __init__(self, ):
        # Portfolio smoothing initialization used by [3].
        self.LR_SMOOTH = 0.01
        self.LR_PORTFOLIO_CSDRO = 0.003
        self.BETA_PORTFOLIO = 0.1
        self.PORTFOLIO_ROUTING_SCALE = 1.0
        # Bounded training loss in Appendix F.3
        self.B = 100.0
        self.a = self.B / 50
        self.z0_bound = 5.0
        # Batch size
        self.BATCH_SIZE = 64
        # Learning rate for SAA method
        self.LR_SAA = 0.05
        # SAA training sample size
        self.N1_SAA = 500
        self.N2_SAA = 60
        self.N3_SAA = 60
        # Evaluation sample size during training (from training set)
        self.N1_EVAL = 40
        # self.N2_EVAL = 20
        # self.N3_EVAL = 20
        # Epoches
        self.EPOCHS_SAA = 500
        self.EPOCHS_SCSC = 500
        # Gradient Clip for SCSC
        self.grad_clip = 4
        # Use Adam optimizer to solve SCSC
        self.use_adam = True
        # Stable loss
        self.scalar = 50

    # Epochs
    def setEpochs(self, EPOCHS_SAA, EPOCHS_SCSC):
        self.EPOCHS_SAA = EPOCHS_SAA
        self.EPOCHS_SCSC = EPOCHS_SCSC

    # Learning Rates for SCSC
    def setLearningRate(self, LR_alpha, LR_beta):
        self.LR_alpha = LR_alpha
        self.LR_beta = LR_beta

    # Structure of 2NN and SRF
    def setModelStrcuture(self, hidden_layer_dim, SRF_depth, Tree_number, Use_Linear_Leaf, Use_Diverse_SRT):
        self.hidden_layer_dim = hidden_layer_dim
        self.SRF_depth = SRF_depth
        self.Tree_number = Tree_number
        self.Use_Linear_Leaf = Use_Linear_Leaf
        self.Use_Diverse_SRT = Use_Diverse_SRT

    # Set training iterations, sample size and learning rates from the CV row.
    def setCVParameters(self, row, n_batches):
        K = int(row['K'])
        model_name = row['Model']
        if model_name == 'ERM':
            self.EPOCHS_SAA = (K + n_batches - 1) // n_batches
            self.LR_SAA = 0.01
        elif model_name == 'Causal-WDRO':
            self.EPOCHS_SCSC = K
            self.LR_CWDRO = 0.01
            self.CWDRO_BATCH_SIZE = 2
            self.LR_CWDRO_INNER = 0.5
        else:
            self.EPOCHS_SCSC = (K + n_batches - 1) // n_batches
            if model_name == 'KL-DRO':
                self.LR_KLDRO = 0.01
            else:
                K = self.EPOCHS_SCSC * n_batches
                self.Q = int(row['Q'])
                # Causal-SDRO and SDRO use the same settings for each norm.
                settings = {1: (0.10392304845413265, 16), 2: (0.3464101615137755, 1)}
                alpha_scale, self.outer_batch = settings[int(row['norm'])]
                beta = 1.0 if self.outer_batch > 1 else min(1.0, (1200 / K) ** 0.5)
                self.setLearningRate(alpha_scale / K ** 0.5, beta)
                self.use_adam = True
                self.adam_decay = 0.0
                self.structure_lr_scale = 1.0
                self.grad_clip = 4.0

