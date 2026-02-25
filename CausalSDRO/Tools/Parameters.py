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
        # SAA training sample size
        self.N1_SAA = 200
        self.N2_SAA = 100
        self.N3_SAA = 100
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
        # Batch size
        self.BATCH_SIZE = 60
        # Learning rate for SAA method
        self.LR_SAA = 0.05
        # SAA training sample size
        self.N1_SAA = 60
        self.N2_SAA = 60
        self.N3_SAA = 60
        # Evaluation sample size during training (from training set)
        self.N1_EVAL = 20
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

'''
    Fixed parameters for Portfolio
'''

class Parameters_Portfolio:
    def __init__(self, ):
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