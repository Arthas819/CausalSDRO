"""
    This file includes our two decision rules.
"""

import torch
import torch.nn as nn


'''
    Two-Layer Neural Network (2NN) Decision Rule, as a baseline
'''

class TNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = nn.Linear(self.input_dim, hidden_dim)
        # self.activation = nn.ReLU()
        self.activation = nn.Softplus(beta=5)
        self.layer2 = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        if x.dim() == 1:
            # add a new dimension
            x = x.unsqueeze(0)
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x.squeeze(-1) if self.output_dim == 1 else x


'''
    The proposed SRF decision rule.
'''

class SRF(nn.Module):

    # For leaf nodes, they can output a single decision, or a linear transformation for input x
    # For each tree, we can decline the number of features or samples, to increase the diversity.

    def __init__(self, input_dim, depth, output_dim, tree_number, params):
        super(SRF, self).__init__()

        # Basic parameters
        self.input_dim = input_dim
        self.depth = depth
        self.output_dim = output_dim
        self.tree_number = tree_number
        self.params = params
        self.Use_Linear_Leaf = self.params.Use_Linear_Leaf
        self.Use_Diverse_SRT = self.params.Use_Diverse_SRT

        # Number of nodes
        self.num_leaves = 2 ** self.depth
        self.num_internal_nodes = (2 ** self.depth) - 1

        # Set internal (non-leaf) nodes
        # w: [tree, nodes, input_dim]
        self.internal_nodes_w = nn.Parameter(
            torch.randn(self.tree_number, self.num_internal_nodes, self.input_dim) * 0.1
        )
        # b: [tree, nodes]
        self.internal_nodes_b = nn.Parameter(
            torch.randn(self.tree_number, self.num_internal_nodes) * 0.1
        )

        # Sigmoid function for each internal node
        self.sigmoid = nn.Sigmoid()
        # Rule function for each leaf node
        self.ReLu = nn.Softplus(beta=5)

        # Leaf
        self.leaf_nodes_pi = None
        self.leaf_nodes_w = None
        self.leaf_nodes_b = None

        # Set leaf nodes
        if self.Use_Linear_Leaf == False:
            # Use constant leaf pi. Directly output a decision at each leaf
            # pi: [tree, nodes, output_dim]
            self.leaf_nodes_pi = nn.Parameter(
                torch.rand(self.tree_number, self.num_leaves, self.output_dim) * 0.1
            )
        else:
            # Use linear leaf. Output a linear transformation on x at each leaf
            # leaf_w: [tree, leaves, output_dim, input_dim]
            self.leaf_nodes_w = nn.Parameter(
                torch.randn(self.tree_number, self.num_leaves, self.output_dim, self.input_dim) * 0.1
            )
            # leaf_b: [tree, leaves, output_dim]
            self.leaf_nodes_b = nn.Parameter(
                torch.randn(self.tree_number, self.num_leaves, self.output_dim) * 0.1
            )

        # Construct a mask (all 1) with size [tree_number, input_dim]
        mask = torch.ones(self.tree_number, self.input_dim)

        # To add the diversity of SRTs in SRF, input different feature subsets for trees that index > 3
        if self.Use_Diverse_SRT == True:
            # For each selected tree, we only use 80% features
            num_drop = int(self.input_dim * 0.2)

            # For the first 3 trees, we remain all features
            if num_drop > 0 and self.tree_number > 3:
                # For other trees
                for t in range(3, self.tree_number):
                    # Select 20% dropped features for each tree randomly
                    drop_indices = torch.randperm(self.input_dim)[:num_drop]
                    # Use mask to ignore these dropped features
                    mask[t, drop_indices] = 0.0

        # Set to register_buffer, ensure that it can be sorted and moved to GPU and is not trained-able
        self.register_buffer('feature_mask', mask)

    def forward(self, x):

        # Inpupt shape: [batch_size, input_dim]
        batch_size = x.shape[0]

        # Mask internal weights, feature_mask: [tree, input_dim] -> [tree, 1, input_dim]
        # effective_internal_w: [tree, nodes, input_dim]
        effective_internal_w = self.internal_nodes_w * self.feature_mask.unsqueeze(1)

        # At each node, calcualte S(w * x + b)
        # x [batch_size, input_dim] * w [tree, nodes, input_dim] -> [batch_size, tree, nodes]
        internal_affine_trans = torch.einsum('bi, tni -> btn', x, effective_internal_w)
        # b [tree, nodes] -(add a dimension at index 0)-> [batch_size, tree, nodes]
        internal_linear_trans = internal_affine_trans + self.internal_nodes_b.unsqueeze(0)
        # Non-linear activation, [batch_size, tree, nodes]
        nodes_probs = self.sigmoid(internal_linear_trans)

        # For all leaf nodes, get their prob path.
        # We construct all paths from the root nodes in each tree.
        path_probs = torch.ones(batch_size, self.tree_number, 1, device=x.device)
        node_index = 0

        # At each layer (start by layer 0, i.e., the root node) :
        for d in range(self.depth):
            nodes_in_layer = 2 ** d

            # Get the probs of all nodes in this layer, [batch_size, tree, nodes in this layer]
            layer_probs = nodes_probs[:, :, node_index: node_index + nodes_in_layer]

            # Split, to [batch_size, tree, nodes in this layer]
            probs_left = path_probs * layer_probs
            probs_right = path_probs * (1.0 - layer_probs)

            # Stack: 2* [batch_size, tree, nodes in this layer] -> [batch, tree, nodes_in_layer, 2]
            path_probs = torch.stack((probs_left, probs_right), dim=3)
            # View: [batch, tree, nodes_in_layer, 2] -> [batch, tree, nodes_in_layer * 2]
            path_probs = path_probs.view(batch_size, self.tree_number, -1)

            # Set node index
            node_index += nodes_in_layer

        # Now, path_probs : [batch_size, tree, leaf_nodes]

        # Then, calculate the outputs of each leaf
        if self.Use_Linear_Leaf == False:
            # If we use constant leaves, just calculate the expectation
            # [batch_size, tree, leaf_nodes] * [tree, leaf_nodes, output_dim] -> [batch_size, tree, output_dim]
            tree_outputs = torch.einsum('btl,tlo->bto', path_probs, self.leaf_nodes_pi)
        else:
            # For linear leaves, first, mask
            effective_leaf_w = self.leaf_nodes_w * self.feature_mask.unsqueeze(1).unsqueeze(1)
            # Then, linear transformation
            # [batch_size, input_dim] * [tree, leaves, output_dim, input_dim] -> [batch_size, tree, leaves, output_dim]
            leaf_affine = torch.einsum('bi, tloi -> btlo', x, effective_leaf_w)
            # Add bias, leaf_nodes_b -> [?, tree, leaves, output_dim]
            leaf_linear = leaf_affine + self.leaf_nodes_b.unsqueeze(0)
            # leaf_values: [batch_size, tree, leaves, output_dim]
            leaf_values = self.ReLu(leaf_linear)

            # path_probs -> [batch_size, tree, leaves, 1]
            # tree_outputs : [batch_size, tree, output_dim]
            tree_outputs = torch.sum(path_probs.unsqueeze(-1) * leaf_values, dim=2)

        # Final decision: average of all trees
        # [batch_size, tree, output_dim] -> [batch_size, output_dim]
        final_output = torch.mean(tree_outputs, dim=1)

        # Output all results, if output_dim = 1, then final_output -> [batch_size]
        return final_output.squeeze(-1) if self.output_dim == 1 else final_output