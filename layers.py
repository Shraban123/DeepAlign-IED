from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *


# # Feature encoder: MLP for input features
# class FeatureEncoder(nn.Module):
#     def __init__(self, input_dim, embed_dim):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, embed_dim)
#         )

#     def forward(self, x):
#         return self.mlp(x)

# # Label encoder: MLP for numeric labels
# class LabelEncoder(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(1, 32),  # 1 input dimension for scalar labels like 59, 87
#             nn.ReLU(),
#             nn.Linear(32, embed_dim)
#         )

#     def forward(self, y):
#         y = y.view(-1, 1).float()  # ensure proper shape
#         return self.mlp(y)

# # Full model: computes embeddings and distance
# class LabelAwareModel(nn.Module):
#     def __init__(self, input_dim, embed_dim):
#         super().__init__()
#         self.feature_encoder = FeatureEncoder(input_dim, embed_dim)
#         self.label_encoder = LabelEncoder(embed_dim)

#     def forward(self, x, y):
#         x_embed = self.feature_encoder(x)
#         y_embed = self.label_encoder(y)
#         return x_embed, y_embed

# # Distance-based loss (e.g., squared L2 loss or cosine)
# def distance_loss(x_embed, y_embed, mode='l2'):
#     if mode == 'l2':
#         return F.mse_loss(x_embed, y_embed)
#     elif mode == 'cosine':
#         cosine_sim = F.cosine_similarity(x_embed, y_embed, dim=-1)
#         return 1 - cosine_sim.mean()  # 1 - cosine similarity
#     else:
#         raise ValueError("Invalid mode. Use 'l2' or 'cosine'.")

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual=False):
        super(GATLayer, self).__init__()
        # equation (1) reference: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.use_residual = use_residual
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, nf, layer_id):
        h = nf.layers[layer_id].data['h']
        # equation (1)
        z = self.fc(h)
        nf.layers[layer_id].data['z'] = z
        # print("test test test")
        A = nf.layer_parent_nid(layer_id)
        # print(A)
        # print(A.shape)
        A = A.unsqueeze(-1)
        B = nf.layer_parent_nid(layer_id + 1)
        # print(B)
        # print(B.shape)
        B = B.unsqueeze(0)
        # print("DEBUG")
        # print(A)
        # print(B)

        _, indices = torch.topk((A == B).int(), 1, 0)
        # print(indices)
        # print(indices.shape)
        # indices = np.asarray(indices)
        indices = indices.cpu().data.numpy()

        nf.layers[layer_id + 1].data['z'] = z[indices]
        # print(nf.layers[layer_id+1].data['z'].shape)
        # equation (2)
        nf.apply_block(layer_id, self.edge_attention)
        # equation (3) & (4)
        nf.block_compute(layer_id,  # block_id _ The block to run the computation.
                         self.message_func,  # Message function on the edges.
                         self.reduce_func)  # Reduce function on the node.

        nf.layers[layer_id].data.pop('z')
        nf.layers[layer_id + 1].data.pop('z')

        if self.use_residual:
            return z[indices] + nf.layers[layer_id + 1].data['h']  # residual connection
        return nf.layers[layer_id + 1].data['h']
    
    # def forward(self, nf, layer_id):
    #     h = nf.layers[layer_id].data['h']
        
    #     # Check if current layer is empty
    #     if h.size(0) == 0:
    #         # Use features from previous layer if available
    #         if layer_id > 0 and 'h' in nf.layers[layer_id - 1].data:
    #             prev_h = nf.layers[layer_id - 1].data['h']
    #             if prev_h.size(0) > 0:
    #                 # Transform previous layer features to match expected output dim
    #                 z_fallback = self.fc(prev_h)
    #                 if self.use_residual:
    #                     return z_fallback + prev_h[:, :z_fallback.size(1)]
    #                 return z_fallback
    #         # If no previous layer, return random features
    #         return torch.randn(1, self.fc.out_features, device=h.device, dtype=h.dtype)
        
    #     # equation (1)
    #     z = self.fc(h)
    #     nf.layers[layer_id].data['z'] = z
        
    #     A = nf.layer_parent_nid(layer_id)
    #     B = nf.layer_parent_nid(layer_id + 1)
        
    #     # Check for empty next layer
    #     if A.size(0) == 0 or B.size(0) == 0:
    #         # Return current layer features instead of empty
    #         nf.layers[layer_id].data.pop('z', None)
    #         if self.use_residual:
    #             return z + h[:, :z.size(1)]
    #         return z
        
    #     # Continue with normal processing...
    #     A = A.unsqueeze(-1)
    #     B = B.unsqueeze(0)
        
    #     _, indices = torch.topk((A == B).int(), 1, 0)
    #     indices = indices.cpu().data.numpy()
    #     nf.layers[layer_id + 1].data['z'] = z[indices]
        
    #     # print(nf.layers[layer_id+1].data['z'].shape)
    #     # equation (2)
    #     nf.apply_block(layer_id, self.edge_attention)
    #     # equation (3) & (4)
    #     nf.block_compute(layer_id,  # block_id _ The block to run the computation.
    #                      self.message_func,  # Message function on the edges.
    #                      self.reduce_func)  # Reduce function on the node.

    #     nf.layers[layer_id].data.pop('z')
    #     nf.layers[layer_id + 1].data.pop('z')

    #     if self.use_residual:
    #         return z[indices] + nf.layers[layer_id + 1].data['h']  # residual connection
    #     return nf.layers[layer_id + 1].data['h']



class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', use_residual=False):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, use_residual))
        self.merge = merge

    def forward(self, nf, layer_id):
        head_outs = [attn_head(nf, layer_id) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads, 'cat', use_residual)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, 'cat', use_residual)

        self.layer1_1 = self.layer1 #MultiHeadGATLayer(in_dim, hidden_dim, num_heads, 'cat', use_residual)
        self.layer2_1 = self.layer2 #MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, 'cat', use_residual)

        self.layer1_2 = self.layer1 #MultiHeadGATLayer(in_dim, hidden_dim, num_heads, 'cat', use_residual)
        self.layer2_2 = self.layer2 #MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, 'cat', use_residual)

        self.layer1_3 = self.layer1 #MultiHeadGATLayer(in_dim, hidden_dim, num_heads, 'cat', use_residual)
        self.layer2_3 = self.layer2 #MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, 'cat', use_residual)
        
        # Attention
        dropout, weights_dropout = 0.2, True
        self.external_attn = MultiheadAttention(out_dim, num_heads, dropout, weights_dropout)
        self.external_attn2 = MultiheadAttention(out_dim, num_heads, dropout, weights_dropout)
        self.external_attn3 = MultiheadAttention(out_dim, num_heads, dropout, weights_dropout)

        self.linear = nn.Linear(4*out_dim, out_dim) # Setting 1
        #self.linear = nn.Linear(128, out_dim) # Event2018
        # self.linear = nn.Linear(2*out_dim, out_dim) # Setting 2

    def forward(self, nf, nf1, nf2, nf3, corrupt=False):
        #print("DEBUG: NF", nf.layers[0].data['features'].shape)
        features = nf.layers[0].data['features']
        if corrupt:
            nf.layers[0].data['h'] = features[torch.randperm(features.size()[0])]
        else:
            nf.layers[0].data['h'] = features
        
        h_f = self.layer1(nf, 0)
        h_f = F.elu(h_f)
        # print(h.shape)
        nf.layers[1].data['h'] = h_f
        h_f = self.layer2(nf, 1)
        
        # print("-------------------------DEBUG---------------------- h_f",h_f.shape)
        
        #XXXX-1
        #print("DEBUG: NF1", nf1.layers[0].data['features'].shape)
        features_1 = nf1.layers[0].data['features']
        if corrupt:
            nf1.layers[0].data['h'] = features_1[torch.randperm(features.size()[0])]
        else:
            nf1.layers[0].data['h'] = features_1
        h_1 = self.layer1_1(nf1, 0)
        h_1 = F.elu(h_1)
        # print(h.shape)
        nf1.layers[1].data['h'] = h_1
        h_1 = self.layer2_1(nf1, 1)

        # print("-------------------------DEBUG---------------------- h_1",h_1.shape)

        #XXXX-1

        #print("DEBUG: NF2", nf2.layers[0].data['features'].shape)
        features_2 = nf2.layers[0].data['features']
        if corrupt:
            nf2.layers[0].data['h'] = features_2[torch.randperm(features.size()[0])]
        else:
            nf2.layers[0].data['h'] = features_2
        h_2 = self.layer1_2(nf2, 0)
        h_2 = F.elu(h_2)
        # print(h.shape)
        nf2.layers[1].data['h'] = h_2
        h_2 = self.layer2_2(nf2, 1) 

        # print("-------------------------DEBUG---------------------- h_2",h_2.shape)

        #XXXX-1
        #print("DEBUG: NF3", nf3.layers[0].data['features'].shape)
        features_3 = nf3.layers[0].data['features']
        if corrupt:
            nf3.layers[0].data['h'] = features_3[torch.randperm(features.size()[0])]
        else:
            nf3.layers[0].data['h'] = features_3
        h_3 = self.layer1_3(nf3, 0)
        h_3 = F.elu(h_3)
        # print(h.shape)
        nf3.layers[1].data['h'] = h_3
        h_3 = self.layer2_3(nf3, 1)

        # print("-------------------------DEBUG---------------------- h_f",h_f.shape)

        # h = torch.cat((h_1, h_2, h_3),dim=1)
        # h = self.linear(h)
        # Attention module
        h_f = torch.reshape(h_f, shape=(h_f.shape[0], 1, h_f.shape[1])) # Added for event2018
        h_1 = torch.reshape(h_1, shape=(h_1.shape[0], 1, h_1.shape[1]))
        h_2 = torch.reshape(h_2, shape=(h_2.shape[0], 1, h_2.shape[1]))
        h_3 = torch.reshape(h_3, shape=(h_3.shape[0], 1, h_3.shape[1]))

        # print("DEBUG: h_1 size", h_1.size())
        # print("DEBUG: h_2 shape", h_2.size())
        # print("DEBUG: h_3 shape", h_3.size())

        a1, external_attn = self.external_attn(query=h_f, key=h_1, value=h_1, key_padding_mask=None,
                                          attn_mask=None, need_weights=False)
        a2, external_attn = self.external_attn(query=h_f, key=h_2, value=h_2, key_padding_mask=None,
                                          attn_mask=None, need_weights=False)
        a3, external_attn = self.external_attn(query=h_f, key=h_3, value=h_3, key_padding_mask=None,
                                          attn_mask=None, need_weights=False) 
        # print("DEBUG: shape a1", a1.shape)
        h_f = torch.reshape(h_f, shape=(h_f.shape[0], h_f.shape[-1])) # Added for event2018
        a1 = torch.reshape(a1, shape=(a1.shape[0], a1.shape[2]))
        a2 = torch.reshape(a2, shape=(a2.shape[0], a2.shape[2]))
        a3 = torch.reshape(a2, shape=(a3.shape[0], a3.shape[2]))
        h = torch.cat((h_f, a1, a2, a3),dim=1) # Setting 1
        #h = torch.cat((a1, a2),dim=1) # Setting 2
        # print("DEBUG: shape H", h.shape)
        #h = torch.reshape(h, shape=(h.shape[0], h.shape[2]))
        # print("DEBUG: After reshape shape H", h.shape)
        h = self.linear(h)
        
        return h
        

        #return h_f
        
        #return h_f # Setting 3
# Applies an average on seq, of shape (nodes, features)
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 0)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 0)
        # print("testing, shape of logits: ", logits.size())
        return logits


class DGI_old(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False):
        super(DGI, self).__init__()
        self.gat = GAT(in_dim, hidden_dim, out_dim, num_heads, use_residual)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(out_dim)

    def forward(self, nf):
        h_1 = self.gat(nf, False)
        c = self.read(h_1)
        c = self.sigm(c)
        h_2 = self.gat(nf, True)
        ret = self.disc(c, h_1, h_2)
        return h_1, ret

    # Detach the return variables
    def embed(self, nf):
        h_1 = self.gat(nf, False)
        return h_1.detach()

class DGI(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False):
        super(DGI, self).__init__()
        self.gat = GAT(in_dim, hidden_dim, out_dim, num_heads, use_residual)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(out_dim)


    def forward(self, nf, nf2, nf3):
        h_1 = self.gat(nf, False)
        c = self.read(h_1)
        c = self.sigm(c)
        h_2 = self.gat(nf, True)
        ret = self.disc(c, h_1, h_2)

        h_2 = self.gat(nf2, False)
        
        return h_1, ret

    # Detach the return variables
    def embed(self, nf):
        h_1 = self.gat(nf, False)
        return h_1.detach()

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            '''
            #XXXX-1
            if len(label_indices) < 2:
                continue
            '''
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            
            
            #XXXX-1
            if len(anchor_positives) == 0: # error handling - error occurs when only one positive sample is there
                #print("WARNING: Isolated label found")
                anchor_positives = [(label_indices[0], label_indices[0])]
            
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                              negative_selection_fn=hardest_negative,
                                                                                              cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                             negative_selection_fn=random_hard_negative,
                                                                                             cpu=cpu)
