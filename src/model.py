import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import TransformerConv

from PPI import *


class GNNLayer(nn.Module):
    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(in_channels=num_hidden, out_channels=int(num_hidden/num_heads), heads=num_heads, dropout=dropout, edge_dim=num_hidden, root_weight=False)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))

        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        h_E = self.edge_update(h_V, edge_index, h_E)

        h_V = self.context(h_V, batch_id)

        return h_V, h_E


class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3*num_hidden, num_hidden, bias=True) 
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True) 
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E


class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                            )

    def forward(self, h_V, batch_id):
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V


class Graph_encoder(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, seq_in=False, num_layers=4, drop_rate=0.2):
        super(Graph_encoder, self).__init__()

        self.seq_in = seq_in
        if self.seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim += 20
        
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)
        
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.layers = nn.ModuleList(
                        GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4)
                        for _ in range(num_layers))


    def forward(self, h_V, edge_index, h_E, seq, batch_id):
        if self.seq_in and seq is not None:
            seq = self.W_s(seq)
            h_V = torch.cat([h_V, seq], dim=-1)

        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)
        
        return h_V


class Attention(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def forward(self, input):  				
        x = torch.tanh(self.fc1(input))  	
        x = self.fc2(x)  					
        x = x.softmax(1)
        attention = x.transpose(1, 2)  		
        return attention


class PMHCBinder(nn.Module): 
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, num_layers, dropout, augment_eps, device):
        super(PMHCBinder, self).__init__()
        self.augment_eps = augment_eps
        self.device = device
        self.hidden_dim = hidden_dim
        
        self.Graph_encoder = Graph_encoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim, hidden_dim=hidden_dim, seq_in=False, num_layers=num_layers, drop_rate=dropout)

        self.attention = Attention(hidden_dim, dense_dim=16, n_heads=4)

        self.add_module("FC_{}1".format('pMHC'), nn.Linear(hidden_dim, hidden_dim, bias=True))
        self.add_module("FC_{}2".format('pMHC'), nn.Linear(hidden_dim, 1, bias=True))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, h_V, edge_index, seq, batch_id, isaugment, h_E, entity):
        if isaugment and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            h_V = h_V + self.augment_eps * torch.randn_like(h_V)

        h_V_geo, h_E_geo = get_geo_feat(X, edge_index) 
        h_V = torch.cat([h_V, h_V_geo], dim=-1)
        h_E = torch.cat([h_E, h_E_geo], dim=-1)
        h_V = self.Graph_encoder(h_V, edge_index, h_E, seq, batch_id) 

        batchx = torch_geometric.utils.unbatch(h_V, batch_id)

        x_entities = [
            batchx[i].unsqueeze(0)[:, torch.tensor(entity[i]).nonzero(as_tuple=True)[0], :]
            for i in range(len(batchx))
        ]
        weighted_entities = [self.attention(x) @ x for x in x_entities]
        feature_embedding = torch.cat(weighted_entities, dim=0).sum(dim=1) 
        
        emb = F.elu(self._modules["FC_{}1".format('pMHC')](feature_embedding))
        output = self._modules["FC_{}2".format('pMHC')](emb)
        return output