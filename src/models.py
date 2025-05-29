import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import (
    global_add_pool, 
    global_mean_pool, 
    global_max_pool, 
    GlobalAttention, 
    Set2Set
)
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.models import GIN

from src.conv import GNN_node, GNN_node_Virtualnode

class GNN(torch.nn.Module):

    def __init__(self, num_class, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, 
                    JK = "attention", graph_pooling = "attention",
                    heads = 4): # Aggiunto parametro heads per GAT
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
            heads (int): number of attention heads for GAT
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling
        self.virtual_node = virtual_node
        self.node_encoder = torch.nn.Embedding(1, emb_dim)  # uniform input node embedding
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        ### GNN to generate node embeddings
        if virtual_node:
            if gnn_type in ['gin', 'gcn', 'sage', 'gat']:
                self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, 
                                                   residual=residual, gnn_type=gnn_type, heads=heads)
            else:
                raise ValueError(f'Undefined GNN type: {gnn_type}. Choose from: gin, gcn, sage, gat')
        else:
            if gnn_type in ['gin', 'gcn', 'sage', 'gat']:
                self.gnn_node = GNN_node(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, 
                                       residual=residual, gnn_type=gnn_type, heads=heads)
            else:
                raise ValueError(f'Undefined GNN type: {gnn_type}. Choose from: gin, gcn, sage, gat')


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

        # Add JK attention layer if needed
        self.JK = JK
        if self.JK == "attention":
            self.jk_attention = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.BatchNorm1d(emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, 1)
            )

    def forward(self, batched_data, return_embeddings=False):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        
        # Encode nodes and collect layer representations
        h = self.node_encoder(x)
        h_list = [h]
        
        # Get node embeddings for each layer
        for layer in range(self.num_layer):
            h = self.gnn_node(batched_data)
            h_list.append(h)
            
        # Apply JK connection
        if self.JK == "attention":
            # Stack all layer representations [num_layers, num_nodes, emb_dim]
            stack_h = torch.stack(h_list, dim=0)
            
            # Calculate attention scores
            att_scores = self.jk_attention(stack_h.view(-1, self.emb_dim))
            att_scores = att_scores.view(len(h_list), -1, 1)
            att_scores = F.softmax(att_scores, dim=0)
            
            # Weighted combination of layer representations
            node_representation = (stack_h * att_scores).sum(dim=0)
        else:
            # ...existing JK options (last, max, sum, concat)...
            node_representation = h_list[-1]
        
        # Graph-level readout
        graph_representation = self.pool(node_representation, batch)
        
        # Predict
        output = self.graph_pred_linear(graph_representation)
        
        if return_embeddings:
            return output, graph_representation
        return output