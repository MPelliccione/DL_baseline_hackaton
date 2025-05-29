import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import (
    global_add_pool, 
    global_mean_pool, 
    global_max_pool, 
    GlobalAttention, 
    Set2Set,
    SAGEConv,
    GATConv,
    GCNConv, 
    GINConv
)
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.models import GIN

from src.conv import GNN_node, GNN_node_Virtualnode

class GNN(torch.nn.Module):

    def __init__(self, num_class, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "max", graph_pooling = "attention",
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

    def forward(self, batched_data, return_embeddings=False):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        
        if self.virtual_node:
            # Virtual node embeddings
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.node_encoder(x)]
        
        for layer in range(self.num_layer):
            if self.virtual_node:
                # Add virtual node to graph
                h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
                
            h = self.convs[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
                
            if self.virtual_node:
                # Update virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                virtualnode_embedding = F.dropout(
                    self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                    self.drop_ratio, training=self.training)
                
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        # Graph-level readout
        graph_representation = self.pool(node_representation, batch)
        output = self.graph_pred_linear(graph_representation)

        if return_embeddings:
            return output, graph_representation
        return output