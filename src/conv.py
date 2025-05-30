import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree

import math

from src.utils import DropPath
### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.LayerNorm(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim):
        """
        GraphSAGE convolution layer
        Args:
            emb_dim (int): dimensionality of embeddings
        """
        super(GraphSAGEConv, self).__init__(aggr='mean')  # Using mean aggregation

        self.linear_neigh = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_self = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        # Edge features transformation
        edge_embedding = self.edge_encoder(edge_attr)
        
        # Compute message passing
        neigh_msg = self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        
        # Transform both self features and aggregated neighbor features
        out_neigh = self.linear_neigh(neigh_msg)
        out_self = self.linear_self(x)
        
        # Combine and normalize
        out = out_neigh + out_self
        out = F.normalize(out, p=2, dim=-1)
        
        return out

    def message(self, x_j, edge_attr):
        # Combine node features with edge features
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=4, dropout=0.5, negative_slope=0.2):
        """
        GAT convolution layer
        Args:
            emb_dim (int): dimensionality of embeddings
            heads (int): number of attention heads
            dropout (float): dropout probability
            negative_slope (float): LeakyReLU angle of negative slope
        """
        super(GATConv, self).__init__(aggr='add', node_dim=0)  # "Add" aggregation with multi-head attention

        self.emb_dim = emb_dim
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.head_dim = emb_dim // heads
        assert self.head_dim * heads == emb_dim, "Embedding dimension must be divisible by number of heads"

        # Linear transformations for node features
        self.linear = torch.nn.Linear(emb_dim, heads * self.head_dim, bias=False)
        # Attention mechanisms
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * self.head_dim))
        # Edge feature transformation
        self.edge_encoder = torch.nn.Linear(7, emb_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize learnable parameters."""
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_normal_(self.linear.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.att, gain=gain)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        
        x = self.linear(x).view(-1, self.heads, self.head_dim)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        
        # Concatenate heads
        out = out.view(-1, self.emb_dim)
        
        return out

    def message(self, edge_index_i, x_i, x_j, edge_attr):
        # Compute attention coefficients
        alpha = torch.cat([x_i, x_j], dim=-1).view(-1, self.heads, 2 * self.head_dim)
        alpha = F.leaky_relu((alpha * self.att).sum(dim=-1), self.negative_slope)
        alpha = F.softmax(alpha, dim=0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Add edge features to the message
        edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)
        return (x_j * alpha.unsqueeze(-1) + edge_attr) / 2

    def update(self, aggr_out):
        return aggr_out

### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            elif gnn_type == 'graphsage':
                self.convs.append(GraphSAGEConv(emb_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.LayerNorm(2*emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch


        ### computing input node embedding

        h_list = [self.node_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        # Alla fine di __init__
        self.droppaths = torch.nn.ModuleList([
            DropPath(drop_prob=0.1) for _ in range(self.num_layer)
        ])

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            elif gnn_type == 'graphsage':
                self.convs.append(GraphSAGEConv(emb_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.LayerNorm(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.LayerNorm(emb_dim), torch.nn.ReLU()))


    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.node_encoder(x)]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = self.droppaths[layer](h + h_list[layer])


            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP
                virtualnode_update = self.mlp_virtualnode_list[layer](virtualnode_embedding_temp)
                virtualnode_update = self.droppaths[layer](virtualnode_update)  # usa lo stesso droppath

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(virtualnode_update, self.drop_ratio, training=self.training)
                else:
                    virtualnode_embedding = F.dropout(virtualnode_update, self.drop_ratio, training=self.training)

                

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation
