import numpy as np
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.nn import ResGatedGraphConv, TransformerConv
import transformer
import torch.nn as nn
from utils import load_ESM_representations

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        # self.MLP = MLP(4)
        self.fc3 = nn.Linear(32, 16)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Transformers_model(torch.nn.Module):
    def __init__(self, args):
        self.args=args
        super(Transformers_model, self).__init__()
        self.transformers = torch.nn.ModuleList([transformer.Transformer(args) for _ in range(5)])
        self.fc1 = torch.nn.Linear(1024, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 16)
        self.fc4 = torch.nn.Linear(16, 1)
        self.dropout = torch.nn.Dropout(0.5)


    def build_edge_matrix(self,edges):
        edges_matrix=np.zeros((self.args.src_len, self.args.src_len))
        for node_pair in torch.transpose(edges,0,1):
            edges_matrix[node_pair[0], node_pair[1]] = 1

        edges_matrix=torch.from_numpy(edges_matrix).eq(0)
        edges_matrix = edges_matrix.unsqueeze(0).unsqueeze(0).repeat(1,self.args.n_heads,1,1)

        return edges_matrix

    def encoder(self,nodes,edges_list):
        processed_nodes_list=[]
        nodes=torch.cat([nodes,torch.zeros((self.args.src_len-len(nodes),4))],dim=0)

        for i, edges in enumerate(edges_list):
            edges_matrix = self.build_edge_matrix(edges)
            processed_nodes_list.append(self.transformers[i](nodes.unsqueeze(0),edges_matrix))
        return torch.cat(processed_nodes_list,dim=1)

    def decode(self, z, edge_index):
        #edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)

        x = torch.cat((z[edge_index[0]], z[edge_index[1]]), dim=1)
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.fc2(x).relu()
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.fc3(x).relu()
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return torch.squeeze(x)

class SLMGAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_graph, esm_dim, mlp_dim, data_dir, gene_mapping, celline_feats, args):
        super(SLMGAE, self).__init__()
        # number of GCN
        self.num_graph = num_graph
        self.esm_dim = esm_dim
        self.mlp_dim = mlp_dim
        self.data_dir = data_dir
        self.gene_mapping = gene_mapping
        self.celline_feats = celline_feats
        self.args = args
        self.GCNs = torch.nn.ModuleList()
        for _ in range(num_graph):
            self.GCNs.append(GCN(in_channels, out_channels))
        # Initialize weights for each view
        self.weight_views = torch.ones(num_graph - 1)
        self.weight_views = torch.nn.Parameter(torch.nn.functional.softmax(self.weight_views, dim=0))
        
        # transforer
        in_channels_transformer = args.d_model + esm_dim + mlp_dim
        self.transformer = transformer.Transformer(args)
        self.transformer_fc1 = torch.nn.Linear(2*in_channels_transformer, in_channels_transformer)
        self.transformer_fc2 = torch.nn.Linear(in_channels_transformer, int(in_channels_transformer / 2))
        self.transformer_fc3 = torch.nn.Linear(int(in_channels_transformer / 2), 2 * out_channels)

        in_channels_classifer = out_channels + esm_dim + mlp_dim if args.pooling != "transformer" else out_channels
        self.fc1 = torch.nn.Linear(2*in_channels_classifer, in_channels_classifer)
        self.fc2 = torch.nn.Linear(in_channels_classifer, int(in_channels_classifer/2))
        self.fc3 = torch.nn.Linear(int(in_channels_classifer/2), int(in_channels_classifer/4))
        self.fc4 = torch.nn.Linear(int(in_channels_classifer/4), 1)
        self.dropout = torch.nn.Dropout(0.5)
        self.cell_line_spec_mlp = MLP(4)

        if self.esm_dim != 0:
            self.esm_representation = load_ESM_representations(self.data_dir, self.gene_mapping)

    def build_mask_matrix(self,input_len):
        mask_matrix=torch.ones(self.args.src_len, self.args.src_len,dtype=torch.bool)
        mask_matrix[0:input_len,0:input_len]=0
        mask_matrix = mask_matrix.unsqueeze(0).unsqueeze(0).repeat(1, self.args.n_heads, 1, 1)
        return mask_matrix

    def modified_transformer(self,transformer_input):
        length=len(transformer_input)
        mask_matrix = self.build_mask_matrix(length)
        transformer_input = torch.cat([transformer_input, torch.zeros((self.args.src_len - length, self.num_graph * self.args.out_channels))], dim=0)
        #transformer_input=self.fc_pre(transformer_input)

        transformer_output=self.transformer(transformer_input,mask_matrix)
        return transformer_output[0:length]
    
    def decode_transformer(self, z, edge_index, node_index_map):
        new_edge_index = torch.zeros((2, len(edge_index[0])), dtype=torch.int64)
        for i in range(len(edge_index[0])):
            new_edge_index[0][i] = node_index_map[edge_index[0][i].item()]
            new_edge_index[1][i] = node_index_map[edge_index[1][i].item()]
        z1 = z[new_edge_index[0]]
        z2 = z[new_edge_index[1]]
        if self.mlp_dim != 0:
            MLP_output = self.cell_line_spec_mlp(self.celline_feats)
            z1 = torch.cat((z1, MLP_output[new_edge_index[0]]), dim = 1)
            z2 = torch.cat((z2, MLP_output[new_edge_index[1]]), dim = 1)
        if self.esm_dim != 0:
            z1 = torch.cat([z1, self.esm_representation[new_edge_index[0]]], dim=1)
            z2 = torch.cat([z2, self.esm_representation[new_edge_index[1]]], dim=1)
        x = torch.cat((z1, z2), dim=1)
        x = self.transformer_fc1(x).relu()
        x = self.dropout(x)
        x = self.transformer_fc2(x).relu()
        x = self.dropout(x)
        x = self.transformer_fc3(x).relu()
        x = self.dropout(x)
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x).relu()
        x = self.dropout(x)
        x = self.fc3(x).relu()
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.squeeze(x)


    def decode(self, z, edge_index):
        if self.mlp_dim != 0:
            MLP_output = self.cell_line_spec_mlp(self.celline_feats)
            z = torch.cat((z, MLP_output), dim = 1)
        if self.esm_dim != 0:
            z = torch.cat([z, self.esm_representation], dim=1)
        x = torch.cat((z[edge_index[0]],z[edge_index[1]]),dim=1)
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x).relu()
        x = self.dropout(x)
        x = self.fc3(x).relu()
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.squeeze(x)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 2*out_channels)
        self.conv2 = GCNConv(2*out_channels, out_channels)
        self.fc1 = torch.nn.Linear(2*(out_channels), out_channels)
        self.fc2 = torch.nn.Linear(out_channels, int(out_channels/2))
        self.fc3 = torch.nn.Linear(int(out_channels/2), int(out_channels/4))
        self.fc4 = torch.nn.Linear(int(out_channels/4), 1)
        self.dropout = torch.nn.Dropout(0.5)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        x = torch.cat((z[edge_index[0]],z[edge_index[1]]),dim=1)
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x).relu()
        x = self.dropout(x)
        x = self.fc3(x).relu()
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.squeeze(x)

class GCN_conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_graph):
        super(GCN_conv, self).__init__()
        self.num_graph = num_graph
        self.conv1 = GCNConv(in_channels, 2*out_channels)
        self.conv2 = GCNConv(2*out_channels, out_channels)
        
        # in_channels is 1, number of filters is 32
        self.filter_num = 128
        self.conv_net = torch.nn.Conv2d(1,self.filter_num, (1,num_graph))
        
        self.fc1 = torch.nn.Linear(2*out_channels, out_channels)
        self.fc2 = torch.nn.Linear(out_channels, int(out_channels/2))
        self.fc3 = torch.nn.Linear(int(out_channels/2), int(out_channels/4))
        self.fc4 = torch.nn.Linear(int(out_channels/4), 1)
        self.dropout = torch.nn.Dropout(0.5)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        # adding one more dimension as channel, set to 1
        z = z.unsqueeze(1)
        z = self.conv_net(z).squeeze(3)
        z = F.max_pool2d(z, (self.filter_num,1)).squeeze(1)
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x = torch.cat((z[edge_index[0]],z[edge_index[1]]),dim=1)
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x).relu()
        x = self.dropout(x)
        x = self.fc3(x).relu()
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.squeeze(x)
    
    
class GAT_Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_graph):
        super(GAT_Net, self).__init__()
        self.conv1 = GATv2Conv(in_channels, 2*out_channels, heads=1, dropout=0.3)
        self.conv2 = GATv2Conv(2*out_channels, out_channels, dropout=0.3)
        #self.conv1 = TransformerConv(in_channels, 2*out_channels, heads=1, dropout=0.3)
        #self.conv2 = TransformerConv(2*out_channels, out_channels, heads=1, dropout=0.3)
        self.fc1 = torch.nn.Linear(num_graph*2*out_channels, num_graph*out_channels)
        self.fc2 = torch.nn.Linear(num_graph*out_channels, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x = torch.cat((z[edge_index[0]],z[edge_index[1]]),dim=1)
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x).relu()
        x = self.dropout(x)
        x = self.fc3(x).relu()
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.squeeze(x)


class GCN_multi(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_graph):
        super(GCN_multi, self).__init__()

        self.conv1_list = []
        self.conv2_list = []
        for _ in range(num_graph):
            self.conv1_list.append( GCNConv(in_channels, 2*out_channels) )
            self.conv2_list.append( GCNConv(2*out_channels, out_channels) )

        self.fc1 = torch.nn.Linear(2*out_channels, out_channels)
        self.fc2 = torch.nn.Linear(out_channels, int(out_channels/2))
        self.fc3 = torch.nn.Linear(int(out_channels/2), int(out_channels/4))
        self.fc4 = torch.nn.Linear(int(out_channels/4), 1)
        self.dropout = torch.nn.Dropout(0.5)
        self.num_graph = num_graph

    def encode(self, x, edge_index, graph_idx):
        x = self.conv1_list[graph_idx].cuda()(x, edge_index).relu()
        x = self.conv2_list[graph_idx].cuda()(x, edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x = torch.cat((z[edge_index[0]],z[edge_index[1]]),dim=1)
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x).relu()
        x = self.dropout(x)
        x = self.fc3(x).relu()
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.squeeze(x)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, reverse=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.reverse = reverse
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):
        if self.reverse == True:
            # loss
            score = -val_loss
        else:
            # AUC/AUPR
            score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss