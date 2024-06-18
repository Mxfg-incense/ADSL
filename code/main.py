import numpy as np
import pandas as pd
import argparse, sys, json, random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import matplotlib.pyplot as plt

from utils import *
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from model import *

import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)

def init_argparse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--type', type=str, default="feature", help="one of these: feature, comparison, test")
    parser.add_argument('--data_dir', type=str, default="../data", help="directory of the data")
    parser.add_argument('--data_source', type=str, default="A549", help="which cell line to train and predict")
    parser.add_argument('--threshold', type=float, default=-3, help="threshold of SL determination")
    parser.add_argument('--specific_graph', type=lambda s:[item for item in s.split("%") if item != ""], default=["SL"], help="lists of cell-specific graphs to use.")
    parser.add_argument('--indep_graph', type=lambda s:[item for item in s.split("%") if item != ""], 
                    default=['PPI-genetic','PPI-physical','co-exp','co-ess'], help="lists of cell-independent graphs to use.")
    parser.add_argument('--node_feats', type=str, default="raw_omics", help="gene node features")

    parser.add_argument('--balanced', type=int, default=1, help="whether the negative and positive samples are balanced")
    parser.add_argument('--pos_weight', type=float, default=50, help="weight for positive samples in loss function")
    parser.add_argument('--CCLE', type=int, default=0, help="whether or not include CCLE features into node features")
    parser.add_argument('--CCLE_dim', type=int, default=64, help="dimension of embeddings for each type of CCLE omics data")
    parser.add_argument('--node2vec_feats', type=int, default=0, help="whether or not using node2vec embeddings")

    parser.add_argument('--model', type=str, default="GCN_pool", help="type of model")
    parser.add_argument('--pooling', type=str, default="max", help="type of pooling operations")
    parser.add_argument('--LR', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--epochs', type=int, default=200, help="number of maximum training epochs")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size")
    parser.add_argument('--out_channels', type=int, default=64, help="dimension of output channels")
    parser.add_argument('--patience', type=int, default=150, help="patience in early stopping")
    parser.add_argument('--training_percent', type=float, default=0.70, help="proportion of the SL data as training set")
    parser.add_argument('--esm_reps_flag', type=int, default=0, help="whether or not include ESM representations into node features")

    parser.add_argument('--save_results', type=int, default=1, help="whether to save test results into json")
    parser.add_argument('--split_method', type=str, default="novel_pair", help="how to split data into train, val and test")
    parser.add_argument('--predict_novel_cellline', type=int, default=0, help="whether to predict on novel cell lines")
    parser.add_argument('--novel_cellline', type=str, default="Jurkat", help="name of novel celllines")
    parser.add_argument('--MLP_celline', type=int, default=0, help="use celline feats or not")
    
    parser.add_argument('--neg_num', type=float, default=1, help='number of negative samples is several times that of the positive samples')

    args = parser.parse_args()

    return args


def train_model(model: SLMGAE, optimizer, node_embedding, SL_data_edge_index, support_views_edge_index, device, train_pos_edge_index, train_neg_edge_index):
    model.train()
    optimizer.zero_grad()

    all_edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], dim=-1)
    labels = get_link_labels(train_pos_edge_index, train_neg_edge_index, support_views_edge_index, device)
    # shuffle training edges and labels
    #lable shape: (num_views+1, num_edges), shuffle the second dimension
    num_samples = all_edge_index.shape[1]
    perm = torch.randperm(num_samples)
    all_edge_index = all_edge_index[:, perm]
    labels = labels[:, perm]

    start = 0
    loss = 0
    while start < num_samples:
        batch_loss = 0
        link_labels = labels[:, start:(start + args.batch_size)]
        this_batch_edge_index = all_edge_index[:, start:(start + args.batch_size)]
        this_batch_node_index = torch.cat((this_batch_edge_index[0], this_batch_edge_index[1])).unique()
        this_batch_node_index_map = {}  # 用来预测的边的位置进行标记
        for i, node_index in enumerate(this_batch_node_index):
            this_batch_node_index_map[node_index.item()] = i

        if args.balanced:
            pos_weight = torch.tensor(1)
        else:
            pos_weight = torch.tensor(args.pos_weight)

        views_z = []
        SL_z = model.GCNs[0].encode(node_embedding, SL_data_edge_index)
        views_z.append(SL_z) 
        link_logits_SL = model.GCNs[0].decode(SL_z, this_batch_edge_index)
        batch_loss += F.binary_cross_entropy_with_logits(link_logits_SL, link_labels[0], pos_weight=pos_weight)

        for i, edge_index in enumerate(support_views_edge_index):
            z = model.GCNs[i+1].encode(node_embedding, edge_index)
            if args.pooling == "agg" or "transformer":
                z = z * model.weight_views[i]
            views_z.append(z)
            link_logits = model.GCNs[i+1].decode(z, this_batch_edge_index)
            batch_loss += model.weight_views[i] * F.binary_cross_entropy_with_logits(link_logits, link_labels[i+1], pos_weight=pos_weight)
        
        z = torch.cat(views_z,1)
        
        # z(batch_size, num_views * numfeatures) -> z(batch_size, num_views, numfeatures) -> z(batch_size, numfeatures, num_views)
        z = z.unsqueeze(1).reshape(z.shape[0], model.num_graph, -1).transpose(1,2)
        if args.pooling == "max":
            z = F.max_pool2d(z, (1,model.num_graph)).squeeze(2)
        elif args.pooling == "mean":    
            z = F.avg_pool2d(z, (1,model.num_graph)).squeeze(2)
        elif args.pooling == "agg":
            # sum up the embeddings of different views with weights
            z = z.sum(-1)
        elif args.pooling == "transformer":
            z = z.sum(-1)
            transformer_output = model.transformer_encoder(z[this_batch_node_index])
            z1 = transformer_output[[this_batch_node_index_map[i.item()] for i in this_batch_edge_index[0]]]
            z2 = transformer_output[[this_batch_node_index_map[i.item()] for i in this_batch_edge_index[1]]]
            
        link_logits = model.decode(z, this_batch_edge_index) if args.pooling != "transformer" else model.decode_transformer(z1, z2)
            
        batch_loss += 2*F.binary_cross_entropy_with_logits(link_logits, link_labels[0], pos_weight=pos_weight)

        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        start += args.batch_size

    return float(loss)


@torch.no_grad()
def test_model(model, node_embedding, SL_data_edge_index, support_views_edge_index, device, pos_edge_index, neg_edge_index):
    model.eval()
    results = {}
    loss = 0

    all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    all_node_index = torch.cat((all_edge_index[0], all_edge_index[1])).unique()
    all_node_index_map = {}  # 用来预测的边的位置进行标记
    for i, node_index in enumerate(all_node_index):
        all_node_index_map[node_index.item()] = i

    link_labels = get_link_labels(pos_edge_index, neg_edge_index, support_views_edge_index, device)
    if args.balanced:
        pos_weight = torch.tensor(1)
    else:
        pos_weight = torch.tensor(args.pos_weight)

    views_z = []
    SL_z = model.GCNs[0].encode(node_embedding, SL_data_edge_index)
    views_z.append(SL_z) 
    link_logits_SL = model.GCNs[0].decode(SL_z, all_edge_index)
    loss += F.binary_cross_entropy_with_logits(link_logits_SL, link_labels[0], pos_weight=pos_weight)

    for i, edge_index in enumerate(support_views_edge_index):
        z = model.GCNs[i+1].encode(node_embedding, edge_index)
        if args.pooling == "agg" or "transformer":
            z = z * model.weight_views[i]
        views_z.append(z)
        link_logits = model.GCNs[i+1].decode(z, all_edge_index)
        loss += model.weight_views[i] *F.binary_cross_entropy_with_logits(link_logits, link_labels[i+1], pos_weight=pos_weight)
    
    z = torch.cat(views_z,1)
    
    z = z.unsqueeze(1).reshape(z.shape[0],len(data.edge_index_list),-1).transpose(1,2)

    if args.pooling == "max":
        z = F.max_pool2d(z, (1,model.num_graph)).squeeze(2)
    elif args.pooling == "mean":    
        z = F.avg_pool2d(z, (1,model.num_graph)).squeeze(2)
    elif args.pooling == "agg":
        z = z.sum(-1)
    elif args.pooling == "transformer":
        z = z.sum(-1)
        transformer_output = model.transformer_encoder(z[all_node_index])
        z1 = transformer_output[[all_node_index_map[i.item()] for i in all_edge_index[0]]]
        z2 = transformer_output[[all_node_index_map[i.item()] for i in all_edge_index[1]]]
        
    link_logits = model.decode(z, all_edge_index) if args.pooling != "transformer" else model.decode_transformer(z1, z2)
            
    loss += 2*F.binary_cross_entropy_with_logits(link_logits, link_labels[0], pos_weight=pos_weight)
    
    link_probs = link_logits.sigmoid()

    results = evaluate_performance(link_labels[0].cpu().numpy(), link_probs.cpu().numpy())

    return float(loss), results


@torch.no_grad()
def predict_oos(model, optimizer, data, device, pos_edge_index, neg_edge_index, esm_reps_flag, data_dir, celline_feats):
    model.eval()
    x = data.x.to(device)

    all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    all_node_index = torch.cat((all_edge_index[0], all_edge_index[1])).unique()
    all_node_index_map = {}  # 用来预测的边的位置进行标记
    for i, node_index in enumerate(all_node_index):
            all_node_index_map[node_index.item()] = i
    link_labels = get_link_labels(pos_edge_index, neg_edge_index, device)

    temp_z_list = []
    for i, edge_index in enumerate(data.edge_index_list):
        edge_index = edge_index.to(device)
        if args.model == 'GCN_multi' or args.model == 'GAT_multi':
            i = torch.tensor(i).to(device)
            temp_z = model.encode(x, edge_index, i)
        else:
            temp_z = model.encode(x, edge_index)
        temp_z_list.append(temp_z)
    
    z = torch.cat(temp_z_list,1)

    if args.pooling == "max" or "mean":
        z = z.unsqueeze(1).reshape(z.shape[0],len(data.edge_index_list),-1).transpose(1,2)
        z = F.max_pool2d(z, (1,len(data.edge_index_list))).squeeze(2) if args.pooling == "max" \
                else F.avg_pool2d(z, (1,len(data.edge_index_list))).squeeze(2)
        if esm_reps_flag:
            esm_representation = load_ESM_representations(data_dir, gene_mapping)
            esm_representation = esm_representation.to(device)
            z = torch.cat([z, esm_representation], dim=1)
        if args.MLP_celline:
            link_logits = model.MLP_decode(z, celline_feats, all_edge_index)
        else:
            link_logits = model.decode(z, all_edge_index)

    elif args.pooling == "attention":
        transformer_output = model.modified_transformer(z[all_node_index])
        z = torch.cat((transformer_output[
                                   [all_node_index_map[i.item()] for i in all_edge_index[0]]],
                               transformer_output[
                                   [all_node_index_map[i.item()] for i in all_edge_index[1]]]), dim=1)
        link_logits = model.decode(z)
    
    link_probs = link_logits.sigmoid()
    results = evaluate_performance(link_labels.cpu().numpy(), link_probs.cpu().numpy())
    return results


if __name__ == "__main__":
    args = init_argparse()
    print(args)
    graph_input = args.specific_graph + args.indep_graph
    print("Number of input graphs: {}".format(len(graph_input)))
    if len(graph_input) == 0:
        print("Please specify input graph features...")
        sys.exit(0)
    # load data
    data, SL_data_train, SL_data_val, SL_data_test, SL_data_novel, gene_mapping = generate_torch_geo_data(args.data_dir, args.data_source, args.CCLE, args.CCLE_dim, args.node2vec_feats, 
                                    args.threshold, graph_input, args.node_feats, args.split_method, args.predict_novel_cellline, args.novel_cellline,  args.training_percent)
    celline_feats = data.x
    num_features = data.x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    esm_dim = 0
    mlp_dim = 0
    if args.esm_reps_flag:
        esm_dim =1280
    if args.MLP_celline:
        mlp_dim = 16
        num_features = num_features + 16

    # load model
    model = SLMGAE(num_features, args.out_channels, len(data.edge_index_list),esm_dim, mlp_dim, args.data_dir, gene_mapping, celline_feats).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR)

    # generate SL torch data
    #train_pos_edge_index, train_neg_edge_index = generate_torch_edges(SL_data_train, args.balanced, True, device)
    val_pos_edge_index, val_neg_edge_index = generate_torch_edges(SL_data_val, True, False, device, 1)
    test_pos_edge_index, test_neg_edge_index = generate_torch_edges(SL_data_test, True, False, device, 1)
    if args.predict_novel_cellline:
        novel_pos_edge_index, novel_neg_edge_index = generate_torch_edges(SL_data_novel, True, False, device, 1)

    
    checkpoint_path = "../ckpt/{}_{}.pt".format(args.data_source,args.model)

    if args.MLP_celline:
        with torch.no_grad():
            MLP_output = model.cell_line_spec_mlp(celline_feats)
            data.x = torch.cat((data.x, MLP_output), dim = 1) 

    if args.predict_novel_cellline:
        #load check point
        print("Loading best model...")
        model.load_state_dict(torch.load(checkpoint_path))
        print("Predicting on novel the cell line...")
        data.edge_index_list
        results = predict_oos(model, optimizer, data, device, novel_pos_edge_index, novel_neg_edge_index)
        save_dict = {**vars(args), **results}
        
    else:
        train_losses = []
        valid_losses = []
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, reverse=True, path=checkpoint_path)
        node_embedding = data.x.to(device)
        SL_data_edge_index = data.edge_index_list[0].to(device)
        support_views_edge_index = [edge_index.to(device) for edge_index in data.edge_index_list[1:]]
        for epoch in range(1, args.epochs + 1):
            # in each epoch, using different negative samples
            train_pos_edge_index, train_neg_edge_index = generate_torch_edges(SL_data_train, args.balanced, True, device, args.neg_num)
            train_loss = train_model(model, optimizer, node_embedding, SL_data_edge_index, support_views_edge_index, device, train_pos_edge_index, train_neg_edge_index)
            train_losses.append(train_loss)
            val_loss, results = test_model(model, node_embedding, SL_data_edge_index, support_views_edge_index, device, val_pos_edge_index, val_neg_edge_index)
            valid_losses.append(val_loss)
            print('Epoch: {:03d}, loss: {:.4f}, AUC: {:.4f}, AP: {:.4f},  F1: {:.4f}, balance_acc: {:.4f}, val_loss: {:.4f}, precision@5: {:.4f}, precision@10: {:.4f}'.format(epoch, 
                                            train_loss, results['AUC'], results['AUPR'], results['F1'], results['balance_acc'], val_loss, results['precision@5'],results['precision@10']))
            
            #early_stopping(results['aupr'], model)
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early Stopping!!!")
                break
        
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(checkpoint_path))

        test_loss, results = test_model(model, node_embedding, SL_data_edge_index, support_views_edge_index, device, test_pos_edge_index, test_neg_edge_index)
    print("\ntest result:")
    print('AUC: {:.4f}, AP: {:.4f}, F1: {:.4f}, balance_acc: {:.4f}, precision@5: {:.4f}, precision@10: {:.4f}'.format(results['AUC'], results['AUPR'], results['F1'], results['balance_acc'], results['precision@5'], results['precision@10']))
    save_dict = {**vars(args), **results}
        
    if args.save_results:
        with open("../results/MVGCN_{}_{}_{}_{}_{}.json".format(args.data_source, args.model, args.pooling, args.MLP_celline, args.esm_reps_flag), "a") as f:
            json.dump(save_dict, f)
            f.write('\n')