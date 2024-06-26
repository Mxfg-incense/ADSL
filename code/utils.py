import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
import matplotlib.pyplot as plt
import os, random
import pickle, json,itertools
import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import Node2Vec
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from networkx.generators.random_graphs import fast_gnp_random_graph,gnp_random_graph
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import random
import ast


def generate_unique_samples(data_dir, cell_name):
    # process raw data
    if cell_name == "K562":
        df = pd.read_table(f"{data_dir}/raw_SL_experiments/K562/CRISPRi_K562_replicateAverage_GIscores_genes_inclnegs.txt", index_col=0)
    else:
        df = pd.read_table(f"{data_dir}/raw_SL_experiments/Jurkat/CRISPRi_Jurkat_emap_gene_filt.txt", index_col=0)
    
    # remove negative samples
    num_genes = df.shape[0]
    df = df.iloc[:(num_genes-1),:(num_genes-1)]
    
    num_genes = df.shape[0]
    
    # don't consider self interactions
    GI_matrix = df.values
    GI_indexs = np.triu_indices(num_genes, k=1)
    GI_values = GI_matrix[GI_indexs]
    
    # get corresponding gene names
    row_indexs = GI_indexs[0]
    col_indexs = GI_indexs[1]
    row_genes = df.index[row_indexs]
    col_genes = df.columns[col_indexs]
    
    all_samples = pd.DataFrame({'gene1':list(row_genes),'gene2':list(col_genes),'GI_scores':GI_values})
    all_samples.to_csv("{}/{}.csv".format(data_dir, cell_name), index=False)
    
    return all_samples

import os
import numpy as np
import torch

def load_ESM_representations(data_dir, gene_mapping):
    """
    Load mean gene representations from 'all_protein_esm2_mean_representations.pt' file.

    Args:
        data_dir (str): Directory containing the 'all_protein_esm2_mean_representations.pt' file.
        gene_mapping (dict): A dictionary that maps gene names to integer IDs.

    Returns:
        torch.Tensor: A 2D tensor of mean gene representations, where rows correspond to genes and columns correspond to features.
        int: The number of genes that were successfully mapped.
    """
    # Load the mean representations from the .pt file
    mean_reps_dict = torch.load(os.path.join(data_dir, 'mean_representations.pt'))

    # Create a new tensor to store the mean representations
    mean_reps = torch.zeros(len(gene_mapping), mean_reps_dict[list(mean_reps_dict.keys())[0]].shape[0])

    # Count the number of genes that were successfully mapped
    mapped_genes = 0

    # Fill the mean representations tensor using the gene mapping
    for gene, gene_id in gene_mapping.items():
        if gene in mean_reps_dict:
            mean_reps[gene_id] = mean_reps_dict[gene]
            mapped_genes += 1
    #print(f"Mapped {mapped_genes} genes out of {len(gene_mapping)}")

    return mean_reps

def load_SL_data(data_dir, cell_name, threshold=-3):
    if cell_name != "synlethdb":
        print(cell_name)
        data = pd.read_csv("{}/SL/{}_GIscore.csv".format(data_dir, cell_name))
        print(data.columns)
        # data['label'] = data['GI_scores'] <= threshold
        all_genes = list(set(np.unique(data['gene1'])) | set(np.unique(data['gene2'])))
    else:
        data = pd.read_csv(f"{data_dir}/SynLethDB_SL.csv")
        data.rename(columns={"gene_a.name":"gene1", "gene_b.name":"gene2"}, inplace=True)
        data = data[["gene1","gene2"]]
        data["label"] = 1
        all_genes = list(set(np.unique(data['gene1'])) | set(np.unique(data['gene2'])))
        # get sampled negative samples
        neg_df = generate_random_negative_samples(data, coeff=10)
        data = pd.concat([data, neg_df])
    return data, all_genes

def load_graph_data(data_dir, graph_type):
    if graph_type == 'PPI-genetic':
        # extract the first two columns
        data = pd.read_csv(f"{data_dir}/filtered_genetic.csv")
        data.rename(columns={"protein1":"gene1", "protein2":"gene2"}, inplace=True)
        # sample ppi
        sampled_data = data.sample(n=20000, random_state=42)
        # make it indirected graph
        data_dup = sampled_data.reindex(columns=['gene2','gene1'])
        data_dup.columns = ['gene1','gene2']
        sampled_data = pd.concat([sampled_data, data_dup])
        return sampled_data
    elif graph_type == 'PPI-physical':
        data = pd.read_csv(f"{data_dir}/filtered_physical.csv")
        data.rename(columns={"protein1":"gene1", "protein2":"gene2"}, inplace=True)
        # sample ppi
        sampled_data = data.sample(n=20000, random_state=42)
        # make it indirected graph
        data_dup = sampled_data.reindex(columns=['gene2','gene1'])
        data_dup.columns = ['gene1','gene2']
        sampled_data = pd.concat([sampled_data, data_dup])
        return sampled_data
    elif graph_type == "pathway":
        data = pd.read_csv(f"{data_dir}/Opticon_networks.csv")
        data.rename(columns={"Regulator":"gene1", "Target gene":"gene2"}, inplace=True)
    elif graph_type == 'co-exp' or graph_type == 'co-ess':
        if graph_type == 'co-exp':
            data = pd.read_csv(f"{data_dir}/coexpression_exp_0.5.csv")
            # sample coexpression
            sampled_data = data.sample(n=20000, random_state=42)
        elif graph_type == 'co-ess':
            data = pd.read_csv(f"{data_dir}/coexpression_ess_0.5.csv")
            # random choose 2000 pairs
            sampled_data = data.sample(n=20000, random_state=42)

        # make it indirected graph
        data_dup = sampled_data.reindex(columns=['gene2','gene1'])
        data_dup.columns = ['gene1','gene2']
        sampled_data = pd.concat([sampled_data, data_dup])
        return sampled_data
        
    elif graph_type == "random":
        data = pd.read_csv(f"{data_dir}/BIOGRID-9606.csv", index_col=0)
        print('random')
        data = data[data['Experimental System Type'] == 'physical']
        data = data[['Official Symbol Interactor A','Official Symbol Interactor B']]
        data.rename(columns={'Official Symbol Interactor A':'gene1', 'Official Symbol Interactor B':'gene2'}, inplace=True)
        all_genes = set(data['gene1'].unique()) | set(data['gene2'].unique())
        dict_mapping = dict(zip(range(len(all_genes)), all_genes))
        
        num_nodes = len(all_genes)
        num_edges = data.shape[0]
        p = 2*num_edges/(num_nodes*(num_nodes-1))
        # make it more sparse
        p = p/10.0
        
        G = fast_gnp_random_graph(num_nodes, p)
        data = nx.convert_matrix.to_pandas_edgelist(G,source='gene1',target='gene2')
        print("generated number of edges: {}".format(G.number_of_edges()))

    return data


def choose_node_attribute(data_dir, attr, gene_mapping, cell_name, graph_data):
    num_nodes = len(gene_mapping)
    if attr == "identity":
        x = np.identity(num_nodes)
    elif attr == 'random':
        x = np.random.randn(num_nodes, 4)
    elif attr == 'raw_omics':
        feat_list = ['exp','mut','cnv','ess']
        dict_list = []
        for feat in feat_list:
            temp_df = pd.read_table("{}/cellline_feats/{}_{}.txt".format(data_dir,cell_name,feat),
                                        names=['gene','value'], sep=r"\s+", na_values=[], keep_default_na=False)
            # filter genes
            temp_df = temp_df[temp_df['gene'].isin(list(gene_mapping.keys()))]
            temp_dict = dict(zip(temp_df['gene'].values, temp_df['value'].values))
            dict_list.append(temp_dict)
        
        x = np.zeros((num_nodes, len(feat_list)))
        for col_idx, feat_dict in enumerate(dict_list):
            for key, value in feat_dict.items():
                row_idx = gene_mapping[key]
                x[row_idx, col_idx] = value if value != '' else 0
        # standardize features
        x = scale(x)
        if np.isnan(x).any():
            print("nan value in raw_omics")
    elif attr == "CCLE_ess" or attr == "CCLE_exp":
        dim = 64
        df = pd.read_csv("{}/CCLE/{}.csv".format(data_dir, attr), index_col=0)
        df.fillna(0, inplace=True)
        df.columns = list(map(lambda x:x.split(" ")[0], df.columns))
        
        df = df.T
        all_genes = list(df.index)

        # perform PCA for dimension reduction
        values = df.values
        pca = PCA(n_components=dim)
        pca.fit(values)
        embed = pca.transform(values)
        embed_df = pd.DataFrame(embed, index=all_genes)

        x = np.zeros((num_nodes, dim))
        for gene, idx in gene_mapping.items():
            if gene in all_genes:
                x[idx] = embed_df.loc[gene]
        x = scale(x)
    elif attr == 'node2vec':
        # need to first build a torch data
        data_x = torch.tensor(np.random.randn(num_nodes, 128), dtype=torch.float)

        # concat all types of graph data
        graph_data_overall = pd.concat(graph_data)
        data_edge_index = torch.tensor([graph_data_overall['gene1'].values, graph_data_overall['gene2'].values], dtype=torch.long)
        
        data = Data(x=data_x, edge_index=data_edge_index)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=10, context_size=10, sparse=True).to(device)
        loader = model.loader(batch_size=512, shuffle=True, num_workers=28)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

        def train():
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss/len(loader)
        
        for epoch in range(1, 51):
            loss = train()
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        
        # get embeddings
        model.eval()
        x = model(torch.arange(data.num_nodes, device=device))
        print(x.size())
        x = x.cpu().detach().numpy()

    return x


def load_CCLE_feats(data_dir, process_method, feats_list, gene_mapping, hidden_dim):
    # load raw data
    df_list = []
    print("loading CCLE...")
    for feat in feats_list:
        print("loading {}".format(feat))
        df = pd.read_csv("{}/CCLE/CCLE_{}.csv".format(data_dir, feat), index_col=0)
        df.fillna(0, inplace=True)
        df.columns = list(map(lambda x:x.split(" ")[0], df.columns))
        df_list.append(df)
    
    # transform raw data
    embedding_list = []
    for df in df_list:
        embed = np.zeros((len(gene_mapping), hidden_dim))
        if process_method == 'PCA':
            temp_df = df[df.columns[df.columns.isin(list(gene_mapping.keys()))]]
            data = temp_df.values.T
            pca = PCA(n_components=hidden_dim)
            z = pca.fit_transform(data)

            # generate embeddings
            for i, gene_id in enumerate(temp_df.columns):
                embed[ gene_mapping[gene_id] ] = z[i]
        elif process_method == 'raw':
            temp_df = df[df.columns[df.columns.isin(list(gene_mapping.keys()))]]
            data = temp_df.values.T

            # generate embeddings
            embed = np.zeros((len(gene_mapping), data.shape[1]))
            for i, gene_id in enumerate(temp_df.columns):
                embed[ gene_mapping[gene_id] ] = data[i]
        
        embedding_list.append(embed)

    # combine embeddings
    combined_embeds = np.hstack(embedding_list)
    with open("{}/CCLE/embeddings_{}.npy".format(data_dir,process_method), "wb") as f:
        np.save(f, combined_embeds)
    
    return combined_embeds


def merge_and_mapping(SL_data, graph_data_list, SL_genes):
    # use the union of SL genes and graph genes as all genes
    temp_concat_graph_data = pd.concat(graph_data_list)
    graph_genes = set(temp_concat_graph_data['gene1'].unique()) | set(temp_concat_graph_data['gene2'].unique())
    all_genes = sorted(list(set(set(SL_genes)| graph_genes)))

    gene_mapping = dict(zip(all_genes, range(len(all_genes))))

    # converting gene names to id
    # iterating over all graph types
    for i in range(len(graph_data_list)):
        graph_data_list[i]['gene1'] = graph_data_list[i]['gene1'].apply(lambda x:gene_mapping[x])
        graph_data_list[i]['gene2'] = graph_data_list[i]['gene2'].apply(lambda x:gene_mapping[x])
    
    SL_data['gene1'] = SL_data['gene1'].apply(lambda x:gene_mapping[x])
    SL_data['gene2'] = SL_data['gene2'].apply(lambda x:gene_mapping[x])

    return SL_data ,graph_data_list, gene_mapping

    
def generate_torch_geo_data(data_dir, cell_name, CCLE_feats_flag, CCLE_hidden_dim, node2vec_feats_flag, threshold, graph_input, attr, split_method, predict_novel_flag, novel_cell_name, training_percent):
    """
    Generate torch geometric data for training a model.

    Args:
        
        cell_name (str): The name of the cell.
        CCLE_feats_flag (bool): Flag indicating whether to use CCLE features.
        CCLE_hidden_dim (int): The hidden dimension for CCLE features.
        node2vec_feats_flag (bool): Flag indicating whether to use node2vec features.
        threshold (float): The threshold for filtering positive SL pairs.
        graph_input (list): List of graph types to include in the input.
        attr (str): The attribute to use for node features.
        split_method (str): split the data based on selected genes or selected pairs.
        predict_novel_flag (bool): Flag indicating whether to predict novel samples.
        training_percent (float): The percentage of data to use for training.

    Returns:
        data (torch_geometric.data.Data): The torch geometric data: node features and edge indices.
        SL_data_train (pd.DataFrame): The training data for SL.
        SL_data_val (pd.DataFrame): The validation data for SL.
        SL_data_test (pd.DataFrame): The test data for SL.
        SL_data_oos (pd.DataFrame): The out-of-sample prediction data for SL.
        gene_mapping (dict): The mapping of genes to indices.
    """
    # load data

    SL_data, SL_genes = load_SL_data(data_dir, cell_name, threshold)
    
    # generate SL torch data, split into train, valid, test
    np.random.seed(5959)
    if split_method == "novel_gene":
        # only use pairwise combinations of selected genes as the training samples
        # all other samples can be validation samples
        np.random.shuffle(SL_genes)
        training_genes = SL_genes[:int(len(SL_genes)*training_percent)]
        val_genes = SL_genes[int(len(SL_genes)*training_percent):max(int(len(SL_genes)*training_percent) + 1 , \
                                                                        int(len(SL_genes)*(training_percent+0.1)))]
        test_genes = SL_genes[int(len(SL_genes)*(training_percent+0.1)):]
        print("#####################################")
        print("Number of genes in train, val and test:", len(training_genes), len(val_genes), len(test_genes))
        print("#####################################")
    elif split_method == "novel_pair":
        all_idx = list(range(len(SL_data)))
        np.random.shuffle(all_idx)
    
    # load graph data
    graph_data_list = []
    # get cell-specific expression values to filter unexpressed genes in the network
    # remove genes whose expression raw count == 0
    if cell_name != "synlethdb":
        exp_df = pd.read_table("{}/cellline_feats/{}_exp.txt".format(data_dir, cell_name),names=['gene','value'], sep=' ')
        exp_df_filtered = exp_df[exp_df["value"]>0]
        kept_genes = exp_df_filtered["gene"].values

    for graph_type in graph_input:
        if graph_type == "SL":
            if predict_novel_flag:
                # use training part of SL data to construct input graph
                if split_method == "novel_gene":
                    graph_data = SL_data_novel[(SL_data_novel['gene1'].isin(training_genes))&(SL_data_novel['gene2'].isin(training_genes))]
                elif split_method == "novel_pair":
                    graph_data = SL_data_novel.iloc[all_idx[:int(len(all_idx)*training_percent)]]
            else:
                # use training part of SL data to construct input graph
                if split_method == "novel_gene":
                    graph_data = SL_data[(SL_data['gene1'].isin(training_genes))&(SL_data['gene2'].isin(training_genes))]
                elif split_method == "novel_pair":
                    graph_data = SL_data.iloc[all_idx[:int(len(all_idx)*training_percent)]]
            
            graph_data = graph_data[graph_data['label']==True]
            graph_data = graph_data[['gene1','gene2']]
        elif graph_type == "PPI-genetic" or graph_type == "PPI-physical" :
            graph_data = load_graph_data(data_dir, graph_type)
            graph_data = pd.DataFrame(graph_data, columns=['gene1','gene2'])
            if cell_name != "synlethdb":
                graph_data = graph_data[(graph_data["gene1"].isin(kept_genes))&(graph_data["gene2"].isin(kept_genes))]
        else:
            graph_data = load_graph_data(data_dir, graph_type)
            if cell_name != "synlethdb":
                graph_data = graph_data[(graph_data["gene1"].isin(kept_genes))&(graph_data["gene2"].isin(kept_genes))]
        print(graph_type, graph_data.shape[0])
        if graph_data.shape[0] == 0:
            continue
        graph_data_list.append(graph_data)
        
    # merge, filter and mapping
    SL_data, graph_data_list, gene_mapping = merge_and_mapping(SL_data, graph_data_list, SL_genes)

    # generate node features
    x = choose_node_attribute(data_dir, attr, gene_mapping, cell_name, graph_data_list)
    
    # generate torch data
    data_x = torch.tensor(x, dtype=torch.float)
    
    data_edge_index_list = []
    for graph_data in graph_data_list:
        # 假设 graph_data['gene1'] 和 graph_data['gene2'] 是 numpy.ndarray
        gene1_arr = np.array(graph_data['gene1'])
        gene2_arr = np.array(graph_data['gene2'])

        # 将 numpy.ndarray 转换为 PyTorch 张量
        temp_edge_index = torch.from_numpy(np.stack([gene1_arr, gene2_arr], axis=0)).long()
        data_edge_index_list.append(temp_edge_index)
        
    data = Data(x=data_x, edge_index_list=data_edge_index_list)
    
    if split_method == "novel_gene":
        training_genes = list(map(gene_mapping.get, training_genes))
        val_genes = list(map(gene_mapping.get, val_genes))
        test_genes = list(map(gene_mapping.get, test_genes))
        SL_data_train = SL_data[(SL_data['gene1'].isin(training_genes))&(SL_data['gene2'].isin(training_genes))]
        SL_data_val = SL_data[(SL_data['gene1'].isin(val_genes))&(SL_data['gene2'].isin(val_genes))]
        SL_data_test = SL_data[(SL_data['gene1'].isin(test_genes))&(SL_data['gene2'].isin(test_genes))]
    elif split_method == "novel_pair":
        end1 = int(len(all_idx)*training_percent)
        end2 = int(len(all_idx)*(training_percent+0.125))
        print(all_idx[end1], all_idx[end2], len(all_idx))
        SL_data_train = SL_data.iloc[all_idx[:end1]]
        SL_data_val = SL_data.iloc[all_idx[end1:end2]]
        SL_data_test = SL_data.iloc[all_idx[end2:]]
    
    # print info
    num_pos_train = SL_data_train[SL_data_train["label"]==True].shape[0]
    num_pos_val = SL_data_val[SL_data_val["label"]==True].shape[0]
    num_pos_test = SL_data_test[SL_data_test["label"]==True].shape[0]
    print("#####################################")
    print("Number of positive samples in train, val and test:", num_pos_train, num_pos_val, num_pos_test)
    print("#####################################")
    
    return data, SL_data_train, SL_data_val, SL_data_test, gene_mapping


def generate_torch_edges(df, balanced_sample, duplicate, device, neg_num):
    df_pos = df[df['label'] == True]
    if balanced_sample:
        # balanced sample
        df_neg = df[df['label'] == False].sample(n=df_pos.shape[0])
        
    else:
        h = int(df_pos.shape[0] * neg_num)

        df_neg = df[df['label'] == False].sample(n=h)
    
    #pos_edge_idx = torch.tensor([df_pos['gene1'].values, df_pos['gene2'].values], dtype=torch.long, device=device)
    #neg_edge_idx = torch.tensor([df_neg['gene1'].values, df_neg['gene2'].values], dtype=torch.long, device=device)

    if duplicate == True:
        pos_edge_idx = torch.tensor(np.array([np.concatenate((df_pos['gene1'].values, df_pos['gene2'].values)),
                                                     np.concatenate((df_pos['gene2'].values,df_pos['gene1'].values))]), dtype=torch.long, device=device)
        neg_edge_idx = torch.tensor(np.array([np.concatenate((df_neg['gene1'].values, df_neg['gene2'].values)),
                                                     np.concatenate((df_neg['gene2'].values,df_neg['gene1'].values))]), dtype=torch.long, device=device)
    else:
        pos_edge_idx = torch.tensor(np.array([df_pos['gene1'].values, df_pos['gene2'].values]), dtype=torch.long, device=device)
        neg_edge_idx = torch.tensor(np.array([df_neg['gene1'].values, df_neg['gene2'].values]), dtype=torch.long, device=device)
    
    return pos_edge_idx, neg_edge_idx


def get_link_labels(pos_edge_index, neg_edge_index, support_views_edge_index, device):
    """
    Generate link labels for positive and negative edges for each view.
    support_views_edge_index: (#views, 2, #edges)
    """
    all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    num_links = all_edge_index.size(1)
    link_labels = torch.zeros((len(support_views_edge_index) + 1, num_links), device=device)
    # SL_data labels
    link_labels[0, :pos_edge_index.size(1)] = 1
    # support views labels
    for _ in range(num_links):
        edge = all_edge_index[:, _].reshape(2,1)
        # whether the edge is in the support views(2, #edges)
        for i, view_edge_index in enumerate(support_views_edge_index):
            edge_expanded = edge.expand_as(view_edge_index)
            if torch.any(torch.all(edge_expanded == view_edge_index, dim=0)):
                link_labels[i + 1, _] = 1

    return link_labels
    
def calculate_coexpression(data_dir, data_type, rho_thres):
    df = pd.read_csv("{}/CCLE/CCLE_{}.csv".format(data_dir,data_type), index_col=0)
    df.fillna(0, inplace=True)
    df.columns = list(map(lambda x:x.split(" ")[0], df.columns))
    ind_mapping = dict(zip(range(df.shape[1]),df.columns.values))

    print("Calculating coexpression...")
    rho, pval = stats.spearmanr(df.values)
    
    rho_lower = np.tril(rho >= rho_thres, k=-1)
    ind_keep = list(zip(*np.where(rho_lower==True)))

    # convert index to entrez ids
    ind_keep_df = pd.DataFrame(ind_keep, columns=['gene1','gene2'])
    ind_keep_df['gene1'] = ind_keep_df['gene1'].apply(lambda x:ind_mapping[x])
    ind_keep_df['gene2'] = ind_keep_df['gene2'].apply(lambda x:ind_mapping[x])

    ind_keep_df.to_csv("{}/coexpression_{}_{}.csv".format(data_dir,data_type,str(rho_thres)), index=False)

    plt.hist(rho.flatten(), bins=1000)
    plt.xlim(-1,1)
    plt.savefig("hist_rho_{}.png".format(data_type))

def ranking_metrics(true_labels, pred_scores, top=0.05):
    sorted_index = np.argsort(-pred_scores)
    top_num = int(top * len(true_labels))
    sorted_true_labels = true_labels[sorted_index[:top_num]]
    if top_num == 0:
        acc = 0
    else:
        acc = float(sorted_true_labels.sum())/float(top_num)
    return acc


def evaluate_performance(label, pred):
    print("Label last 10:", label[-10:])    
    print("Pred last 10:", pred[-10:])
    AUC = roc_auc_score(label, pred)
    AUPR = average_precision_score(label, pred)
    F1 = f1_score(label, np.round(pred))
    balance_acc = balanced_accuracy_score(label, np.round(pred))  # Add balanced accuracy measurement
    
    rank_score_5 = ranking_metrics(label, pred, top=0.05)
    rank_score_10 = ranking_metrics(label, pred, top=0.1)

    performance_dict = {"AUC":AUC, "AUPR":AUPR, "F1":F1, "balance_acc":balance_acc, "precision@5":rank_score_5, "precision@10":rank_score_10}

    return performance_dict

def compute_fmax(true_labels, pred_scores):
    pred_scores = np.round(pred_scores, 2)
    true_labels = true_labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (pred_scores > threshold).astype(np.int32)
        tp = np.sum(predictions * true_labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(true_labels) - tp
        sn = tp / (1.0 * np.sum(true_labels))
        sp = np.sum((predictions ^ 1) * (true_labels ^ 1))
        sp /= 1.0 * np.sum(true_labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max, p_max, r_max, t_max

def generate_random_negative_samples(pos_samples, coeff=10):
    # randomly generate same amount of negative samples as positive samples times coeff
    # generate coeff times number of positive samples
    num = pos_samples.shape[0] * coeff
    all_genes = list(set(pos_samples['gene1'].unique()) | set(pos_samples['gene2'].unique()))
    neg_candidates_1 = random.choices(all_genes, k=2*num)
    neg_candidates_2 = random.choices(all_genes, k=2*num)
    
    pos_list = [tuple(r) for r in pos_samples[['gene1','gene2']].to_numpy()] + [tuple(r) for r in pos_samples[['gene2','gene1']].to_numpy()]
    sampled_list = list(zip(neg_candidates_1, neg_candidates_2))
    # remove gene pairs that have positive effects
    remained_list = set(sampled_list) - set(pos_list)
    # remove gene pairs where gene1 = gene2
    remained_list = [x for x in remained_list if x[0] != x[1]]
    remained_list = random.sample(remained_list, num)
    
    neg_df = pd.DataFrame({"gene1":[x[0] for x in remained_list],
                           "gene2":[x[1] for x in remained_list],
                           "label":[0]*num})
    return neg_df
