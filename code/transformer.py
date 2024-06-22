# %%
# code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

# def make_batch(sentences):
#     input_batch = [[src_vocab[n] for n in sentences[0].split()]]
#     output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
#     target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
#     return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
    # batch_size, len_q = (1,len(seq_q))
    # batch_size, len_k = (1,len(seq_k))
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self,opt):
        self.opt=opt
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.opt.d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,opt):
        self.opt=opt
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(opt.d_model, opt.d_k * opt.n_heads)
        self.W_K = nn.Linear(opt.d_model, opt.d_k * opt.n_heads)
        self.W_V = nn.Linear(opt.d_model, opt.d_k * opt.n_heads)
        self.linear = nn.Linear(opt.n_heads * opt.d_k, opt.d_model)
        self.layer_norm = nn.LayerNorm(opt.d_model)

    def forward(self, Q, K, V, attn_mask,edge_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.opt.n_heads, self.opt.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.opt.n_heads, self.opt.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.opt.n_heads, self.opt.d_k).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        #attn_mask = attn_mask.unsqueeze(1).repeat(1, self.opt.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        attn_mask = edge_mask #这一行是追加的内容，使用图上的知识做mask

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention(self.opt)(q_s, k_s, v_s, attn_mask,)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.opt.n_heads * self.opt.d_k) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,opt):
        super(PoswiseFeedForwardNet, self).__init__()
        self.opt=opt
        self.conv1 = nn.Conv1d(in_channels=opt.d_model, out_channels=opt.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=opt.d_ff, out_channels=opt.d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(opt.d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self,opt):
        super(EncoderLayer, self).__init__()
        self.opt=opt
        self.enc_self_attn = MultiHeadAttention(opt)
        self.pos_ffn = PoswiseFeedForwardNet(opt)

    def forward(self, enc_inputs, enc_self_attn_mask,edge_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask,edge_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self,opt):
        super(Encoder, self).__init__()
        #self.src_emb = nn.Embedding(src_vocab_size, d_model)
        #self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model),freeze=True)
        self.opt=opt
        self.layers = nn.ModuleList([EncoderLayer(opt) for _ in range(opt.n_layers)])

    def forward(self, enc_inputs,edge_mask): # enc_inputs : [batch_size x source_len]
        #enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1,2,3,4,0]]))
        #enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attn_mask = edge_mask
        enc_self_attns = []
        enc_outputs=enc_inputs
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask,edge_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
#         self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
#         self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
#
#     def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
#         dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5,1,2,3,4]]))
#         dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
#         dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
#         dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
#
#         dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
#
#         dec_self_attns, dec_enc_attns = [], []
#         for layer in self.layers:
#             dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
#             dec_self_attns.append(dec_self_attn)
#             dec_enc_attns.append(dec_enc_attn)
#         return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self,opt):
        super(Transformer, self).__init__()
        self.opt=opt
        self.line1=nn.Linear((len(self.opt.indep_graph) + 1) * opt.out_channels, self.opt.d_model)
        self.encoder = Encoder(self.opt)

        #self.decoder = Decoder()
        #self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, edge_mask):
        nn.TransformerEncoder
        enc_inputs= self.line1(enc_inputs)
        enc_outputs, enc_self_attns = self.encoder(enc_inputs.unsqueeze(0),edge_mask)
        #dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        #dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        #return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
        return enc_outputs.squeeze(0)

# def showgraph(attn):
#     attn = attn[-1].squeeze(0)[0]
#     attn = attn.squeeze(0).data.numpy()
#     fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
#     ax = fig.add_subplot(1, 1, 1)
#     ax.matshow(attn, cmap='viridis')
#     ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
#     ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
#     plt.show()

def process_points_and_edges(all_points,points_index,edges,opt):
    index_dict={}
    edges_matrix=np.zeros((opt.src_len,opt.src_len))
    points_index=sorted(points_index)
    point=all_points[points_index]

    for i,elem in enumerate(points_index):
        index_dict[elem]=i

    for node_pair in edges:
        edges_matrix[index_dict[node_pair[0]],index_dict[node_pair[1]]]=1

    edges_matrix_list=[]
    temp=torch.from_numpy(edges_matrix)

    edges_matrix_list.append(temp)
    for i in range(opt.n_heads-1):
        edges_matrix_list.append(torch.matmul(edges_matrix_list[i],temp))

    return (all_points,[edges_matrix_list,torch.cat(edges_matrix_list,dim=1)])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_len', default=5, type=int, help='length of source')
    parser.add_argument('--tgt_len', default=5, type=str, help='length of target')
    parser.add_argument('--d_model', default=512, type=int, help='Embedding Size')
    parser.add_argument('--d_ff', default=2048, type=int, help='FeedForward dimension')
    parser.add_argument('--d_k', default=64, type=int, help='dimension of K(=Q), V')
    parser.add_argument('--d_v', default=64, type=int, help='dimension of K(=Q), V')
    parser.add_argument('--d_q', default=64, type=int, help='dimension of K(=Q), V')
    parser.add_argument('--n_layers', default=6, type=int, help='number of Encoder of Decoder Layer')
    parser.add_argument('--n_heads', default=8, type=int, help='number of heads in Multi-Head Attention')
    opt = parser.parse_args()

    #sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    # Transformer Parameters
    # Padding Should be Zero
    #src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    #src_vocab_size = len(src_vocab)

    # tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    # number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    # tgt_vocab_size = len(tgt_vocab)

    print(opt)

    model = Transformer(opt)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    enc_inputs=torch.rand()


    all_points='temp'
    points_index='temp'
    edges='temp'

    points,edges_matrixs=process_points_and_edges(all_points,points_index,edges)

    outputs = model(points,edges_matrixs)

    #enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    # for epoch in range(20):
    #     optimizer.zero_grad()
    #     outputs = model(enc_inputs,edge_mask)
    #     loss = criterion(outputs, target_batch.contiguous().view(-1))
    #     print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    #     loss.backward()
    #     optimizer.step()
    #
    # # Test
    # predict, _, _, _ = model(enc_inputs, dec_inputs)
    # predict = predict.data.max(1, keepdim=True)[1]
    # print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])
    #
    # print('first head of last state enc_self_attns')
    # showgraph(enc_self_attns)
    #
    # print('first head of last state dec_self_attns')
    # showgraph(dec_self_attns)
    #
    # print('first head of last state dec_enc_attns')
    # showgraph(dec_enc_attns)