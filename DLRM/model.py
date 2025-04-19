import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl.function as fn
import numpy as np
from torch.linalg import eigh


def reshape_PCA( emb , h_dim):
    mean = emb.mean(dim=0, keepdim=True)
    std = emb.std(dim=0, keepdim=True)
    embeddings_normalized = (emb - mean) / std
    cov_matrix = torch.mm(embeddings_normalized.t(), embeddings_normalized) / (emb.size(0) - 1)
    eigenvalues, eigenvectors = eigh(cov_matrix)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    selected_eigenvectors = eigenvectors[:, sorted_indices][:, : h_dim]
    reduced_embeddings = torch.mm(embeddings_normalized, selected_eigenvectors)
    return reduced_embeddings

class DQGCN(nn.Module):
    def __init__(self, timestamps, num_ents, num_rels,args):
        super(DQGCN, self).__init__()
        self.timestamps = timestamps
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.h_dim = args.n_hidden
        self.gpu = args.gpu
        self.alpha = args.alpha
        self.weight4f = args.weight4f
        self.pi = 3.14159265358979323846

        self.time_embs = torch.nn.Parameter(torch.Tensor(self.timestamps, self.h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.time_embs)
        self.time_w = torch.nn.Parameter(torch.Tensor(num_ents), requires_grad=True).float() 
        torch.nn.init.normal_(self.time_w)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  


        relemb_path = r'data\ICEWS14\relation&inverse_relation_tensor.pt'  # replace 
        emb_rel = torch.load(relemb_path)
        emb_rel=reshape_PCA(emb_rel,self.h_dim)
        self.emb_rel = nn.Parameter(emb_rel, requires_grad=True).float()

        emb_path = r'data\ICEWS14\entity_tensor.pt'  # replace
        static_emb = torch.load(emb_path)
        static_emb=reshape_PCA(static_emb,self.h_dim)
        self.static_emb = nn.Parameter(static_emb, requires_grad=True).float()

        self.temporal_w = torch.nn.Parameter(torch.Tensor(self.h_dim*2, self.h_dim), requires_grad=True).float()
        self.time_gate = nn.Linear(self.h_dim,self.h_dim)
        torch.nn.init.normal_(self.temporal_w)

        self.loss_e = torch.nn.CrossEntropyLoss()
        
        self.rgcn = RGCNCell(self.h_dim,
                             self.h_dim,
                             num_rels * 2,
                             args.n_layers,
                             args.dropout,
                             self.emb_rel)
        
        self.fourier = FourierConvLayer(self.h_dim, 
                                        self.h_dim, 
                                        activation=F.rrelu ) 

        self.decoder_ob = ConvTransE(num_ents, self.h_dim, args.dropout)
    
    def get_time_emb2(self,t,i):
        # return self.static_emb
        time_emb = self.time_embs[t.to(torch.int64).item() + i] 
        time_relu_t=self.leaky_relu(self.time_w) 
        time_relude_emb=torch.ger(time_relu_t,time_emb)
        attn = torch.cat([self.static_emb, time_relude_emb], 1) 
        return torch.mm(attn, self.temporal_w) 


    def forward(self, g_list, err_mat, t): 
        related_emb = torch.spmm(err_mat,self.emb_rel) 
        if self.weight4f!=0:
            for i in range(len(g_list)):
                if i==0:
                    self.inputs=[F.normalize(self.get_time_emb2(t, i))]
                else:
                    self.inputs.append(F.normalize(self.get_time_emb2(t, i))) 
            f_output = F.normalize(self.fourier.forward(g_list, self.inputs, self.gpu))
            self.inputs.append(f_output)
            self.inputs.append(self.weight4f * self.inputs[-1] + (1-self.weight4f) * F.normalize(self.get_time_emb2(t, 0)))
        else:
            self.inputs=[F.normalize(self.get_time_emb2(t, 0))]
        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            cur_output = F.normalize(self.rgcn.forward(g, self.inputs[-1]))
            self.inputs.append(self.get_composed(cur_output, related_emb))     
        return self.inputs[-1], self.emb_rel 

    def get_composed(self,cur_output, related_emb):
        self.time_weights = []
        for i in range(len(self.inputs)):
            self.time_weights.append(self.time_gate(self.inputs[i]+related_emb))
        self.time_weights.append(torch.zeros(self.num_ents,self.h_dim).cuda())
        self.time_weights = torch.stack(self.time_weights,0)
        self.time_weights = torch.softmax(self.time_weights,0)
        output = cur_output*self.time_weights[-1]
        for i in range(len(self.inputs)):
            output += self.time_weights[i]*self.inputs[i]
        return F.normalize(output)

    def predict(self, test_graph, tlist, test_triplets1, test_triplets2, err_mat1, err_mat2):
        with torch.no_grad():
            evolve_embs, r_emb = self.forward(test_graph, err_mat1, tlist[0])
            score1 = self.decoder_ob.forward(evolve_embs, r_emb, test_triplets1)

            evolve_embs, r_emb = self.forward(test_graph, err_mat2, tlist[0])
            score2 = self.decoder_ob.forward(evolve_embs, r_emb, test_triplets2)

            score = torch.cat([score1,score2])
            score = torch.softmax(score, dim=1)
            return score

    def get_loss(self, glist, tlist, triples, err_mat):
        evolve_embs, r_emb = self.forward(glist, err_mat, tlist[0])
        scores_ob = self.decoder_ob.forward(evolve_embs, r_emb, triples)
        loss_ent = self.loss_e(scores_ob, triples[:, 2])
        return loss_ent

class RGCNCell(nn.Module):
    def __init__(self, h_dim, out_dim, num_rels, 
                 num_hidden_layers=1, dropout=0,  rel_emb=None):
        super(RGCNCell, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.rel_emb = rel_emb

        self.layers = nn.ModuleList()
        for idx in range(self.num_hidden_layers):
            h2h = UnionRGCNLayer(self.h_dim, self.out_dim, self.num_rels,
                             activation=F.rrelu, dropout=self.dropout, rel_emb=self.rel_emb)
            self.layers.append(h2h)

    def forward(self, g, init_ent_emb): 
        node_id = g.ndata['id'].squeeze() 
        g.ndata['h'] = init_ent_emb[node_id] 
        for i, layer in enumerate(self.layers):
            layer(g, [])
        return g.ndata.pop('h') 
    
class FourierConvLayer(nn.Module): 
    def __init__(self, in_feat, out_feat, bias=None,
                 activation=None, dropout=0.2, sparsity_threshold=0.01):
        super(FourierConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.sparsity_threshold = sparsity_threshold 
        self.scale = 0.02
        self.ob = None
        self.sub = None
        self.dropout = nn.Dropout(dropout)  
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.in_feat, self.in_feat))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.in_feat, self.in_feat))
        self.w3 = nn.Parameter(self.scale * torch.randn(2, self.in_feat, self.in_feat))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.in_feat))        
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.in_feat))        
        self.b3 = nn.Parameter(self.scale * torch.randn(2, self.in_feat))

    def fourierGC(self, x, B, N, D):
        #初始化输出张量
        o1_real = torch.zeros([B, N//2 + 1, D],
                              device=x.device) 
        o1_imag = torch.zeros([B, N//2 + 1, D],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        o3_real = torch.zeros(x.shape, device=x.device)
        o3_imag = torch.zeros(x.shape, device=x.device)

        o1_real = F.relu( 
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
            self.b1[1]
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = self.dropout(y) 

        o2_real = F.relu(
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = F.relu(
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = self.dropout(x)  
        x = x + y

        o3_real = F.relu(
                torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
                torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
                self.b3[0]
        )

        o3_imag = F.relu(
                torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
                torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
                self.b3[1]
        )

        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = self.dropout(z) 
        z = z + x
        z = torch.view_as_complex(z)
        return z
    
    def forward(self, g_list, init_embs, gpu):
        x=[] 
        for i, g in enumerate(g_list):
            g = g.to(gpu)
            node_id = g.ndata['id'].squeeze() 
            g.ndata['h'] = init_embs[i] 
            x4g = g.ndata['h'] 
            x.append(x4g) 

        x = torch.stack(x, dim=0) 
        x = x.permute(1, 0, 2).contiguous() 
        B, N, D =x.shape
        x = torch.fft.rfft(x, dim=1, norm='ortho') 
        x = x.reshape(B, N//2+1, D) 
        bias = x
        # FourierGNN
        x = self.fourierGC(x, B, N, D) 
        x = x + bias
        x = x.reshape(B, N//2+1, D)
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho") 
        x = x.reshape(B, N, D) 
        x = x.permute(1, 0, 2)  
        x, _ = x.max(dim=0) 
        return x


class UnionRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, bias=None,
                 activation=None, dropout=0.0,  rel_emb=None):
        super(UnionRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation

        self.num_rels = num_rels
        self.emb_rel = rel_emb
        self.ob = None
        self.sub = None

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def propagate(self, g): 
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def forward(self, g, prev_h): 
        masked_index = torch.masked_select( 
            torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(), 
            (g.in_degrees(range(g.number_of_nodes())) > 0))
        loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight) 
        loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :] 
        self.propagate(g)
        node_repr = g.ndata['h'] 
        node_repr = node_repr + loop_message 
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr 
        return node_repr

    def msg_func(self, edges):
        relation = self.emb_rel.index_select(0, edges.data['type']).view(-1, self.out_feat) 
        edge_type = edges.data['type'] 
        edge_num = edge_type.shape[0] 
        node = edges.src['h'].view(-1, self.out_feat) 
        msg = node + relation
        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes): 
        return {'h': nodes.data['h'] * nodes.data['norm']}

class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, dropout=0, channels=50, kernel_size=3):

        super(ConvTransE, self).__init__()

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.embedding_dim = embedding_dim

        self.conv = torch.nn.Conv1d(2, channels, kernel_size, 
                                    stride=1, padding=int(math.floor(kernel_size/2)))
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)

        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, emb_rel, triplets):
        batch_size = len(triplets)
        e1_embedded_all = torch.tanh(embedding)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.dropout1(stacked_inputs)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.dropout3(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))

        return x
