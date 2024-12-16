import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DenseLayer(nn.Module):

    def __init__(self, 
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 zero_init=False, # initialize weights as zeros; use Xavier uniform init if zero_init=False
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out)

        # Initialization
        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, 
    			node_feats, # input node features
    			):

        node_feats = self.linear(node_feats)

        return node_feats


class GATSingleHead(nn.Module):

    def __init__(self, 
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 temp=1, # temperature parameter
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out)
        self.v0 = nn.Parameter(torch.Tensor(c_out, 1))
        self.v1 = nn.Parameter(torch.Tensor(c_out, 1))
        self.temp = temp

        # Initialization
        nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)
        nn.init.uniform_(self.v0.data, -np.sqrt(6 / (c_out + 1)), np.sqrt(6 / (c_out + 1)))
        nn.init.uniform_(self.v1.data, -np.sqrt(6 / (c_out + 1)), np.sqrt(6 / (c_out + 1)))

    def forward(self, 
                node_feats, # input node features
                adj_matrix, # adjacency matrix including self-connections
                ):

        # Apply linear layer and sort nodes by head
        node_feats = self.linear(node_feats)
        f1 = torch.matmul(node_feats, self.v0)
        f2 = torch.matmul(node_feats, self.v1)
        attn_logits = adj_matrix * (f1 + f2.T)
        unnormalized_attentions = (F.sigmoid(attn_logits) - 0.5).to_sparse()
        attn_probs = torch.sparse.softmax(unnormalized_attentions / self.temp, dim=1)
        attn_probs = attn_probs.to_dense()
        node_feats = torch.matmul(attn_probs, node_feats)

        return node_feats


class GATMultiHead(nn.Module):

    def __init__(self, 
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
    	         n_heads=1, # number of attention heads
    	         concat_heads=True, # concatenate attention heads or not
    	         ):

        super().__init__()

        self.n_heads = n_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % n_heads == 0, "The number of output features should be divisible by the number of heads."
            c_out = c_out // n_heads

        self.block = nn.ModuleList()
        for i_block in range(self.n_heads):
            self.block.append(GATSingleHead(c_in=c_in, c_out=c_out))

    def forward(self, 
                node_feats, # input node features
                adj_matrix, # adjacency matrix including self-connections
                ):

        res = []
        for i_block in range(self.n_heads):
            res.append(self.block[i_block](node_feats, adj_matrix))
        
        if self.concat_heads:
            node_feats = torch.cat(res, dim=1)
        else:
            node_feats = torch.mean(torch.stack(res, dim=0), dim=0)

        return node_feats


class DeconvNet(nn.Module):

    def __init__(self, 
                 hidden_dims, # dimensionality of hidden layers
                 n_celltypes, # number of cell types
                 n_slices, # number of slices
                 n_heads, # number of attention heads
                 slice_emb_dim, # dimensionality of slice id embedding
                 coef_fe,
                 ):

        super().__init__()

        # define layers
        # encoder layers
        self.encoder_layer1 = GATMultiHead(hidden_dims[0], hidden_dims[1], n_heads=n_heads, concat_heads=True)
        self.encoder_layer2 = DenseLayer(hidden_dims[1], hidden_dims[2])
        # decoder layers
        self.decoder_layer1 = GATMultiHead(hidden_dims[2] + slice_emb_dim, hidden_dims[1], n_heads=n_heads, concat_heads=True)
        self.decoder_layer2 = DenseLayer(hidden_dims[1], hidden_dims[0])
        # deconvolution layers
        self.deconv_alpha_layer = DenseLayer(hidden_dims[2] + slice_emb_dim, 1, zero_init=True)
        self.deconv_beta_layer = DenseLayer(hidden_dims[2], n_celltypes, zero_init=True)

        self.gamma = nn.Parameter(torch.Tensor(n_slices, hidden_dims[0]).zero_())

        self.slice_emb = nn.Embedding(n_slices, slice_emb_dim)

        self.coef_fe = coef_fe

    def forward(self, 
                adj_matrix, # adjacency matrix including self-connections
                node_feats, # input node features
                count_matrix, # gene expression counts
                # library_size, # library size (based on Y)
                slice_label, # slice label
                basis, # basis matrix，
                logit_step1, # n_step1 * K，加入 step1 的信息
                belong_matrix, # n_step1 * n_step2，表示 每个 小spot 归属于哪个 大sopt
                feat_step1, # step1 的 features
                counts_step1, # step1 的 counts
                libra_size_step1,
                ):
        # encoder
        Z = self.encoder(adj_matrix, node_feats)

        # deconvolutioner
        slice_label_emb = self.slice_emb(slice_label)
        beta, alpha = self.deconvolutioner(Z, slice_label_emb, logit_step1, belong_matrix) # 加入 step1 的信息

        # decoder
        node_feats_recon = torch.matmul(belong_matrix, self.decoder(adj_matrix, Z, slice_label_emb)) # 根据 belong_matrix, 把同一个spot内的所有 cell 加在一起

        # reconstruction loss of node features
        self.features_loss = torch.mean(torch.sqrt(torch.sum(torch.pow(counts_step1-node_feats_recon, 2), axis=1))) # node_feats_recon 和 step1 的 counts 的差距

        # deconvolution loss
        log_lam = torch.log(torch.matmul(belong_matrix, torch.matmul(beta, basis)) + 1e-6) + torch.matmul(belong_matrix, alpha) + torch.matmul(belong_matrix, self.gamma[slice_label]) # 根据 belong_matrix, 把同一个spot内的所有 cell 加在一起
        lam = torch.exp(log_lam)
        self.decon_loss = - torch.mean(torch.sum(counts_step1 * 
                                       (torch.log(libra_size_step1 + 1e-6) + log_lam) - libra_size_step1 * lam, axis=1)) # 换成 step1 的 counts 和 library_size

        # Total loss
        loss = self.decon_loss + self.coef_fe * self.features_loss

        return loss

    def evaluate(self, adj_matrix, node_feats, slice_label, logit_step1, belong_matrix): # 加入 step1 的信息
        slice_label_emb = self.slice_emb(slice_label)
        # encoder
        Z = self.encoder(adj_matrix, node_feats)

        # deconvolutioner
        beta, alpha = self.deconvolutioner(Z, slice_label_emb, logit_step1, belong_matrix) # 加入 step1 的信息

        return Z, beta, alpha, self.gamma
            
    def encoder(self, adj_matrix, node_feats):
        H = node_feats
        H = F.elu(self.encoder_layer1(H, adj_matrix))
        Z = self.encoder_layer2(H)
        return Z
        
    def decoder(self, adj_matrix, Z, slice_label_emb):
        H = torch.cat((Z, slice_label_emb), axis=1)
        H = F.elu(self.decoder_layer1(H, adj_matrix))
        X_recon = self.decoder_layer2(H)
        return X_recon
    
    def deconvolutioner(self, Z, slice_label_emb, logit_step1, belong_matrix):
        logit_step1 = torch.matmul(torch.transpose(belong_matrix, 0, 1), logit_step1) # step1 的信息
        logit_step2 = self.deconv_beta_layer(F.elu(Z))
        beta = F.softmax(logit_step1 + logit_step2, dim=1) # 加入 step1 的信息
        H = F.elu(torch.cat((Z, slice_label_emb), axis=1))
        alpha = self.deconv_alpha_layer(H)
        return beta, alpha
