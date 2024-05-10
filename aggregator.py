import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import scipy.sparse as sp
import math

class RGATLayer(nn.Module):#RGAT（Relational Graph Attention Network）关系图注意力网络，专注于在图结构数据中编码节点之间的关系信息。
    def __init__(self, dim, edge_type_num, alpha, dropout=0.):
        super(RGATLayer, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.edge_type_num = edge_type_num # 设置边的类型数量
        self.training_edges = list(range(1, self.edge_type_num+1)) # 1~11 训练时使用的边的列表 edge_type_num=10
        # self.inference_edges = [1,2,4]
        self.inference_edges = self.training_edges #推理时使用的边的列表，这里与训练时的边列表相同

        self.W = nn.Parameter(torch.FloatTensor(dim, dim))# 定义一个可学习的权重矩阵 W，形状为 (dim, dim)
        # 定义一个参数列表 a_params，包含 edge_type_num 个形状为 (dim, 1) 的可学习参数
        self.a_params = nn.ParameterList([nn.Parameter(torch.Tensor(self.dim, 1)) for i in range(self.edge_type_num)]) # R*[D 1]
        self.bias = nn.Parameter(torch.Tensor(self.dim))
        # 创建一个可学习的参数，用于存储放大器的相对比例。该参数的形状为(len(self.training_edges), 1)，即训练边的数量乘以1。
        self.amplifier_rel_ratio = nn.Parameter(torch.Tensor(len(self.training_edges), 1))
        self.feed_forward_layer = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        # 创建一个可学习的参数，用于实现前馈神经网络层。该参数的形状为(2 * self.dim, self.dim)，即输入维度的两倍乘以输出维度。
        self.leakyrelu = nn.LeakyReLU(alpha)# 创建一个LeakyReLU激活函数，alpha参数控制负数部分的斜率。

    def forward(self, hidden, adj, node_type_mask, stage):
        batch_size = hidden.shape[0]
        N = hidden.shape[1] # number of nodes
        hidden = torch.matmul(hidden, self.W)# 对隐藏层进行矩阵乘法操作
        if stage == 'train': #根据阶段选择训练边或推理边
            selected_edges = self.training_edges
        elif stage == 'test':
            selected_edges = self.inference_edges

        a_input = (hidden.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * hidden.repeat(1, N, 1)).view(batch_size, N, N, self.dim) # [B L L D]

        alpha = -9e15 * torch.ones(batch_size, N, N).cuda() # [B L L]
        for r in selected_edges:# 遍历选定的边，计算alpha值
            r_idx = r-1
            e = torch.matmul(a_input, self.a_params[r_idx])
            e = self.leakyrelu(e).squeeze(-1).view(batch_size, N, N)
            alpha = torch.where(adj.eq(r), e, alpha)	# condition, if ture=x, if false=y

        # # @_@: Relation Amplifier, having it is betters # 遍历选定的边，调整alpha值
        for rel_idx in selected_edges:
            alpha = torch.where(adj.eq(rel_idx), alpha*self.amplifier_rel_ratio[rel_idx-1], alpha)	# condition, if ture=x, if false=y

        alpha = torch.softmax(alpha, dim=-1)# 对alpha进行softmax操作
        output = torch.matmul(alpha, hidden)#计算输出

        # feed-forward
        # output = torch.matmul(torch.cat([output, hidden], -1), self.feed_forward_layer)

        return output # [B L D]



class HeteAttenLayer(nn.Module):
    # @_@: Dot-Attention + QKV-Attention
    def __init__(self, config, dim, edge_type_num, alpha, dropout=0.):
        super(HeteAttenLayer, self).__init__()
        self.dim = dim
        self.fusion_type = config['fusion_type']
        self.node_type_num = 3
        self.edge_type_num = edge_type_num#设置边类型的数量
        self.training_edges = list(range(1, self.edge_type_num+1)) # 1~11

        # @_@: for homogenous relations (1, 4, 9 region)
        self.homo_edges = list(range(1,5)) #同质边列表存放1-4的正数,存放同质边的编号
        self.hete_edges = [1]+list(range(5,11))#异质边列表存放1然后是5-10的整数 存放异质边的编号

        # @_@: for homogenous relations，遍历同质边列表为每条边创建一个[D,1]的张量作为参数
        self.a_params = nn.ParameterList([nn.Parameter(torch.Tensor(self.dim, 1)) for i in self.homo_edges]) # R*[D 1]
        
        # @_@: for heterogenous relations 
        self.W_Q = nn.Parameter(torch.Tensor(4, dim, dim)) # [4 D D], 1,2,3 are item, image, text weight
        self.W_K = nn.Parameter(torch.Tensor(4, dim, dim)) # [4 D D], 1,2,3 are item, image, text weight
        self.W_V = nn.Parameter(torch.Tensor(4, dim, dim)) # [4 D D], 1,2,3 are item, image, text weight
        
        self.cat_projection = nn.Linear(2*self.dim, self.dim) # for concatenation # 定义一个线性层，输入维度为2倍的dim，输出维度为dim
        if self.fusion_type == 'cat':
            self.projection = nn.Linear(2*self.dim, self.dim) # for concatenation
        elif self.fusion_type == 'gate':
            self.projection = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(True), nn.Linear(self.dim, 1)) # for gate
            self.global_local_projection = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(True), nn.Linear(self.dim, 1)) # for gate
        elif self.fusion_type == 'asy_mask':
            self.projection = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(True), nn.Linear(self.dim, 1)) # for gate
            self.global_local_projection = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(True), nn.Linear(self.dim, 1)) # for gate

        # Feed-Forward
        dff = self.dim * 2
        self.fc1 = nn.Linear(self.dim, dff)  # TODO: define dff dim
        self.fc2 = nn.Linear(dff, self.dim)
        # Output
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.fc_out = nn.Linear(self.dim, self.dim)
        self.dropout_atten = nn.Dropout(dropout)

        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)
    
    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def get_qkv_hidden(self, hidden, node_type_mask):
        # @_@: transformed hidden
        batch_size = hidden.shape[0]
        N = hidden.shape[1] # number of nodes
        #QKV隐藏层，【B*N D】的全零张量
        query_hidden = torch.zeros(batch_size* N, self.dim).cuda() # [B*N D]
        key_hidden = torch.zeros(batch_size* N, self.dim).cuda() # [B*N D]
        value_hidden = torch.zeros(batch_size* N, self.dim).cuda() # [B*N D]
        node_type_mask = node_type_mask.reshape(-1) # [B,N]->[B*N]一维张量
        flatten_hidden = hidden.reshape(-1, hidden.size(-1)) # [B*N D]#flatten_hidden的作用是将原始的hidden张量进行形状变换，使其能够被模型接受作为输入。
        for n in range(1, self.node_type_num+1): # for differnt type of nodes
            n_idx = node_type_mask==n
            _query_hidden = torch.matmul(flatten_hidden, self.W_Q[n]) # [B*N D]
            query_hidden[n_idx, :] = _query_hidden[n_idx, :]# 将计算得到的查询向量赋值给query_hidden的第n_idx行
            _key_hidden = torch.matmul(flatten_hidden, self.W_K[n]) # [B*N D]
            key_hidden[n_idx, :] = _key_hidden[n_idx, :]
            _value_hidden = torch.matmul(flatten_hidden, self.W_V[n]) # [B*N D]
            value_hidden[n_idx, :] = _value_hidden[n_idx, :]
        query_hidden = query_hidden.reshape(batch_size, N, self.dim)
        key_hidden = query_hidden.reshape(batch_size, N, self.dim)
        value_hidden = query_hidden.reshape(batch_size, N, self.dim)
        return query_hidden, key_hidden, value_hidden #均是【B N D】

    def calculate_homo_hidden(self, hidden, adj):#adj是邻接矩阵 #计算同质隐藏层用于节点间的信息传递和聚合。
        batch_size = hidden.shape[0]
        N = hidden.shape[1] # number of nodes
        # @_@: content-based attention 主要关注输入序列内部元素之间的关系。在自注意力机制中，查询（Query）、键（Key）和值（Value）通常来自同一个输入序列
        a_input = (
                    hidden.repeat(1, 1, N).view(batch_size, N * N, self.dim) * 
                    hidden.repeat(1, N, 1)
                    ).view(batch_size, N, N, self.dim)  # [B N N D]
        alpha = -9e15 * torch.ones(batch_size, N, N).cuda()  # [B N N]
        for r in self.homo_edges:
            r_idx = r-1
            e = torch.matmul(a_input, self.a_params[r_idx])
            e = self.leakyrelu(e).squeeze(-1).view(batch_size, N, N)
            alpha = torch.where(adj.eq(r), e, alpha)	# condition, if ture=x, if false=y
        
        content_attention_weights = torch.softmax(alpha, dim=-1)
        content_attention_weights = self.dropout_atten(content_attention_weights)
        homo_output = torch.matmul(content_attention_weights, hidden)
        return content_attention_weights, homo_output
    
    def calculate_hete_hidden(self, query_hidden, key_hidden, value_hidden, adj):
        # @_@: cross attention交叉注意力机制交叉注意力则涉及到两个不同的序列，其中一个序列提供查询（Query），另一个序列提供键（Key）和值（Value）。这种设置使得交叉注意力能够建立不同序列之间的联系。
        scross_a_scores = torch.matmul(query_hidden, key_hidden.transpose(-2, -1))  # [B N N]查询和键之间的相似度得分。
        
        global_a_scores = self.dropout_atten(torch.tanh(scross_a_scores))
        global_output = torch.matmul(global_a_scores, value_hidden)
        global_output = self.dropout_atten(global_output)

        scores = -9e15 * torch.ones_like(scross_a_scores) # [B L L]
        for r in self.hete_edges:
            scores = torch.where(adj.eq(r), scross_a_scores, scores)	# condition, if ture=x, if false=y
        scores = self.leakyrelu(scores)
        cross_attention_weights = torch.softmax(scores, dim=-1)
        cross_attention_weights = self.dropout_atten(cross_attention_weights)
        hete_output = torch.matmul(cross_attention_weights, value_hidden)   # [B N D] from cross-attention

        return global_a_scores, global_output, cross_attention_weights, hete_output
#global_output全局输出特征，global_out能够捕捉图中所有节点的信息，对所有节点的隐藏状态进行聚合
    def forward(self, hidden, adj, node_type_mask, stage='train'):
        '''
        nodes' hidden [B N D]
        asy_mask [B N 2]
        ''' 
        ablation = 0

        query_hidden, key_hidden, value_hidden = self.get_qkv_hidden(hidden, node_type_mask)
        _, homo_output = self.calculate_homo_hidden(value_hidden, adj)
        _, global_output, _, hete_output = self.calculate_hete_hidden(query_hidden, key_hidden, value_hidden, adj)

        if ablation == 1:
            self.fusion_type = 'cat'
        elif ablation == 5:
            return homo_output
        elif ablation == 6:
            return hete_output

        # @_@: concatenation
        if self.fusion_type == 'cat':
            attention_output = self.cat_projection(torch.cat((homo_output, hete_output), -1))
        # @_@: gate
        elif self.fusion_type == 'gate':
            input_h = torch.cat((homo_output.unsqueeze(-2), hete_output.unsqueeze(-2)), -2) # [B N 2 D]
            energy = self.projection(input_h) # [B N 2 1]
            weights = torch.softmax(energy.squeeze(-1), dim=-1) # [B, N, 2]
            local_output = (input_h * weights.unsqueeze(-1)).sum(dim=-2) # # (B, N, 2, D) * (B, N, 2, 1) -> (B, N, D)
            # step 2
            input_h = torch.cat((local_output.unsqueeze(-2), global_output.unsqueeze(-2)), -2) # [B N 2 D]
            energy = self.global_local_projection(input_h) # [B N 3 1]
            weights = torch.softmax(energy.squeeze(-1), dim=-1) # [B, N, 3]
            attention_output = (input_h * weights.unsqueeze(-1)).sum(dim=-2)
        elif self.fusion_type == 'asy_mask': # TODO: here only step of 2; can be intergrated into gate
            BA_scores, BA_output = self.calculate_homo_hidden(hete_output, adj)#先同后异
            noninvasive_BA_output = torch.matmul(BA_scores, value_hidden) # non-invasive
            
            _, _, AB_scores, AB_output = self.calculate_hete_hidden(query_hidden, key_hidden, homo_output, adj)
            noninvasive_AB_output = torch.matmul(AB_scores, value_hidden)   # [B N D] from cross-attention

            if ablation == 2:
                return noninvasive_BA_output
            elif ablation == 3:
                return noninvasive_AB_output
            # 将noninvasive_AB_output和noninvasive_BA_output在最后一个维度上拼接起来，得到input_h
            input_h = torch.cat((noninvasive_AB_output.unsqueeze(-2), noninvasive_BA_output.unsqueeze(-2)), -2) # [B N 2 D]
            asy_mask = self.projection(input_h) # [B N 2 1]调用三层映射模型，得到异步掩码; TODO: add a loss to make it binary
            weights = torch.softmax(asy_mask.squeeze(-1), dim=-1) # [B, N, 2]
            local_output = (input_h * weights.unsqueeze(-1)).sum(dim=-2) # # (B, N, 2, D) * (B, N, 2, 1) -> (B, N, D)
            attention_output = local_output#local_output通常指的是节点的局部邻居信息，它关注的是节点及其直接相连的邻居之间的关系。
            # local+global
            # input_h = torch.cat((local_output.unsqueeze(-2), global_output.unsqueeze(-2)), -2) # [B N 2 D]
            # energy = self.global_local_projection(input_h) # [B N 3 1]
            # weights = torch.softmax(energy.squeeze(-1), dim=-1) # [B, N, 3]
            # attention_output = (input_h * weights.unsqueeze(-1)).sum(dim=-2)

        return attention_output
    
        # hidden_states = self.gelu(self.fc1(attention_output))
        # feed_forward_output = self.dropout(self.fc2(hidden_states))
        # output = self.norm2(attention_output + feed_forward_output)
        # output = self.fc_out(hidden)
        # return output # [B L D]


class KVAttentionLayer(nn.Module):
    # @_@: QKV-Attention
    def __init__(self, dim, edge_type_num, alpha, dropout=0.):
        super(KVAttentionLayer, self).__init__()
        self.dim = dim
        self.node_type_num = 3
        self.edge_type_num = edge_type_num
        self.training_edges = list(range(1, self.edge_type_num+1)) # 1~11
        # self.inference_edges = self.training_edges
        
        # Attention
        self.W_Q = nn.Parameter(torch.Tensor(self.node_type_num+1, dim, dim)) # [4 D D], 1,2,3 are item, image, text weight
        self.W_K = nn.Parameter(torch.Tensor(self.node_type_num+1, dim, dim)) # [4 D D], 1,2,3 are item, image, text weight
        self.W_V = nn.Parameter(torch.Tensor(self.node_type_num+1, dim, dim)) # [4 D D], 1,2,3 are item, image, text weight

        self.fc = nn.Linear(self.dim, self.dim)

        self.amplifier_rel_ratio = nn.Parameter(torch.Tensor(len(self.training_edges), 1))

        # Feed-Forward
        dff = self.dim * 4
        self.fc1 = nn.Linear(self.dim, dff)
        self.fc2 = nn.Linear(dff, self.dim)
        # Output
        self.fc_out = nn.Linear(self.dim, self.dim)

        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(dropout)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, hidden, adj, node_type_mask, stage=None):
        batch_size = hidden.shape[0]
        N = hidden.shape[1] # number of nodes
        selected_edges = self.training_edges

        query_hidden = torch.zeros(batch_size* N, self.dim).cuda() # [B*N D]
        key_hidden = torch.zeros(batch_size* N, self.dim).cuda() # [B*N D]
        value_hidden = torch.zeros(batch_size* N, self.dim).cuda() # [B*N D]
        node_type_mask = node_type_mask.reshape(-1) # [B*N]
        flatten_hidden = hidden.reshape(-1, hidden.size(-1)) # [B*N D]
        for n in range(1, self.node_type_num+1): # for differnt type of nodes
            n_idx = node_type_mask==n
            _query_hidden = torch.matmul(flatten_hidden, self.W_Q[n]) # [B*N D]
            query_hidden[n_idx, :] = _query_hidden[n_idx, :]
            _key_hidden = torch.matmul(flatten_hidden, self.W_K[n]) # [B*N D]
            key_hidden[n_idx, :] = _key_hidden[n_idx, :]
            _value_hidden = torch.matmul(flatten_hidden, self.W_V[n]) # [B*N D]
            value_hidden[n_idx, :] = _value_hidden[n_idx, :]
        query_hidden = query_hidden.reshape(batch_size, N, self.dim)
        key_hidden = query_hidden.reshape(batch_size, N, self.dim)
        value_hidden = query_hidden.reshape(batch_size, N, self.dim)

        
        scores = torch.matmul(query_hidden, key_hidden.transpose(-2, -1))  # [B N N]
        scores = self.leakyrelu(scores)

        alpha = -9e15 * torch.ones(batch_size, N, N).cuda() # [B L L]
        for r in selected_edges:
            # alpha = torch.where(adj.eq(r), scores*self.amplifier_rel_ratio[r-1], alpha)	# condition, if ture=x, if false=y
            alpha = torch.where(adj.eq(r), scores, alpha)	# condition, if ture=x, if false=y

        alpha = torch.softmax(alpha, dim=-1)
        attention_output = torch.matmul(alpha, value_hidden)
        return attention_output

        ## @_@: feed-forward layer
        # # attention_output = self.fc(attention_output)
        # output1 = self.dropout(attention_output)
        # output1 = self.norm1(hidden + output1)

        # feed_forward_output = self.fc2(F.relu(self.fc1(output1)))

        # output2 = self.norm2(output1 + self.dropout(feed_forward_output))
        # output_hidden = self.fc_out(output2)

        # return output_hidden
    
    
'''
Heterogenous GNN:
This implementation takes in the node features x, adjacency matrix adj, and node type mask node_type_mask. The adjacency matrix is a tensor of shape (num_nodes, num_nodes) with 1-10 values indicating the edge type between nodes. The node type mask is a tensor of shape (num_nodes,) with 1-3 values indicating the type of each node.
The implementation first splits the node features and adjacency matrix into relation-specific subgraphs based on the node type mask. It then applies dropout to the node features, and loops over the layers to perform message passing and feature aggregation on each relation-specific subgraph. It concatenates the output of each subgraph and applies the final linear layer and log softmax to obtain the class probabilities.
'''
class HeteroGNN(nn.Module):
    def __init__(self, num_node_features, num_classes, num_relations, num_layers, hidden_dim, dropout=0.5):
        super(HeteroGNN, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.num_relations = num_relations

        self.convs = nn.ModuleList()
        for i in range(num_relations):
            self.convs.append(nn.Linear(num_node_features, hidden_dim))

        self.sage_convs = nn.ModuleList()
        for i in range(num_relations):
            self.sage_convs.append(nn.Linear(hidden_dim, hidden_dim))

        self.lin1 = nn.Linear(num_relations * hidden_dim, num_classes)

    def forward(self, x, adj, node_type_mask):
        x_list = [x[node_type_mask == i] for i in range(1, node_type_mask.max()+1)]
        adj_list = [adj[node_type_mask == i, :][:, node_type_mask == j] for i in range(1, node_type_mask.max()+1) for j in range(1, node_type_mask.max()+1)]

        x_list = [F.dropout(x, p=self.dropout, training=self.training) for x in x_list]

        for i in range(self.num_relations):
            x = F.relu(self.convs[i](x_list[i]))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.spmm(adj_list[i], x)
            x = F.relu(self.sage_convs[i](x))
            x_list[i] = x

        x = torch.cat(x_list, dim=-1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)

        return F.log_softmax(x, dim=-1)



'''
Heterogenous GAT:
In this implementation, the HeteroGAT takes in node features x, edge indices edge_index, and edge types edge_type as input. It performs message passing and node feature aggregation on each relation-specific subgraph using GATConv layers, and concatenates the output of each subgraph for final classification.
The num_node_features parameter specifies the number of node features, num_classes specifies the number of classes, num_relations specifies the number of relation-specific subgraphs, num_layers specifies the number of GATConv layers used for message passing and feature aggregation, and hidden_dim specifies the hidden dimension for the GATConv layers.
The HeteroGAT uses dropout for regularization, and elu activation function for nonlinearity. At each relation-specific subgraph, it applies dropout to the node features, performs message passing and aggregation using the GATConv layers, and applies dropout again. It concatenates the outputs of all relation-specific subgraphs and applies a linear layer for classification.
'''
from torch_geometric.nn import GATConv

class HeteroGAT(nn.Module):
    def __init__(self, num_node_features, num_classes, num_relations, num_layers, hidden_dim, dropout=0.5):
        super(HeteroGAT, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.num_relations = num_relations

        self.convs = nn.ModuleList()
        for i in range(num_relations):
            self.convs.append(GATConv(num_node_features, hidden_dim))

        self.lin1 = nn.Linear(num_relations * hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_type):
        x_list, edge_index_list = [], []

        for i in range(self.num_relations):
            x_list.append(x[edge_type == i])
            edge_index_list.append(edge_index[:, edge_type == i])

        x_list = [F.dropout(x, p=self.dropout, training=self.training) for x in x_list]

        for i in range(self.num_layers):
            x_list = [F.elu(conv(x_list[j], edge_index_list[j])) for j, conv in enumerate(self.convs)]
            x_list = [F.dropout(x, p=self.dropout, training=self.training) for x in x_list]

        x = torch.cat(x_list, dim=-1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)

        return F.log_softmax(x, dim=-1)



class GCNLayer(nn.Module):#图卷积网络
    """ one layer of GCN """
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, h, adj, node_type_mask=None):
        adj = torch.clamp(adj, max=1)# 将邻接矩阵的值限制在0到1之间
        if self.dropout:# 如果设置了dropout，则对输入h进行dropout操作
            h = self.dropout(h)
        x = h @ self.W
        x = adj @ x
        if self.b is not None:
            x = x + self.b
        if self.activation:
            x = self.activation(x)
        return x


class SAGELayer(nn.Module):#一种基于图卷积网络（Graph Convolutional Network, GCN）的节点嵌入方法，用于学习图中节点的低维表示
    """ one layer of GraphSAGE with gcn aggregator """
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=False):
        super(SAGELayer, self).__init__()
        self.linear_neigh = nn.Linear(input_dim, output_dim, bias=False)
        # self.linear_self = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, h, adj, node_type_mask=None):
        adj = torch.clamp(adj, max=1)
        if self.dropout:
            h = self.dropout(h)
        x = adj @ h
        x = self.linear_neigh(x)
        # x_neigh = self.linear_neigh(x)
        # x_self = self.linear_self(h)
        # x = x_neigh + x_self
        if self.activation:
            x = self.activation(x)
        # x = F.normalize(x, dim=1, p=2)
        return x


class GATLayer(nn.Module):#图注意力网络
    """ one layer of GAT """
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(GATLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        self.n_heads = n_heads
        self.attn_l = nn.Linear(output_dim, self.n_heads, bias=False)
        self.attn_r = nn.Linear(output_dim, self.n_heads, bias=False)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, h, adj, node_type_mask=None): # TODO: this is not batched model
        adj = torch.clamp(adj, max=1)
        if self.dropout:
            h = self.dropout(h)
        x = h @ self.W # torch.Size([2708, 128])
        # calculate attentions, both el and er are n_nodes by n_heads
        el = self.attn_l(x)
        er = self.attn_r(x) # torch.Size([2708, 8])
        if isinstance(adj, torch.sparse.FloatTensor):
            nz_indices = adj._indices()
        else:
            nz_indices = adj.nonzero().T
        attn = el[nz_indices[0]] + er[nz_indices[1]] # torch.Size([13264, 8])
        attn = F.leaky_relu(attn, negative_slope=0.2).squeeze()
        # reconstruct adj with attentions, exp for softmax next
        attn = torch.exp(attn) # torch.Size([13264, 8]) NOTE: torch.Size([13264]) when n_heads=1
        if self.n_heads == 1:
            adj_attn = torch.zeros(size=(adj.size(0), adj.size(1)), device=adj.device)
            adj_attn.index_put_((nz_indices[0], nz_indices[1]), attn)
        else:
            adj_attn = torch.zeros(size=(adj.size(0), adj.size(1), self.n_heads), device=adj.device)
            adj_attn.index_put_((nz_indices[0], nz_indices[1]), attn) # torch.Size([2708, 2708, 8])
            adj_attn.transpose_(1, 2) # torch.Size([2708, 8, 2708])
        # edge softmax (only softmax with non-zero entries)
        adj_attn = F.normalize(adj_attn, p=1, dim=-1)
        # message passing
        x = adj_attn @ x # torch.Size([2708, 8, 128])
        if self.b is not None:
            x = x + self.b
        if self.activation:
            x = self.activation(x)
        if self.n_heads > 1:
            x = x.flatten(start_dim=1)
        return x # torch.Size([2708, 1024])
    
    
