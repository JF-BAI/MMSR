import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import RGATLayer, GCNLayer, SAGELayer, GATLayer
from torch.nn import Module, Parameter
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_, xavier_normal_

from transformer import TransformerEncoder

class BPRLoss(nn.Module): #贝叶斯损失函数
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma
    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return 

class SASRec(Module):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.
    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, max_seq_length):
        super(SASRec, self).__init__()

        # load parameters info
        self.batch_size = config['batch_size']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.num_item = config['num_node'][config['dataset']]

        # define layers and loss
        self.item_embedding = nn.Embedding(self.num_item, self.hidden_size, padding_idx=0)#嵌入
        self.position_embedding = nn.Embedding(max_seq_length, self.hidden_size)#加入位置编码
        self.trm_encoder = TransformerEncoder( #加载transformer编码器，定义在transformer.py中
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)#加载归一化层
        self.dropout = nn.Dropout(self.hidden_dropout_prob)#引入dropout机制，防止过拟合
        #选择使用哪个损失函数--交叉熵损失还是贝叶斯损失，若不是两者抛出异常
        if self.loss_type == 'BPR':
            self.loss_function = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_function = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights) #初始化模型的权重
        # 使用Adam优化器，设置学习率和L2正则化系数（l2范数，防止过拟合）
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'], weight_decay=config['l2'])
        # 使用学习率调度器，每隔config['lr_dc_step']个epoch，学习率乘以config['lr_dc']
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['lr_dc_step'], gamma=config['lr_dc'])

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):# 如果模块是线性层或嵌入层，则使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):# 如果模块是LayerNorm层，则将偏置项初始化为0，权重初始化为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()# 如果模块是线性层且存在偏置项，则将偏置项初始化为0

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]
    

class GRU4Rec(Module):# 初始化一个GRU模型，该模型由多个层组成，每个层都是一个GRU单元
    def __init__(self, config):
        super(GRU4Rec, self).__init__()
        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hiddenSize']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']

        # define layers and loss
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):# 如果传入的模块是嵌入层（Embedding）使用Xavier正态分布初始化嵌入层的权重
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):# 如果传入的模块是GRU（门控循环单元）
            xavier_uniform_(module.weight_hh_l0)# 使用Xavier均匀分布初始化GRU的隐藏状态权重
            xavier_uniform_(module.weight_ih_l0) # 使用Xavier均匀分布初始化GRU的输入权重

    def forward(self, item_seq_emb):
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output) # [B L D]
        
        return gru_output
    

