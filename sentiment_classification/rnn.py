import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentTextGRU(nn.Module):
    def __init__(self, vocab_size, embedding, embedding_dim, hidden_dim, layer_num, output_dim, dropout=0.5):
        """
        vocab_szie: 词向量总数/词表长度
        embedding: 训练好的词向量
        embedding_dim: 词向量长度，也是每个卷积核的宽度
        hidden_dim: 隐藏状态的维度
        layer_num: GRU的层数
        output_dim: 分类数目
        dropout: 随机丢弃的比率
        """
        super().__init__()
        # 设置全局嵌入层以供索引
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 嵌入层加入训练
        self.embedding.weight.requires_grad = True
        # 利用词向量模型进行初始化
        tensor_embedding = torch.stack([torch.from_numpy(array).float() for array in embedding])
        self.embedding.weight.data.copy_(tensor_embedding)

        self.GRU = nn.GRU(embedding_dim, hidden_dim, layer_num, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.GRU(embedded) # 等价于 self.GRU(embedded, None)
        return F.log_softmax(self.fc(output[:, -1, :]), dim=1)
