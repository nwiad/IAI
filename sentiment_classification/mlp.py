import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentTextMLP(nn.Module):
    def __init__(self, vocab_size, embedding, embedding_dim, hidden_dim, output_dim):
        """
        vocab_szie: 词向量总数/词表长度
        embedding: 训练好的词向量
        embedding_dim: 词向量长度，也是每个卷积核的宽度
        hidden_dim: 隐含层的维度
        output_dim: 分类数目
        """
        super().__init__()
        # 设置全局嵌入层以供索引
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 嵌入层加入训练
        self.embedding.weight.requires_grad = True
        # 利用词向量模型进行初始化
        tensor_embedding = torch.stack([torch.from_numpy(array).float() for array in embedding])
        self.embedding.weight.data.copy_(tensor_embedding)
        self.fc_1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)
        hidden = F.relu(self.fc_1(embedded))
        return F.log_softmax(self.fc_2(hidden), dim=1)