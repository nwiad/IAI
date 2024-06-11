import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding, embedding_dim, filter_sizes: list, filter_num, output_dim, dropout=0.5):
        """
        vocab_szie: 词向量总数/词表长度
        embedding: 训练好的词向量
        embedding_dim: 词向量长度，也是每个卷积核的宽度
        filter_sizes: 卷积核的长度的列表
        filter_num: 每个卷积核的数量
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
        # 设置卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=filter_num, kernel_size=(size, embedding_dim)) 
            for size in filter_sizes
        ])
        # 设置全连接层
        self.fc = nn.Linear(len(filter_sizes) * filter_num, output_dim)
        # 设置随机丢弃
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): # x的尺寸为[batch_size, sentence_len]
        embedded = self.embedding(x) # embedded 的尺寸为 [batch_size, sentence_len, embbeddng_dim]
        embedded = embedded.unsqueeze(1) # embedded 的尺寸为 [batch_size, 1, sentence_len, embbedding_dim]，从第1维开始升1维以满足conv2d的输入要求

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs] # conved[n] 的尺寸为 [batch_size, filter_num, sentence_len - filter_sizes[n] + 1]，压缩了卷积结果的宽度所在的维度
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] # pooled[n] 的尺寸为 [batch_size, filter_num]，压缩了最大池化的结果的长度，由于使用默认参数，结果只有一个
        
        cat = self.dropout(torch.cat(pooled, dim=1)) # cat = [batch_size, filter_num * len(filter_sizes)] 每一个句子的结果拼接起来
        
        return F.log_softmax(self.fc(cat), dim=1) # self.fc(cat) 的尺寸为 [batch_size, output_dim]
        # F.log_softmax(self.fc(cat)) = [batch_size, output_dim] 默认对最后一维操作，等价于F.log_softmax(self.fc(cat), dim=1)
