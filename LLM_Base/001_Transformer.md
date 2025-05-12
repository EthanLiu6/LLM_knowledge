<h1 align="center"> <p>001_Transformer</p></h1>

## 1. 基本架构

中英文对照论文：[Attention Is All You Need](https://yiyibooks.cn/arxiv/1706.03762v7/index.html)

<img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250425144838781.png" alt="image-20250425144838781"  />

### 1.1 编码器

编码器由N = 6 个完全相同的层堆叠而成。 每一层都有两个子层。 第一个子层是一个`multi-head self-attention`机制，第二个子层是一个简单的、位置完全连接的前馈网络(`FFN`)。 我们对每个子层再采用一个残差连接(代码使用`short_cut`或者`res_net`指代)，接着进行层标准化（代码用`Norm`指代）。也就是说，每个子层的输出是$LayerNorm(x + Sublayer(x))$，其中$Sublayer(x)$ 是由子层本身实现的函数。 为了方便这些残差连接，模型中的所有子层以及嵌入层产生的输出维度都为$d_{model}$ = 512。

### 1.2 解码器

解码器同样由N = 6 个完全相同的层堆叠而成。除了每个编码器层中的两个子层之外，**解码器还插入第三个子层**，该层对编码器堆栈的输出执行`multi-head attention`。 与编码器类似，我们在每个子层再采用残差连接，然后进行层标准化。 **我们还修改解码器堆栈中的self-attention子层，以防止位置关注到后面的位置。 这种掩码结合将输出嵌入偏移一个位置，确保对位置的预测 i 只能依赖小于i 的已知输出。**——Masking（sequence masking）

### 1.3 Pipeline

`Encoder`:

<img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250507110500398.png" alt="image-20250507110500398" style="zoom:30%;" />



`Decoder`:

根据架构图同上整理（基本类似）



## 2. 主要组件

- 残差连接（short cut）

- 注意力机制（self attention）

    Scaled Dot-Product Attention

    Multi-Head Attention

    Cross Multi-Head Attention

- 全连接层（FFN）

- 归一化层（Norm）

- Dropout

- 掩码机制（Masking）

- 位置编码（Position Embedding）

### 2.1 Scaled Dot-Product Attention

就是标准的注意力机制，需要除以$\sqrt{d_k}$

Attention可以描述为将query和一组 **key-value对** 映射到输出(output)，其中query、key、value和 output都是向量(vector)。 输出为value的加权和，其中分配给每个value的权重通过query与相应key的兼容函数来计算。

<img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250425145018652.png" alt="image-20250425145018652"  />
$$
Attention(Q, K, V)=softmax(\frac{Q K^{T}}{\sqrt{d_{k}}}) V
$$

### 2.2 Multi-Head self Attention

之前的注意力只注重单独某个向量空间，势必导致虽然最终生成的向量可以在该空间上有效将人类概念进行映射，但是无法有效反映外部丰富的世界。因此，我们需要一种可以允许模型在不同的子空间中进行信息选择的机制。元论文作者是这样说的：

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.



<img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250425150546293.png" alt="image-20250425150546293"  />
$$
MultiHead(Q, K, V) = Concat(head_{1}, \ldots, head_{h}) W^{O}
$$

$$
where\  head_{i} = Attention(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V})
$$

其中：
$$
W_{i}^{Q} \in \mathbb{R}^{d_{model} \times d_{k}}; W_{i}^{K} \in \mathbb{R}^{d_{model} \times d_{k}}; W_{i}^{V} \in \mathbb{R}^{d_{model} \times d_{v}}; W^{O} \in \mathbb{R}^{hd_{v} \times d_{model}}
$$

**思考：**

- 实际工程上如何切分头？
- 如何合并多头Attention结果？



>  NOTE：
>
> 每个头的注意力计算其实和单头注意力没啥区别，但是有一个点可以留意下，即单头计算是使用最后两个维度（seq_len, d_k），跳过前两个维度（batch_size, h）。而每个注意力头的输出形状为：(batch_size，h，seq_len，d_k）。之所以要这么处理，完全是因为计算的需要。因为Q、K和V的前两个维度（多头与batch）是等价的，本质上都是并行计算。所以计算时也可以把它们放在同一个维度上：batch_size * num_heads。也正是因为计算的需要，注意力权重 ( QK^T ) 的形状有时是三维张量 (batch_size*num_heads, tgt_seq_len, src_seq_len)，有时是四维张量 (batch_size, num_heads, tgt_seq_len, src_seq_len) ，会根据需要在二者间切换。
>
> 通常，独立计算具有非常简单的并行化过程。尽管这取决于 GPU 线程中的底层低级实现。理想情况下，我们会为每个batch 和每个头部分配一个 GPU 线程。例如，如果我们有 batch=2 和 heads=3，我们可以在 6 个不同的线程中运行计算。即使尺寸是d_k=d_model/heads。由于每个头的计算是并行进行的（不同的头拿到相同的输入，进行相同的计算），模型可以高效地处理大规模输入。相比于顺序处理的 RNN，注意力机制本身支持并行，而多头机制进一步增强了这一点。

### 2.3 Cross Multi-Head Attention

将encoder的key和value与decoder的query进行attention，论文好像没有明确指出这一块内容

<img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250425151609839.png" alt="image-20250425151609839"  />

### 2.4 FFN

#### 基本内容

Position-wise 类型的 Feed-Forward Networks

![image-20250425152721037](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250425152721037.png)

论文指出了几点，我这里height light了一下：

- 每个编码和解码层都有一个扩展的FFN子层
- 两个线性转换和一个默认的ReLU激活函数
- 说相当于用了个两个大小为1的卷积核？
- FFN层先是将输入（MHA的输出）进行进行$model_{dim}$✖️4的操作，然后再转成$model_{dim}$（两个线性转换）。

#### 注意点

- 自注意力机制模块会接到全连接网络，FFN需要的输入是一个矩阵而不是多个矩阵。而且因为有残差连接的存在，多头注意力机制的输入和输出的维度应该是一样的。

## 3. 其他组件

### 3.1 掩码机制（Masking）

mask 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果。Transformer 模型里面涉及两种 mask，分别是 padding mask 和 sequence mask。

两个的计算原理一样：xxxx（待补充）

**思考**：为什么需要添加这两种mask码呢？？？

#### padding mask

什么是 padding mask 呢？因为每个批次输入序列长度是不一样的也就是说，我们要对输入序列进行对齐。具体来说，就是给在较短的序列后面填充 0。但是如果输入的序列太长，则是截取左边的内容，把多余的直接舍弃。因为这些填充的位置，其实是没什么意义的，所以我们的attention机制不应该把注意力放在这些位置上，所以我们需要进行一些处理。

具体的做法是，把**这些位置**的值**加上一个非常大的负数(负无穷)**，这样的话，经过 softmax，这些位置的概率就会接近0！

**思考**：上句中的 "这些位置" 指哪些位置呢？

- pytorch 代码实现

```python
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # mask步骤，用 -1e9 代表负无穷
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```

#### sequence mask

![figure19](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/attention-figure19.jpg)

sequence mask 是为了使得 decoder 不能看见未来的信息。对于一个序列，在 time_step 为 t 的时刻，我们的解码输出应该只能依赖于 t 时刻之前的输出，而不能依赖 t 之后的输出。因此我们需要想一个办法，把 t 之后的信息给隐藏起来。这在训练的时候有效，因为训练的时候每次我们是将target数据完整输入进decoder中地，预测时不需要，预测的时候我们只能得到前一时刻预测出的输出。

![figure20](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/attention-figure20.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;那么具体怎么做呢？也很简单：产生一个上三角矩阵，上三角的值全为0。把这个矩阵作用在每一个序列上，就可以达到我们的目的。

**思考：**

- decoder 中需要 padding mask 吗？

### 3.2 位置编码（Position Embedding）

![image-20250425160140494](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250425160140494.png)

论文指出使用的是三角位置编码，可以想想，为啥可以实现位置编码呢？有怎样的效果呢？（后续位置编码章节会讲）

### 3.3 Other

由于别的一些组件适合一个章节讲解，有很多细节知识，单独列成模块讲解了

包括：

- 别的Attention

- 位置编码
- Norm
- Activation
- 等等

## 4. 补充

MQA

GQA

Flash Attention

重计算

KV-cache

Page Attention



## 5. 代码实现

### 5.1 MHA

```python
# 实现1
class MHA(nn.Module):
    def __init__(self, model_dim=1024, head_num=8, mask=None, dropout=0):
        super().__init__()
        self.model_dim = model_dim
        self.head_num = head_num
        self.k_dim = self.model_dim // self.head_num

        self.q_k_v_c = nn.Linear(self.model_dim, self.model_dim * 4)

    def forward(self, in_x):
        seq_len = in_x.shape[1]

        q_k_v_c = self.q_k_v_c(in_x)
        query, key, value, short_cut = torch.split(q_k_v_c, self.model_dim, dim=-1)

        # multi-head and transpose
        query_head = query.reshape(-1, seq_len, self.head_num, self.k_dim).transpose(1, 2)
        key_head = key.reshape(-1, seq_len, self.head_num, self.k_dim).transpose(1, 2)
        value_head = value.reshape(-1, seq_len, self.head_num, self.k_dim).transpose(1, 2)

        # q @ k.T / sqrt(k_dim)
        atten_score = query_head @ key_head.transpose(-1, -2) / (self.k_dim ** 2)
        atten = torch.softmax(atten_score, dim=-1) @ value_head
        res = atten.transpose(1, 2).contiguous().view(in_x.shape)

        return res, atten_score, short_cut

```



![img](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/1850883-20250308125852651-1423435728.jpg)

```python
# 实现2
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        # 因为后续要给每个头分配等量的词特征，把词嵌入拆分成h组Q/K/V，所以要确保d_model可以被h整除，保证 d_k = d_v = d_model/h
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h # 单个头的注意力维度
        self.h = h # 注意力头数量
        # 定义W^Q, W^K, W^V和W^O矩阵，即四个线性层，每个线性层都具有d_model的输入维度和d_model的输出维度，前三个线性层分别用于对Q向量、K向量、V向量进行线性变换，第四个用来融合多头结果
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None # 初始化注意力权重
        self.dropout = nn.Dropout(p=dropout) # 进行dropout操作时置0比率，默认是0.1


     def forward(self, query, key, value, mask=None):
         """
         本函数是论文中图2（多头注意力的架构图）的实现。
         - query, key, value：并非论文公式中经过W^Q, W^K, W^V计算后的Q, K, V，而是原始输入X。query, key, value的维度是(batch_size, seq_len, d_model)
         - mask：注意力机制中可能需要的mask掩码张量，默认是None
         """        
         if mask is not None:
             # 对所有h个头应用同样的mask
             # 单头注意力下，mask和X的维度都是3，即(batch_size, seq_len, d_model)，但是多头注意力机制下，会在第二个维度插入head数量，因此X的维度变成(batch_size, h,seq_len,d_model/h)，所以mask也要相应的把自己拓展成4维，这样才能和后续的注意力分数进行处理
             mask = mask.unsqueeze(1) # mask增加一个维度
         nbatches = query.size(0) # 获取batch_size

         # 1) Do all the linear projections in batch from d_model => h x d_k
         """
         1). 批量执行从 d_model 到 h x d_k 的线性投影，即计算多头注意力的Q,K,V，所以query、value和key的shape从(batch_size,seq_len,d_model)变化为(batch_size,h,seq_len,d_model/h)。
            zip(self.linears, (query, key, value)) 是把(self.linears[0],self.linears[1],self.linears[2])这三个线性层和(query, key, value)放到一起
            然后利用for循环将(query, key, value)分别传到线性层中进行遍历，每次循环操作如下：
             1.1 通过W^Q,W^K,W^V（self.linears的前三项）求出自注意力的Q,K,V，此时Q,K,V的shape为(batch_size,seq_len,d_model), 对应代码为linear(x)。
             以self.linears[0](query)为例，self.linears[0] 是一个 (512, 512) 的矩阵，query是(batch_size,seq_len,d_model)，相乘之后得到的新query还是512(d_model)维的向量。
             key和value 的运算完全相同。
             1.2 把投影输出拆分成多头，即增加一个维度，将最后一个维度变成(h,d_model/h)，投影输出的shape由(batch_size,seq_len,d_model)变为(batch_size,seq_len,h,d_model/h)。对应代码为`view(nbatches, -1, self.h, self.d_k)`，其中的-1代表自适应维度，计算机会根据这种变换自动计算这里的值。
             因此我们分别得到8个头的64维的key和64维的value。这样就意味着每个头可以获得一部分词特征组成的句子。
             1.3 交换“seq_len”和“head数”这两个维度，将head数放在前面，最终shape变为(batch_size,h,seq_len，d_model/h)。对应代码为`transpose(1, 2)`。交换的目的是方便后续矩阵乘法和不同头部的注意力计算。也是为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义与句子位置的关系，从attention函数中可以看到，利用的是原始输入的倒数第一和第二维.这样我们就得到了每个头的输入。
             多头与batch本质上都是并行计算。所以计算时把它们放在同一个维度上，在用GPU计算时，大多依据batch_size * head数来并行划分。就是多个样本并行计算，具体到某一个token上，可以理解为n个head一起并行计算。
         """          
         query, key, value = [
             lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) # 对应图上的序号2，3
             for lin, x in zip(self.linears, (query, key, value)) # 对应图上的序号1
         ]

         # 2) Apply attention on all the projected vectors in batch.
         """
         2) 在投影的向量上批量应用注意力机制，具体就是求出Q,K,V后，通过attention函数计算出Attention结果。因为head数量已经放到了第二维度，所以就是Q、K、V的每个头进行一一对应的点积。则：     
            x的shape为(batch_size,h,seq_len,d_model/h)。
            self.attn的shape为(batch_size,h,seq_len,seq_len)
         """          
         x, self.attn = attention( # 对应图上的序号4
             query, key, value, mask=mask, dropout=self.dropout
         )

         # 3) "Concat" using a view and apply a final linear.
         """
         3) 把多个头的输出拼接起来，变成和输入形状相同。
            通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量，我们需要将其转换为输入的形状以方便后续的计算，即将多个头再合并起来，进行第一步处理环节的逆操作，先对第二和第三维进行转置，将x的shape由(batch_size,h,seq_len,d_model/h)转换为 (batch_size,seq_len,d_model)。
            3.1 交换“head数”和“seq_len”这两个维度，结果为(batch_size,seq_len,h,d_model/h)，对应代码为：`x.transpose(1, 2).contiguous()`。`contiguous()`方法将变量放到一块连续的物理内存中，是深拷贝，不改变原数据，这样能够让转置后的张量应用view方法，否则将无法直接使用。
            3.2 然后将“head数”和“d_model/head数”这两个维度合并，结果为(batch_size,seq_len,d_model)，代码是view(nbatches, -1, self.h * self.d_k)。
            比如，把8个head的64维向量拼接成一个512的向量。然后再使用一个线性变换(512,512)，shape不变。因为有残差连接的存在使得输入和输出的维度至少是一样的。
            即(5, 8, 10, 64)  ==> (5, 10, 512)
         """            
         x = (
             x.transpose(1, 2) # 对应图上的序号5
             .contiguous() # 对应图上的序号6
             .view(nbatches, -1, self.h * self.d_k) # 对应图上的序号7
         )
         del query
         del key
         del value
         # 当多头注意力机制计算完成后，将会得到一个形状为[src_len,d_model]的矩阵，也就是多个z_i水平堆叠后的结果。因此会初始化一个线性层（W^O矩阵）来对这一结果进行一个线性变换得到最终结果，并且作为多头注意力的输出来返回。
         # self.linears[-1]形状是(512, 512)，因此最终输出还是(batch_size, seq_len, d_model)。
         return self.linears[-1](x) # 对应图上的序号8

```



## n. 面试问题

- 为什么需要qkv参数矩阵，不能共用同一个吗？

    每个的作用不一样，………………

- 为啥要除以$\sqrt{d_k}$

    代码演示矩阵乘法使用缩放和不使用缩放的对比。

    > 当 $d_{k}$ 的值比较小的时候，两种点积机制(additive 和 Dot-Product)的性能相差相近，当 $d_{k}$ 比较大时，additive attention 比不带scale 的点积attention性能好。 我们怀疑，对于很大的 $d_{k}$ 值，点积大幅度增长，将softmax函数推向具有极小梯度的区域。 为了抵消这种影响，我们缩小点积 $\frac{1}{\sqrt{d_{k}}}$ 倍。

- 为什么拆多头？有什么作用？

    子空间信息多样化（有点类似卷积的多个`channel`）。好像还可以减少计算量（显存占用量/计算量好像减少了$num_{head}$倍）

- self Attention如何实现多模态？

    交叉注意力实现。
    
- self attention和传统的attention的区别？

- 想想Transformer架构中存在哪些问题？

- 点积注意力与加性注意力对比？为啥使用点积注意力？