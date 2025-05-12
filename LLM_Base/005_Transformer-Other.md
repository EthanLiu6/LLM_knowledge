<h1 align="center"> <p>005_Transformer-Other</p></h1>

> 该篇主要讲解在Transformer架构基础上的更多优化，包括软硬件优化
>
> 学习每个小知识点的时候一定要站在全局视角去看看他处在什么位置，要解决什么问题

想想Transofrmer有哪些都需要优化呢？

> Since being introduced seven years ago（不过现在25年是第八年了）, many modifications to the Transformer architecture have been proposed. However, relatively few of them generalize well across domains and scales and have seen widespread adoption (Narang et al., 2021) Some notable successful ones include Transformer-XL (Dai et al., 2019) and Rotary Position Encoding (Su et al., 2024) for improving long-context handling and position encoding, GLU MLP (Shazeer, 2020) and Sparse Mixture-of-Experts (MoE) MLP (Lepikhin et al., 2020; Fedus et al., 2022) for more expressive or efficient MLP nonlinearty and architecture, UL2 (Tay et al., 2022) and GLM (Du et al., 2021) for better training objectives. Among these, RoPE and SwiGLU MLP have been adopted by recent well-known foundation models such as Palm (Chowdhery et al., 2023) and LLaMA (Touvron et al., 2023), and are also used as our strong baseline (Transformer++).（还有MoE）



## 1. KV-cache

在生成式Transformer中，缓存(Caching) Key(K)和 Value(V)状态的技术已经存在一段时间了。这种技术可以显著提高推理速度，回想一下注意力机制，Key和Value状态用于计算带缩放的点积注意力机制(scaled dot-product attention)。

KV Cache发生在多个tokens生成步骤中，只在Decoder中进行（即在仅解码器的模型如GPT、LLama、DeepSeek等中，或者在编码器-解码器模型如T5中的解码器部分）。像BERT这样的模型不是生成模型，因此没有KV Cache。（专门有一章节进行整理LLM层面的基于Transformer架构的大致分枝）

解码器以自回归(auto-regressive)的方式工作，就像下图GPT-2文本生成示例所示的那样。

![figure1](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f76322f726573697a653a6669743a313130302f666f726d61743a776562702f302a7365784f3661644768614b72376148302e676966.gif)

> *在Decoder的自回归生成中，给定一个输入，模型会预测下一个token，然后在下一步中使用组合的输入进行下一个预测。)*

这种自回归行为会重复(repeats)一些操作，我们可以通过放大(zoom in) Decoder 中计算的带掩码的缩放点积注意力(masked scaled dot-product attention)来更好地理解这一点。

![figure2](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/kv-cache-gif1.gif)

> *(Decoder中缩放点积注意力的逐步可视化。emb_size表示embedding size.)*

根据动图可以直观发现其中存在大量的冗余计算，每生成一个token需重新计算所有历史token的Key/Value，复杂度为 $O(n^2)$ ，显存和计算时间随序列长度急剧增长，比如：

- 生成embedding有冗余计算。（动图里面没展示）
- KV生成有冗余计算。
- $QK^T$有冗余计算。
- softmax操作以及与V相乘有冗余计算。

~~这种优化为什么重要呢？如上图所示，使用KV缓存得到的矩阵要小得多，这导致矩阵乘法更快。唯一的缺点是它需要更多的GPU VRAM（或者如果没有使用GPU，则需要更多的CPU RAM）来缓存键(Key)和值(Value)的状态。~~

**思考：**

- 如果引入缓存，在哪块做缓存呢？如何做呢？

### 1.1 Splitwise

在"Splitwise: Efficient generative llm inference using phase splitting."论文中，有相关的一些分析和优化，我们先来看看这篇paper。

在论文摘要部分就指出，LLM在推理过程中往往包含两个阶段：

> 一个计算密集型的prompt计算阶段和一个内存密集型的token生成阶段，每个阶段都有不同的延迟。

![image-20250508162535149](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250508162535149.png)

论文中的直观图片：

![image-20250508162933736](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250508162933736.png)

他们所做的工作：

> **我们的工作。** 鉴于生成大语言模型的推理请求的计算阶段清晰，我们建议将提示计算和词符生成阶段分离到不同的机器上。 这提高了硬件利用率，从而提高了系统的整体效率。 它还允许为每个阶段使用不同的、配置更好的硬件。为了实现这样的设置，来自提示计算的缓存上下文必须以低延迟从提示机传送到词符机。 我们使用可用带宽以优化的方式实现这些传输，这使我们能够提高效率，而不会造成任何明显的性能损失。

其实可以看得见，图片里面已经展示了KV cache，下面我们主要围绕KV cache讲解，这篇论文暂时先看到这里就可以了，这里主要是引出一下宏观认识。最后，我引用一下柳浩老师的博客内容，写的真的很好

对于未使用KV cache的情况：

> 我们对两个阶段的特点进行深入分析。
>
> 1. prompt phase（预填充阶段），也有叫启动阶段（initiation phase），其特点如下：
>
> - 时机：发生在计算第一个输出 token 过程中。
> - 输入：输入一个prompt序列。
> - 作用：一次性处理所有的用户输入。LLMs对输入序列（即输入提示）的上下文进行总结，并生成一个新标记作为解码阶段的初始输入。
> - 执行次数：其通过一次 Forward 就可以完成。（**笔者觉得柳老师吧这点提出来真的很清晰了**）
> - 计算类型：存在大量 GEMM (GEneral Matrix-Matrix multiply) 操作，属于 Compute-bound 类型（计算密集型）计算。
> - 并行：输入的Tokens之间以并行方式执行运算，是一种高度并行化的矩阵操作，具备比较高的执行效率。
>
> > 想想，在训练阶段是不是类似这种情况？
>
> 2. token-generation phase的特点如下：
>
> - 时机：在prompt阶段生成第一个 Token之后，开始进入token-generation phase阶段。发生在计算第二个输出 token 至最后一个 token 过程中。
> - 输入：新生成的token会与输入tokens（这里包括用户输入的prompt） 拼接在一起，作为下一次推理的输入。
> - 作用：新生成的token被反馈回解码阶段作为输入（应该就是新的token对应的embedding），从而创建了一个用于token生成的自回归过程。
> - 执行次数：假设输出总共有 N 个 Token，则 token-generation phase阶段需要执行 N-1 次 Forward。
> - 计算类型：存在大量访存操作，属于 Memory-bound 类型（内存密集型）计算。
> - 并行：假设输出总共有 N 个 Token，则 Decoding 阶段需要执行 N-1 次 Forward，这 N-1 次 Forward 只能串行执行，因此效率相对比较低。另外，在生成过程中，需要关注的 Token 越来越多（每个 Token 的生成都需要 Attention 之前的 Token），计算量也会适当增大。
>
> 自回归的生成模式是两阶段的根本原因，两阶段是自回归的生成模式的外在体现形式，KV cache是优化手段。
>
> 注：在SplitWise论文中，分别把这两个阶段称为prompt phase 和 token-generation phase。在实践中，“预填充（pre-fill）”和“初始化（initiation）”这两个术语可以互换。为了更好的说明，现在我们将更倾向于使用前者。

对于使用KV cache的情况：

> 

### 1.2



### 1.n KV cache稀疏化

> https://zhuanlan.zhihu.com/p/704710823

## 2. MHA的优化

回想一下MHA的流程

![img](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/1850883-20250308125852651-1423435728.jpg)



### 2.1 MQA

- 背景：
        MQA（Multi Query Attention）最早是出现在2019年谷歌的一篇论文 [《Fast Transformer Decoding: One Write-Head is All You Need》](https://arxiv.org/pdf/1911.02150)，之所以没有被关注到，是因为文本生成类任务还没这么火热，解码序列长度也没有现阶段大模型的要求那么高。

- 核心思想：
        MQA 让所有的头之间 共享 同一份 Key 和 Value 矩阵，每个头只单独保留了一份 Query 参数，从而大大减少 Key 和 Value 矩阵的参数量。

    > Multi-query attention is identical except that the different heads share a single set of keys and values.

    ![image-20250507113148432](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250507113148432.png)

- 直接看图

    ![image-20250507113448111](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250507113448111.png)

**思考：**

- 换做是你，你会对MHA（多头注意力）做怎样的优化呢？
- 对于MHA系列（各种优化）**多个头直接拼接的操作， 相当于默认了每个头或者说每个子空间的重要性是一样的， 在每个子空间里面学习到的相似性的重要度是一样的，即这些头的权重是一样的**。然而，各个头的权重事实上肯定不同，如何有机融合？或者说，如何调整不同头之间的权重比例？

### 2.2 Uptraining

> converting the checkpoint and uptraining

- 引入

    (**uptraining** 是指对已有的模型进行进一步的训练(pre-train)或微调(fine-tune)。它可以是为了适应新的任务或结构，或者改进模型的性能。在这里， **uptraining** 是指将具有多头注意力的语言模型转换为具有多查询注意力的模型，并通过额外的预训练阶段来适应新的结构。)

    也就是说，**uptraining** 其实是为了优化MQA的（MQA有啥问题？）

- 概念
    在 Multi-Query Attention 方法中只会保留一个单独的key-value头，这样虽然可以提升推理的速度，但是会带来精度上的损失。《Multi-Head Attention:Collaborate Instead of Concatenate 》这篇论文的第一个思路是基于多个 MQA 的 checkpoint 进行 finetuning，来得到了一个质量更高的 MQA 模型。这个过程也被称为 Uptraining。

- 从多头模型生成多查询模型分为两个步骤：

    - 首先是转换检查点(checkpoint)，将多头检查点转换为多查询检查点。key和value头的投影矩阵被平均汇总为单个投影矩阵，我们发现这比选择单个键和值头或从头开始随机初始化新的键和值头效果更好。
    - 转换后的检查点接着使用相同的预训练方法进行预训练，但仅进行原始训练步骤的一小部分α。

- 图示

    ![figure21](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/gqa-figure1.jpg)



vanilla Transformer中，对于不同的注意力采取的整合方式是直接拼接。论文"Multi-Head Attention: Collaborate Instead of Concatenate“提出了其它整合方式。该论文发现所有注意力头之间捕捉的信息肯定是存在冗余的，头与头之间存在较多的通用信息。拼接后的 $𝑊_𝑄$$𝑊_𝐾^T$ 只需要大概1/3的维度就足够捕捉绝大部分的信息了。因此论文作者设计了一个混合向量来提取注意力头之间的通用信息。这个向量可以通过跟模型一起学习得到，然后应用到原始的多头注意力计算中。这种方案可以让注意力头的表示方式更加灵活，注意力头的维度可以根据实际情况进行改变。也让参数计算更加高效。

![image-20250507153800973](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250507153800973.png)

### 2.3 GQA（大模型神器）

从MHA到MQA，速度的确是提高了，但是质量很大可能是降低了的，后来Google在2023年又发表了相关的一篇论文[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245)

- 论文摘要（翻译后）

    > 多查询注意力（MQA）仅使用单个键值头，极大地加快了解码器推理速度。 然而，MQA可能会导致质量下降，而且仅仅为了更快的推理而训练一个单独的模型可能并不理想。 我们（1）提出了一种方法，使用原始预训练计算量的5%将现有的多头语言模型检查点升级为具有MQA的模型；（2）引入了分组查询注意力（GQA），它是多查询注意力的泛化，它使用中间数量（多于一个，少于查询头的数量）的键值头。 **我们表明，经过训练的 GQA 达到了接近多头注意力的质量，并且速度与 MQA 相当。**

- 核心结构

    GQA（Grouped Query Attention）将查询头分成G个组，每个组共享一个键头和值头。GQA-G表示具有G个组的分组查询。GQA-1表示单个组，因此具有单个键头和值头，等效于MQA。而GQA-H表示组数等于头数，等效于MHA。下图显示了分组查询注意力和多头/多查询注意力的比较。在将多头检查点转换为GQA检查点时，我们通过对该组内所有原始头进行平均汇总来构建每个组的键头和值头。

    ![image-20250507114518347](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250507114518347.png)

大型模型的MHA会将单个键和值头复制到模型分区的数量，MQA代表了内存带宽和容量的更大幅度的削减，而GQA 使我们能够随着模型大小的增加保持带宽和容量的相同比例下降，可以为较大的模型提供特别好的权衡。GQA 消除了这种分片带来的浪费。因此，我们预计 GQA 将为较大的模型提供特别好的权衡。

![img](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/1850883-20250413203221768-1617176553.jpg)



GQA在推理阶段可以显著降低KV Cache的大小，为更大的 Batch Size 提供了空间，可以进一步提升吞吐。

**思考：**

- 虽然最新的模型基本都在预训练阶段默认采用 GQA，我们也可以思考下，如何将已经训练好的MHA结构的模型转换成MQA或者GQA？

### 2.4 MOHSA

Transformer模型成功的主要原因是不同 Token 之间的有效信息交换，从而使每个 Token 都能获得上下文的全局视图。然而，每个Head中的 Query 、 Key和Value 是分开的，没有重叠，当在各个Head中计算注意力时也没有信息交换。换句话说，在计算当前Head的注意力时，它没有其他Head中的信息。尽管 Token 在注意力之后会通过线性投影进行处理，但那时的信息交换仅限于每个 Token。

论文“[Improving Vision Transformers by Overlapping Heads in Multi-Head Self-Attention](https://arxiv.org/pdf/2410.14874)”就对此进行了研究。作者提出信息交换在视觉 Transformer （Vision Transformers）的注意力计算过程中可以提高性能。这可以通过将每个Head的 queries、keys和values与相邻Head的 queries、keys和values重叠来实现。为此，作者提出了一种名为MOHSA（Multi-Overlapped-Head Self-Attention/多重叠头自注意力）的方法，通过重叠Head来改进多Head自注意力（Multi-Head Self-Attention）机制，使得在计算注意力时，每个Head中的 Q、 K和 V也可以被其相邻Head的 Q、 K和 V所影响，Head间信息交流可以为视觉 Transformer 带来更好的性能。如图所示。

![image-20250507161610401](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250507161610401.png)

- 拆分头的时候就进行重叠处理，使用重叠（Soft）除而不是直接除

### 2.5 MoH & MoA

> 好吧，我的想法已经被pku（北京大学）学者在24年10月份发布啦：[MoH: Multi-Head Attention as Mixture-of-Head Attention](https://arxiv.org/pdf/2410.11842)。查阅之后发现，22年就已经有类似的MoA论文发表在了ACL上了：[Mixture of Attention Heads: Selecting Attention Heads Per Token](https://aclanthology.org/2022.emnlp-main.278.pdf)

![image-20250507162412593](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250507162412593.png)

![image-20250507163127509](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250507163127509.png)

### 2.6 DCMHA

> [Improving Transformers with Dynamically Composable Multi-Head Attention](https://arxiv.org/pdf/2405.08553)
>
> 暂时不补充了

### 2.7 MLA（DeepSeek）

> DeepSeek的单独技术章节也会讲解
>
> [论文：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/pdf/2405.04434)
>
> 2.1. Multi-Head Latent Attention: Boosting Inference Efficiency

#### 2.7.1 背景

![image-20250507193411670](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250507193411670.png)

主要也是为了解决推理时候KV cache过大的问题，以及实现更好的效果。也是在MQA和GQA（可减少kv cache）的问题上进行了自己的创新性的改进。

#### 2.7.2 基本思想

> 笔者个人觉得是借鉴了LoRA的低秩压缩思想，但该思想应用在MHA的时候还是会引出很多问题，就此，DeepSeek团队设计出了MLA。

- **先回顾一下MHA**

![image-20250507194202385](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250507194202385.png)

- **我们再看看MLA的思想**

![image-20250507202319273](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250507202319273.png)

- 训练时期对queries也进行了Low-Rank处理

#### 2.7.3 结构图对比

![image-20250507194435654](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250507194435654.png)

- 整体思想

    > 

    MLA（Multi-Head Latent Attention / 多头潜在注意力）的基本思想是将注意力输入$h_t$ 压缩成一个低维的潜在向量 $c_t^{KV}$ ，维度为$d_c$，且$d_c$远小于原始的维度（$h_n d_h$）。在需要计算注意力时，可将这个潜在向量$c_t^{KV}$映射回高维空间。因此，只需要存储潜在向量$c_t^{KV}$，就可以显著减少内存的占用。

    这个过程可以通过以下公式更正式地进行描述。其中$c_t^{KV}$表示潜在向量；$W^{DKV}$是压缩矩阵（上标 D 代表"下投影"，即降维操作），负责将 $h_t$ 的维度从（$h_n⋅d_h$）压缩到$d_c$；$W^{UK}$和 $W^{UV}$ 是上投影矩阵，负责将共享的潜在向量 $c_t^{KV}$ 映射回高维空间。只需要存储这个潜在向量 $c_t^{KV}$ ，就能获得对应不同文本特征的Key和Value，而不需要对每个文本特征都存储对应的Key和Value。

    类似地，训练过程中，我们也可以将查询向量映射到一个潜在的低维向量，然后再将其映射回原始的高维空间。而且，MLA又结合了权重吸收技术，减少了计算开销。

## 2. Flash Attention

> 主要解决Attention机制的访存优化



## 重计算



## Page Attention



## vLLM



## 10. 代码实现

### 10.1 MHA

- 哈佛：“The Annotated Transformer”中MHA代码的实现

```python
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        '''
        h: head number
        '''
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d
        self.d = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d)
        return self.linears[-1](x)

```

- llm-foundry：工业界实现方式

> [源码地址](https://github.com/mosaicml/llm-foundry/blob/9c89ab263e72fb9610f28c8ab9cde5d2205b6bff/llmfoundry/models/layers/attention.py)

```python
class MultiheadAttention(nn.Module):
    """Multi-head self attention.

    Using torch or triton attention implemetation enables user to also use
    additive bias.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_impl: str = 'triton',
        clip_qkv: Optional[float] = None,
        qk_ln: bool = False,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        low_precision_layernorm: bool = False,
        verbose: int = 0,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln

        self.d_model = d_model
        self.n_heads = n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = attn_pdrop

        self.Wqkv = nn.Linear(self.d_model, 3 * self.d_model, device=device)
        # for param init fn; enables shape based init of fused layers
        fuse_splits = (d_model, 2 * d_model)
        self.Wqkv._fused = (0, fuse_splits)  # type: ignore

        if self.qk_ln:
            layernorm_class = LPLayerNorm if low_precision_layernorm else nn.LayerNorm
            self.q_ln = layernorm_class(self.d_model, device=device)
            self.k_ln = layernorm_class(self.d_model, device=device)

        if self.attn_impl == 'flash':
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == 'triton':
            self.attn_fn = triton_flash_attn_fn
        elif self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
        else:
            raise ValueError(f'{attn_impl=} is an invalid setting.')

        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True  # type: ignore

    def forward(
        self,
        x,
        past_key_value=None,
        attn_bias=None,
        attention_mask=None,
        is_causal=True,
        needs_weights=False,
    ):
        qkv = self.Wqkv(x)

        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        query, key, value = qkv.chunk(3, dim=2)

        key_padding_mask = attention_mask

        if self.qk_ln:
            # Applying layernorm to qk
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)

        context, attn_weights, past_key_value = self.attn_fn(
            query,
            key,
            value,
            self.n_heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=needs_weights,
        )

        return self.out_proj(context), attn_weights, past_key_value

    
    
def scaled_multihead_dot_product_attention(
    query,
    key,
    value,
    n_heads,
    past_key_value=None,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
    multiquery=False,
):
    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
    kv_n_heads = 1 if multiquery else n_heads
    k = rearrange(key, 'b s (h d) -> b h d s', h=kv_n_heads)
    v = rearrange(value, 'b s (h d) -> b h s d', h=kv_n_heads)

    if past_key_value is not None:
        if len(past_key_value) != 0:
            k = torch.cat([past_key_value[0], k], dim=3)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = (k, v)

    b, _, s_q, d = q.shape
    s_k = k.size(-1)

    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)

    attn_weight = q.matmul(k) * softmax_scale

    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - s_q)
        _s_k = max(0, attn_bias.size(3) - s_k)
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
        attn_weight = attn_weight + attn_bias

    min_val = torch.finfo(q.dtype).min

    if key_padding_mask is not None:
        attn_weight = attn_weight.masked_fill(
            ~key_padding_mask.view((b, 1, 1, s_k)), min_val)

    if is_causal and (not q.size(2) == 1):
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k),
                                              min_val)

    attn_weight = torch.softmax(attn_weight, dim=-1)

    if dropout_p:
        attn_weight = torch.nn.functional.dropout(attn_weight,
                                                  p=dropout_p,
                                                  training=training,
                                                  inplace=True)

    out = attn_weight.matmul(v)
    out = rearrange(out, 'b h s d -> b s (h d)')

    if needs_weights:
        return out, attn_weight, past_key_value
    return out, None, past_key_value

```

- LLaMA：LLaMA的MHA源码实现（GQA）

> [源码地址](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

```python
class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads  # 这里使用的就是GQA
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```



### 10.2 MQA

![img](https://img2024.cnblogs.com/blog/1850883/202504/1850883-20250413202905108-376926323.jpg)

```python
# 我的版本
class MQA(nn.Module):
    def __init__(self, model_dim, head_num):
        super(MQA, self).__init__()
        self.model_dim = model_dim
        self.head_num = head_num
        self.k_dim = self.model_dim // self.head_num

        self.short_cut = nn.Linear(model_dim, model_dim)
        # multi-head query, all query go to an only same key and value
        self.q_k_v = nn.Linear(self.model_dim, self.model_dim + self.k_dim * 2)

    def forward(self, in_x: Tensor):
        ipt_shape = in_x.shape  # (batch_size, seq_len, model_dim)

        short_cut = self.short_cut(in_x)

        # multi-head query, all query go to an only same key and value
        q_k_v = (self.q_k_v(in_x))
        query_head, same_one_key, same_one_value = torch.split(q_k_v, [self.model_dim, self.k_dim, self.k_dim], dim=-1)


        query_head = query_head.reshape(-1, ipt_shape[1], self.head_num, self.k_dim).transpose(1, 2)
        same_one_key = torch.unsqueeze(same_one_key, dim=2).transpose(1, 2)
        same_one_value = torch.unsqueeze(same_one_value, dim=2).transpose(1, 2)

        # MQA
        score = query_head @ same_one_key.transpose(-1, -2) / (self.k_dim ** 2)
        # (bs, head, seq, kd) @ (bs, 1, seq, kd) = (bs, head, seq, seq)
        atten = torch.softmax(score, dim=-1) @ same_one_value

        res = atten.transpose(1, 2).contiguous().view(ipt_shape)

        return score, atten, res, short_cut

```

```python
# 别人的版本
class MultiQueryAttention(nn.Module):
    """Multi-Query self attention.

    Using torch or triton attention implemetation enables user to also use
    additive bias.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_impl: str = 'triton',
        clip_qkv: Optional[float] = None,
        qk_ln: bool = False,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        low_precision_layernorm: bool = False,
        verbose: int = 0,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.head_dim)
        self.attn_dropout_p = attn_pdrop

        # NOTE: if we ever want to make attn TensorParallel, I'm pretty sure we'll
        # want to split Wqkv into Wq and Wkv where Wq can be TensorParallel but
        # Wkv shouldn't be TensorParallel
        # - vchiley
        self.Wqkv = nn.Linear(
            d_model,
            d_model + 2 * self.head_dim,
            device=device,
        )
        # for param init fn; enables shape based init of fused layers
        fuse_splits = (d_model, d_model + self.head_dim)
        self.Wqkv._fused = (0, fuse_splits)  # type: ignore

        if self.qk_ln:
            layernorm_class = LPLayerNorm if low_precision_layernorm else nn.LayerNorm
            self.q_ln = layernorm_class(d_model, device=device)
            self.k_ln = layernorm_class(self.head_dim, device=device)

        if self.attn_impl == 'flash':
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == 'triton':
            self.attn_fn = triton_flash_attn_fn
        elif self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
        else:
            raise ValueError(f'{attn_impl=} is an invalid setting.')

        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True  # type: ignore

    def forward(
        self,
        x,
        past_key_value=None,
        attn_bias=None,
        attention_mask=None,
        is_causal=True,
        needs_weights=False,
    ):
        qkv = self.Wqkv(x)

        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        query, key, value = qkv.split(
            [self.d_model, self.head_dim, self.head_dim], dim=2)

        key_padding_mask = attention_mask

        if self.qk_ln:
            # Applying layernorm to qk
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)

        context, attn_weights, past_key_value = self.attn_fn(
            query,
            key,
            value,
            self.n_heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=needs_weights,
            multiquery=True,
        )

        return self.out_proj(context), attn_weights, past_key_value

```



### 10.3 GQA

```python
pass

```

### 10.4 MLA

```python
pass

```



### 10.5 KV cache

```python
pass

```





## 参考资料

MHA相关

> https://www.cnblogs.com/rossiXYZ/p/18759167
>
> 

KV cache相关

> Patel, Pratyush, et al. "Splitwise: Efficient generative llm inference using phase splitting." *2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)*. IEEE, 2024.
>
> https://www.cnblogs.com/rossiXYZ/p/18799503







