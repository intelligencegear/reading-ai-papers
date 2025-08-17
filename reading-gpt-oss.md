# Reading GPT OSS

## 模型介绍

GPT OSS 是一个基于 Transformer 架构的大型语言模型，采用了混合专家 (MoE) 架构。

## 模型架构

### 1. 整体架构

模型采用标准的 Transformer 解码器架构，结合混合专家层：

- **模型大小**：项目发布了两种规模的模型：
  - `gpt-oss-120b`：117B 总参数，每次前向传播激活 5.1B 参数。
  - `gpt-oss-20b`：21B 总参数，每次前向传播激活 3.6B 参数。

- **Transformer 模块 (`TransformerBlock`)**：每个 Transformer 块由两个核心部分组成：
  1. **注意力模块 (`AttentionBlock`)**
  2. **混合专家模块 (`MLPBlock`)**

### 2. 模型向前传播流程

```
Token IDs → Embedding → [N × TransformerBlock] → LayerNorm → Unembedding → Logits

TransformerBlock:
├── AttentionBlock: RMSNorm → QKV → RoPE → SDPA → Output → Residual
└── MLPBlock(MoE): RMSNorm → Gating → Expert Selection → Computation → Fusion → Residual
```

```python
def forward_flow():
    """
    1. Token Embedding: [seq_len] -> [seq_len, hidden_size]
    2. For each layer:
       a. Attention Block:
          - RMSNorm
          - QKV projection (GQA)
          - RoPE position encoding
          - SDPA with sliding window & sinks
          - Output projection
          - Residual connection
       b. MLP Block (MoE):
          - RMSNorm  
          - Expert gating & selection
          - Sparse expert computation
          - Expert result fusion
          - Residual connection
    3. Final RMSNorm
    4. Unembedding: [seq_len, hidden_size] -> [seq_len, vocab_size]
    """
    pass
```

## 关键组件

### 1. 注意力机制 (`attention.py`)

模型采用了高度优化的注意力机制：

- **分组查询注意力 (GQA)**：为了减少内存带宽和计算量，模型使用了 GQA，其中键（Key）和值（Value）头的数量少于查询（Query）头。
- **滑动窗口注意力**：为了处理长序列，部分层采用了滑动窗口注意力，限制了每个 token 的注意力范围，从而提高了效率。
- **Learned Sinks**：引入了“注意力池（sinks）”的概念，允许模型在注意力计算中学习并保留关键的上下文信息，即使这些信息超出了滑动窗口的范围。
- **Rotary Position Embedding (RoPE)**：使用 RoPE 进行位置编码，并结合 YaRN (Yet another RoPE extensioN) 技术来更好地处理长序列。
- **Triton 优化**：`triton/attention.py` 中的实现利用了 Triton 语言，通过核函数融合（fused kernels）来最大化 GPU 计算效率，减少了内存读写开销。

### 2. 混合专家模型 (MoE) (`moe.py`)

MoE 是该模型的核心创新之一，替代了传统 Transformer 中的密集前馈网络。

- **专家层 (`MLPBlock`)**：每个 MoE 模块包含大量的“专家”（例如 128 个），每个专家都是一个独立的前馈网络。
- **路由器 (`gate`)**：一个门控网络（gating network）或称为路由器，根据输入 token 动态地选择一小部分专家（例如 4 个）来处理该 token。
- **Top-K 路由**：路由器计算每个专家的权重，并选择得分最高的 K 个专家。
- **MXFP4 量化**：MoE 层的权重采用了 MXFP4 格式进行原生量化存储，这是一种 4 位浮点格式，极大地减小了模型的体积，使得 `gpt-oss-120b` 这样的大模型可以部署在单张 H100 GPU 上。
- **SwiGLU 激活函数**：专家网络内部使用 SwiGLU 激活函数，这在现代语言模型中已被证明非常有效。

### 3. 模型实现 (`model.py`)

`torch/model.py` 和 `triton/model.py` 中包含了模型的整体组装逻辑。

- **`ModelConfig`**：一个数据类，用于定义模型的所有超参数，如层数、头数、专家数等。
- **`Transformer` 类**：将嵌入层、多个 `TransformerBlock` 和最终的输出层连接在一起，构成了完整的模型。
- **权重加载**：提供了从 Hugging Face 下载的 checkpoint 加载权重的逻辑，并支持张量并行（Tensor Parallelism）以在多个 GPU 上运行。

## 代码阅读 `torch/model.py`
### 1. ModelConfig
模型结构超参数，定义了模型的层数、头数、专家数等。

```python
@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
```


### 2. RMSNorm
RMSNorm

```python
class RMSNorm(torch.nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)
```

RMSNorm，LayerNorm 的改进。

$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$$
$$\hat{x} = \frac{x}{\text{RMS}(x)}$$
$$y = \hat{x} \odot \gamma$$

### 3. SwiGLU
SwiGLU结合了Swish激活函数和门控线性单元(GLU):

```python
def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)
```

- 双重控制机制: GLU门控 + 线性变换
- 数值稳定性: 通过clamp防止梯度爆炸
- 偏置增强: x_linear + 1提供额外的表达能力

$$\text{SwiGLU}(x) = \text{Swish}(x) \odot x_{linear}$$

### 4. RoPE & YaRN
```python
def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 添加维度以支持广播
    cos = cos.unsqueeze(-2).to(x.dtype) # [seq_len, 1, head_dim//2]
    sin = sin.unsqueeze(-2).to(x.dtype) # [seq_len, 1, head_dim//2]
    
    # 将向量分成两部分进行旋转
    x1, x2 = torch.chunk(x, 2, dim=-1) # 沿最后一维分割成两半
    
    # 2D旋转变换公式     
    # [x1']   [cos  -sin] [x1]     
    # [x2'] = [sin   cos] [x2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)  # 重新拼接

class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            # YaRN优化：改善长序列外推性能
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            # NTK (Neural Tangent Kernel) 混合策略
            d_half = self.head_dim / 2
            # NTK by parts
            # 计算低频和高频的分界点
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1
            
            # 混合插值和外推策略
            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq
            
            # 软过渡掩码
            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            
            # 平滑混合两种策略
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        # 时间步序列
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        # 频率-时间矩阵
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [seq_len, head_dim//2]
        
        # 生成sin/cos值
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]
        # 计算位置编码
        cos, sin = self._compute_cos_sin(num_tokens)

        # 对Query应用旋转编码
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        # 对Key应用相同的旋转编码
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key
```
- RoPE通过旋转变换将位置信息嵌入到注意力计算中，YaRN通过 NTK-by-parts 策略解决长序列外推问题：
- 频率分层处理: 不同频率分量采用不同的扩展策略
- 平滑过渡: 通过掩码实现插值和外推的平滑混合
- 动态浓度调整: 根据缩放因子动态调整注意力浓度

### 5. Scaled Dot-Product Attention, SDPA

```python
def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    """
    实现带滑动窗口和额外分数的缩放点积注意力机制
    
    参数:
        Q: Query矩阵 [n_tokens, n_heads, q_mult, d_head]
        K: Key矩阵   [n_tokens, n_heads, d_head]  
        V: Value矩阵 [n_tokens, n_heads, d_head]
        S: 额外分数  [n_heads, q_mult, 1, 1] - 用于bias或特殊处理
        sm_scale: 缩放因子，通常是 1/sqrt(d_head)
        sliding_window: 滑动窗口大小，0表示无窗口限制
    
    返回:
        attn: 注意力输出 [n_tokens, n_heads * q_mult * d_head]
    """
    
    # 获取输入维度信息
    n_tokens, n_heads, q_mult, d_head = Q.shape
    
    # 验证K和V的维度匹配
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    
    # 扩展K和V以匹配Q的q_mult维度
    # K: [n_tokens, n_heads, d_head] -> [n_tokens, n_heads, q_mult, d_head]
    # V: [n_tokens, n_heads, d_head] -> [n_tokens, n_heads, q_mult, d_head]
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)  # 在第3维添加并扩展
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)  # 同样操作V
    
    # 扩展额外分数S以匹配attention矩阵维度
    # S: [n_heads, q_mult, 1, 1] -> [n_heads, q_mult, n_tokens, 1]
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    
    # 构建因果掩码（上三角矩阵，防止看到未来信息）
    # 创建 [n_tokens, n_tokens] 的上三角掩码，对角线以上为 -inf
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    
    # 可选：添加滑动窗口掩码
    if sliding_window > 0:
        # 创建下三角掩码，限制只能看到sliding_window范围内的历史
        # diagonal=-sliding_window 表示对角线下方sliding_window行开始遮蔽
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), 
            diagonal=-sliding_window
        )
    
    # 计算Query-Key点积注意力分数
    # einsum: "qhmd,khmd->hmqk" 
    # Q[q,h,m,d] × K[k,h,m,d] -> QK[h,m,q,k]
    # 其中 q,k 是token位置，h是head，m是q_mult，d是特征维度
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    
    # 应用缩放因子（防止softmax梯度消失）
    QK *= sm_scale
    
    # 应用因果掩码和滑动窗口掩码
    # mask维度扩展: [n_tokens, n_tokens] -> [1, 1, n_tokens, n_tokens]
    # 以便广播到 [n_heads, q_mult, n_tokens, n_tokens]
    QK += mask[None, None, :, :]
    
    # 将额外分数S拼接到注意力分数后
    # QK: [n_heads, q_mult, n_tokens, n_tokens]
    # S:  [n_heads, q_mult, n_tokens, 1]
    # 拼接后: [n_heads, q_mult, n_tokens, n_tokens+1]
    QK = torch.cat([QK, S], dim=-1)
    
    # 应用softmax计算注意力权重
    # 在最后一个维度上进行softmax归一化
    W = torch.softmax(QK, dim=-1)
    
    # 移除额外分数对应的权重（只保留对token的注意力权重）
    # W: [n_heads, q_mult, n_tokens, n_tokens+1] -> [n_heads, q_mult, n_tokens, n_tokens]
    W = W[..., :-1]
    
    # 计算加权的Value输出
    # einsum: "hmqk,khmd->qhmd"
    # W[h,m,q,k] × V[k,h,m,d] -> attn[q,h,m,d]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    
    # 重塑输出维度为二维
    # [n_tokens, n_heads, q_mult, d_head] -> [n_tokens, n_heads*q_mult*d_head]
    return attn.reshape(n_tokens, -1)
```

Step 1: 输入维度适配
- 维度校验：先确认 Q、K、V 的形状合法性 ——Q 是(n_tokens, n_heads, q_mult, d_head)（n_tokens = 序列长度，n_heads = 键值头数量，q_mult = 每个键值头对应的查询头数量，d_head = 单个注意力头维度），K/V 是(n_tokens, n_heads, d_head)（键值头数量少于查询头）。
- 维度扩展：通过K[:, :, None, :].expand(...)和V[:, :, None, :].expand(...)，将 K/V 的维度扩展为(n_tokens, n_heads, q_mult, d_head)，让键值对与查询的 “头数量” 匹配（每个键值头对应 q_mult 个查询头），这是分组查询注意力（GQA） 的核心操作 —— 既保留多查询头的表达能力，又减少 K/V 的存储和计算量。

Step 2: 构建注意力掩码，控制模型关注范围
- 因果掩码（上三角掩码）：
用torch.triu(..., diagonal=1)生成(n_tokens, n_tokens)的矩阵，上三角区域填充-inf。作用是禁止模型关注 “未来 token”（比如计算第 3 个 token 的注意力时，不能看第 4、5 个 token），确保自回归生成的因果性（符合语言模型 “从左到右预测” 的逻辑）。
- 滑动窗口掩码：
若sliding_window > 0，用torch.tril(..., diagonal=-sliding_window)生成下三角掩码，对超出当前 token 前后sliding_window范围的历史 token 填充-inf。作用是限制注意力仅关注最近的有限 token（比如窗口 = 128 时，第 1000 个 token 只看第 872-1000 个 token），大幅减少长序列（如 128k 长度）的计算量，避免冗余关联。

Step 3: 计算注意力分数
- QK 点积：用torch.einsum("qhmd,khmd->hmqk")计算 Q 与 K 的点积 —— 本质是矩阵乘法，结果QK的每个元素代表 “第 h 个键值头、第 m 个查询头下，第 q 个查询 token 与第 k 个键 token 的相关性分数”。
- 缩放（sm_scale）：QK *= sm_scale（sm_scale 通常是1/√d_head），避免点积结果过大导致 softmax 函数梯度消失（数值过大时 softmax 会 “过度聚焦” 少数 token，权重分布过陡）。
- 应用掩码：QK += mask[None, None, :, :]，将掩码矩阵广播到 QK 的维度，被掩码的位置分数变为-inf（softmax 后权重接近 0，即模型不关注这些 token）。

Step 4: 融入 learned sinks
- 维度调整与扩展：将可学习参数S（形状为(n_heads*q_mult,)）重塑并扩展为(n_heads, q_mult, n_tokens, 1)，与 QK 的维度匹配。
- 拼接与权重计算：QK = torch.cat([QK, S], dim=-1)，把S作为 “虚拟 token 的分数” 拼接到 QK 最后一维；通过softmax得到归一化权重W后，再剔除S对应的最后一维权重（W[..., :-1]）。
- 核心作用：S是可学习的偏置，让模型能自主学习 “是否忽略部分 token”（比如遇到噪声文本时，通过调整S降低无关 token 的权重）。

Step 5: 计算注意力输出
- 权重与 V 加权：用torch.einsum("hmqk,khmd->qhmd")将注意力权重W与值V做加权求和，得到每个查询 token 的注意力输出（融合了相关键 token 的信息）。
- 形状重塑：attn.reshape(n_tokens, -1)将输出调整为(n_tokens, 总隐藏维度)（总隐藏维度 = n_headsq_multd_head），与输入序列维度对齐，方便后续与原始输入做残差连接。

### 6. MLP MoE
```python
class MLPBlock(torch.nn.Module):
    """
    混合专家系统 (Mixture of Experts, MoE) 的MLP块实现
    
    核心思想：
    - 拥有多个专家网络，每个token只激活其中几个专家
    - 通过门控网络选择最相关的专家
    - 实现稀疏激活，提高模型容量而不增加计算量
    """
    
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        
        # MoE核心参数
        self.num_experts = config.num_experts          # 总专家数量 (如8个专家)
        self.experts_per_token = config.experts_per_token  # 每个token激活的专家数 (如top-2)
        self.swiglu_limit = config.swiglu_limit        # SwiGLU激活函数的限制值
        
        # 分布式训练支持
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Layer Normalization
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        # 门控网络：决定哪些专家被激活
        # 输入: [batch, seq_len, hidden_size] -> 输出: [batch, seq_len, num_experts]
        self.gate = torch.nn.Linear(
            config.hidden_size, 
            config.num_experts, 
            device=device, 
            dtype=torch.bfloat16
        )
        
        # 确保中间层大小能被world_size整除（分布式训练要求）
        assert config.intermediate_size % self.world_size == 0
        
        # 第一个MLP层的权重和偏置
        # 维度: [num_experts, intermediate_size*2//world_size, hidden_size]
        # intermediate_size*2 是因为SwiGLU需要两倍的中间维度
        # //world_size 是因为参数在多个设备间分片
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2 // self.world_size,
                    config.hidden_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        
        # 第一个MLP层的偏置
        # 维度: [num_experts, intermediate_size*2//world_size]
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2 // self.world_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        
        # 第二个MLP层的权重和偏置
        # 维度: [num_experts, hidden_size, intermediate_size//world_size]
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size // self.world_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        
        # 第二个MLP层的偏置
        # 维度: [num_experts, hidden_size]
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoE前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            
        返回:
            输出张量 [batch_size, seq_len, hidden_size]
        """
        
        # Layer Normalization
        t = self.norm(x)  # [batch, seq_len, hidden_size]
        
        # 门控机制：为每个token计算专家选择分数
        g = self.gate(t)  # [batch, seq_len, num_experts]
        
        # Top-K专家选择
        # 选择得分最高的 experts_per_token 个专家
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        
        # 专家权重归一化
        # 对选中的专家分数进行softmax，得到权重
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        # expert_weights: [batch, seq_len, experts_per_token]
        
        # 专家索引
        expert_indices = experts.indices  # [batch, seq_len, experts_per_token]

        # MLP第一层计算
        # 根据选中的专家索引获取对应的权重和偏置
        mlp1_weight = self.mlp1_weight[expert_indices, ...]  
        # 形状: [batch, seq_len, experts_per_token, intermediate_size*2//world_size, hidden_size]
        
        mlp1_bias = self.mlp1_bias[expert_indices, ...]
        # 形状: [batch, seq_len, experts_per_token, intermediate_size*2//world_size]
        
        # 第一层线性变换
        # einsum: "beck,bk->bec"
        # b=batch, e=experts_per_token, c=intermediate_size*2//world_size, k=hidden_size
        t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
        # 输出: [batch, seq_len, experts_per_token, intermediate_size*2//world_size]
        
        # SwiGLU激活函数
        # SwiGLU: x ↦ Swish(xW₁) ⊙ (xW₂)，需要2倍的中间维度
        t = swiglu(t, limit=self.swiglu_limit)
        # 输出: [batch, seq_len, experts_per_token, intermediate_size//world_size]

        # MLP第二层计算
        mlp2_weight = self.mlp2_weight[expert_indices, ...]
        # 形状: [batch, seq_len, experts_per_token, hidden_size, intermediate_size//world_size]
        
        mlp2_bias = self.mlp2_bias[expert_indices, ...]
        # 形状: [batch, seq_len, experts_per_token, hidden_size]
        
        # 第二层线性变换
        # einsum: "beck,bek->bec" 
        # 注意这里的维度变化：intermediate_size -> hidden_size
        t = torch.einsum("beck,bek->bec", mlp2_weight, t)
        # 输出: [batch, seq_len, experts_per_token, hidden_size]
        
        # 分布式训练：跨设备聚合结果
        if self.world_size > 1:
            # 在所有设备间对结果进行求和，因为每个设备只计算了部分中间维度
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        
        # 添加偏置
        t += mlp2_bias

        # 专家结果加权融合
        # einsum: "bec,be->bc"
        # 将多个专家的输出按权重进行加权平均
        t = torch.einsum("bec,be->bc", t, expert_weights)
        # 输出: [batch, seq_len, hidden_size]

        # 残差连接
        return x + t
```

### 7. AttentionBlock
```python
class AttentionBlock(torch.nn.Module):
    """
    高级多头注意力块，支持以下特性：
    - Grouped Query Attention (GQA) 
    - 旋转位置编码 (RoPE)
    - 滑动窗口注意力
    - Attention Sinks 机制
    """
    
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        
        # 注意力头配置
        self.head_dim = config.head_dim                    # 每个注意力头的维度 (通常64)
        self.num_attention_heads = config.num_attention_heads  # Query头数量 (如32)
        self.num_key_value_heads = config.num_key_value_heads  # Key/Value头数量 (如8, GQA架构)
        
        # 滑动窗口策略：隔层应用滑动窗口
        # 偶数层使用滑动窗口，奇数层使用全局注意力
        # 这种设计平衡了效率和性能
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        
        # Attention Sinks：特殊的可学习注意力锚点
        # 每个注意力头都有一个sink参数，用于稳定长序列的注意力模式
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )
        
        # Layer Normalization
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        # QKV投影层的维度计算
        # 由于GQA架构：Q头数 > KV头数，所以需要特殊计算
        qkv_dim = config.head_dim * (
            config.num_attention_heads +           # Q投影维度
            2 * config.num_key_value_heads        # K投影维度 + V投影维度
        )
        
        # QKV联合投影层
        # 一次性生成Query、Key、Value，提高计算效率
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )
        
        # 输出投影层
        # 将多头注意力结果投影回hidden_size维度
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,  # 输入：所有Q头的拼接
            config.hidden_size,                            # 输出：原始隐藏维度
            device=device,
            dtype=torch.bfloat16,
        )
        
        # 注意力缩放因子
        # 防止注意力分数过大导致softmax梯度消失
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        
        # 旋转位置编码 (RoPE)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,                    # 基础频率参数
            torch.float32,                        # 计算精度
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,      # 长度外推因子
            ntk_alpha=config.rope_ntk_alpha,                # NTK插值参数
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 [seq_len, hidden_size]
            
        返回:
            输出张量 [seq_len, hidden_size]
        """
        
        # Layer Normalization
        t = self.norm(x)  # [seq_len, hidden_size]
        
        # QKV投影
        qkv = self.qkv(t)  # [seq_len, qkv_dim]
        
        # 分离Q、K、V
        # Q部分：前 num_attention_heads * head_dim 维度
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        
        # K部分：中间 num_key_value_heads * head_dim 维度  
        k = qkv[
            :,
            self.num_attention_heads * self.head_dim : 
            (self.num_attention_heads + self.num_key_value_heads) * self.head_dim,
        ].contiguous()
        
        # V部分：最后 num_key_value_heads * head_dim 维度
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : 
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
        ].contiguous()

        # 重塑为多头格式 (支持GQA)
        # Q: [seq_len, num_kv_heads, q_mult, head_dim] 
        # 其中 q_mult = num_attention_heads // num_key_value_heads
        q = q.view(
            -1,                                              # seq_len
            self.num_key_value_heads,                       # KV头数作为主维度
            self.num_attention_heads // self.num_key_value_heads,  # 每个KV头对应的Q头数
            self.head_dim,
        )
        
        # K,V: [seq_len, num_kv_heads, head_dim]
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        
        # 应用旋转位置编码
        q, k = self.rope(q, k)
        
        # 执行缩放点积注意力 (包含滑动窗口和attention sinks)
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
        # 输出: [seq_len, total_hidden_size]
        
        # 输出投影
        t = self.out(t)  # [seq_len, hidden_size]
        
        # 残差连接
        t = x + t
        return t
```

### 8. Transformer
```python
class Transformer(torch.nn.Module):
    """
    完整的Transformer模型，包含：
    - Token嵌入层
    - 多层TransformerBlock
    - 最终归一化层
    - 输出投影层（unembedding）
    """
    
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        
        # Token嵌入层
        # 将token ID映射为dense向量表示
        self.embedding = torch.nn.Embedding(
            config.vocab_size,      # 词汇表大小 (如32000)
            config.hidden_size,     # 嵌入维度 (如4096)
            device=device, 
            dtype=torch.bfloat16
        )
        
        # Transformer块堆叠
        # 每一层都是一个完整的TransformerBlock (包含注意力+MLP)
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)  # 如32层
            ]
        )
        
        # 最终层归一化
        # 在输出前对特征进行最后一次归一化
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        # 输出投影层 (unembedding/lm_head)
        # 将隐藏状态投影回词汇表维度，用于预测下一个token
        self.unembedding = torch.nn.Linear(
            config.hidden_size,     # 输入：隐藏维度
            config.vocab_size,      # 输出：词汇表大小
            bias=False,             # 通常不使用偏置
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：从token ID到下一个token的概率分布
        
        参数:
            x: token ID序列 [batch_size, seq_len] (整数张量)
            
        返回:
            logits: 每个位置的词汇表概率分布 [batch_size, seq_len, vocab_size]
        """
        
        # Token嵌入
        x = self.embedding(x)  # [batch_size, seq_len, hidden_size]
        
        # 逐层通过Transformer块
        for block in self.block:
            x = block(x)  # 每个block内部包含attention + MLP + 残差连接
        
        # 最终归一化
        x = self.norm(x)  # [batch_size, seq_len, hidden_size]
        
        # 输出投影得到logits
        x = self.unembedding(x)  # [batch_size, seq_len, vocab_size]
        
        return x

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda"
    ) -> "Transformer":
        """
        从检查点加载预训练模型
        支持分布式训练中的参数分片加载
        """
        if not isinstance(device, torch.device):
            device = torch.device(device)

        # 加载配置文件
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = ModelConfig(**json_config)

        # 创建模型实例
        model = Transformer(
            config=config,
            device=device,
        )
        model.eval()  # 设置为评估模式

        # 分布式训练参数
        my_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        per_rank_intermediate_size = config.intermediate_size // world_size

        # 创建检查点加载器
        checkpoint = Checkpoint(path, device)

        # 逐参数加载权重
        for name, param in model.named_parameters():
            loaded_tensor = checkpoint.get(name)

            # MoE参数的分片处理
            # 注意：这里是在MXFP4上采样后进行分片，实际应该在之前分片提高效率·
            
            if "mlp1" in name:  # MLP第一层 (权重和偏置都需要分片)
                # 分片中间维度：每个rank负责一部分intermediate_size
                loaded_tensor = loaded_tensor[
                    :,  # 专家维度不分片
                    my_rank * 2 * per_rank_intermediate_size : 
                    (my_rank + 1) * 2 * per_rank_intermediate_size,  # *2因为SwiGLU
                    ...,
                ]
                
            elif "mlp2_weight" in name:  # MLP第二层权重 (只有权重需要分片)
                loaded_tensor = loaded_tensor[
                    ...,
                    my_rank * per_rank_intermediate_size : 
                    (my_rank + 1) * per_rank_intermediate_size,
                ]
            
            # 复制参数到模型
            try:
                param.data.copy_(loaded_tensor)
            except:
                # 调试信息：维度不匹配时输出详细信息
                print(f"{name=} {param.data.shape=} {loaded_tensor.shape=}")
                raise

        return model
```

### 9. TokenGenerator
```python
class TokenGenerator:
    """
    基于Transformer模型的自回归文本生成器
    
    核心功能：
    - 加载预训练模型
    - 支持多种采样策略（贪心、温度采样）
    - 流式生成token
    - 可选返回log概率
    - 支持停止条件控制
    """
    
    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device):
        """
        初始化生成器
        
        参数:
            checkpoint: 模型检查点路径
            device: 计算设备 (cuda/cpu)
            
        @torch.inference_mode() 装饰器:
        - 禁用梯度计算，节省内存
        - 提高推理速度
        - 确保不会意外修改模型参数
        """
        self.device = device
        
        # 从检查点加载预训练模型
        # 自动处理分布式参数分片、配置加载等
        self.model = Transformer.from_checkpoint(checkpoint, device=self.device)

    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: list[int],      # 输入提示的token序列
                 stop_tokens: list[int],        # 停止生成的token列表
                 temperature: float = 1.0,      # 采样温度参数
                 max_tokens: int = 0,           # 最大生成token数 (0=无限制)
                 return_logprobs: bool = False  # 是否返回log概率
                 ):
        """
        自回归文本生成主函数
        
        生成流程：
        1. 基于当前序列预测下一个token
        2. 将预测token添加到序列末尾
        3. 重复步骤1-2直到满足停止条件
        
        参数详解:
            prompt_tokens: 初始提示词的token ID列表，如 [1, 2, 3]
            stop_tokens: 遇到这些token时停止生成，如 [<EOS>, <PAD>]
            temperature: 
                - 0.0: 贪心解码 (总是选择概率最高的token)
                - 1.0: 标准随机采样
                - >1.0: 更随机的输出
                - <1.0: 更确定性的输出
            max_tokens: 最大生成长度限制
            return_logprobs: 是否同时返回每个token的log概率
            
        返回:
            生成器，每次yield一个新的token (和可选的logprob)
        """
        
        # 初始化token序列
        # 复制输入避免修改原始prompt_tokens
        tokens = list(prompt_tokens)  # [prompt_token_1, prompt_token_2, ...]
        num_generated_tokens = 0
        
        # 自回归生成循环
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            
            # 模型前向传播获取logits
            # 输入: 当前完整token序列
            # 输出: 每个位置对词汇表的概率分布
            logits = self.model(
                torch.as_tensor(tokens, dtype=torch.int32, device=self.device)
            )[-1]  # 只取最后一个位置的logits: [vocab_size]
            
            # 根据温度参数选择下一个token
            if temperature == 0.0:
                # 贪心解码：选择概率最高的token
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                # 温度采样：随机采样
                # 温度缩放：temperature越小越确定，越大越随机
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                # 多项式采样：按概率分布随机选择
                predicted_token = torch.multinomial(probs, num_samples=1).item()
            
            # ➕ 将新token添加到序列
            tokens.append(predicted_token)
            num_generated_tokens += 1

            # 可选：计算并返回log概率
            if return_logprobs:
                # 计算完整的log概率分布
                logprobs = torch.log_softmax(logits, dim=-1)  # [vocab_size]
                # 提取选中token的log概率
                selected_logprobs = logprobs[predicted_token].item()
                # yield (token, logprob) 对
                yield predicted_token, selected_logprobs
            else:
                # yield token
                yield predicted_token

            # 检查停止条件
            if predicted_token in stop_tokens:
                break
```

## 代码阅读 `torch/weights.py`

### 1. _get_mxfp4_tensor
_get_mxfp4_tensor 和 _get_mxfp4_tensor_copy 的目标完全相同：将 MXFP4 格式的权重反量化为 bfloat16 或其他高精度浮点格式。它们最核心的区别在于 内存效率 和 实现复杂度 。

1. _get_mxfp4_tensor (主方法)
- 内存效率高 : 这是生产版本的方法。它通过 分块处理 (chunking) 的方式来反量化张量。它不会一次性将整个巨大的权重张量加载到内存中进行计算，而是以 rows_per_chunk 为单位，一小块一小块地处理。这使得它可以用相对较少的内存加载非常大的模型权重。
- 原地操作 (In-place-like) : 它预先分配好最终输出张量 out 的全部空间，然后在循环中计算每个小块的结果，并直接填充到 out 的相应位置 ( out=sub )。这进一步减少了中间变量的内存占用。
- 实现更复杂 : 为了实现分块处理和内存优化，代码逻辑相对复杂，需要手动管理循环和索引。

2. _get_mxfp4_tensor_copy (简易版/调试版)
- 内存消耗大 : 这个方法的注释明确写着 "short version that uses a lot of memory" 。它一次性将所有 blocks 和 scales 加载到内存，并执行一系列的张量操作（ stack , view , unsqueeze ）。这些操作会产生多个完整的、巨大的中间张量副本，因此内存占用非常高。对于几十亿甚至上百亿参数的模型，这个方法很容易导致内存溢出 (OOM)。
- 实现简单直观 : 代码更短，更符合 PyTorch 的函数式编程风格，可读性更高，更容易理解反量化的核心逻辑。它很可能是开发和调试阶段使用的版本。

 _get_mxfp4_tensor 是为实际使用设计的 优化版本 ，牺牲了代码的简洁性来换取极高的内存效率。而 _get_mxfp4_tensor_copy 是一个 教学或调试版本 ，代码直观但内存开销巨大，不适用于加载大模型。
核心算法是 基于查找表 (LUT) 和指数缩放的 MXFP4 反量化算法 。

MXFP4 是一种微缩浮点格式，它将一个数字表示为两部分：
1. 尾数 (Mantissa) : 一个 4-bit 的索引，指向一个预定义的值列表（比如 [0.0, 0.5, 1.0, ...] )。
2. 指数 (Exponent) : 一个共享的 8-bit 指数，用于对一组尾数值进行缩放。
反量化公式可以简化为： FinalValue = MantissaValue * 2^Exponent 。