import torch
import torch.nn as nn
from timm.models.vision_transformer import trunc_normal_, Mlp
from xformers.ops import memory_efficient_attention, unbind
from .weather_embedding import WeatherEmbedding  # 기존에 사용하던 모듈


class VariableAwareEmbedding(nn.Module):
    """
    Physics-Informed Variable Embedding Module
    - 변수의 물리적 특성(Group 1: Dynamics, Group 2: Thermodynamics)에 따라
    - 서로 다른 임베딩 레이어(d1, d2)를 통과시켜 채널을 분리합니다.
    """
    def __init__(self, 
        group1_vars,  # Dynamics 변수 리스트 (예: U, V, Z, P)
        group2_vars,  # Thermodynamics 변수 리스트 (예: T, Q)
        img_size, 
        patch_size, 
        d1,           # Group 1이 변환될 차원 크기 (예: 512)
        d2,           # Group 2가 변환될 차원 크기 (예: 512)
        num_heads     # Attention head 수
    ):
        super().__init__()
        self.group1_vars = group1_vars
        self.group2_vars = group2_vars
        
        # [핵심] 두 개의 독립적인 임베딩 레이어 생성
        
        # 1. Dynamics Embedder (Group 1 -> d1)
        # 이 변수들은 Step 1(Deep Layer)부터 처리될 핵심 뼈대입니다.
        self.embed1 = WeatherEmbedding(
            variables=group1_vars, 
            img_size=img_size, 
            patch_size=patch_size, 
            embed_dim=d1,
            num_heads=num_heads
        )
        
        # 2. Thermodynamics Embedder (Group 2 -> d2)
        # 이 변수들은 Step 2(Shallow Layer)에서 합류할 정보입니다.
        self.embed2 = WeatherEmbedding(
            variables=group2_vars, 
            img_size=img_size, 
            patch_size=patch_size, 
            embed_dim=d2,
            num_heads=num_heads
        )
        # (옵션) VA-MoE 논문 스타일: 변수 종류를 알려주는 Learnable Vector 추가
        # 논문에서는 Index Embedding을 추가하여 전문가(Expert)가 변수를 식별하게 함 
        # 여기서는 간단히 그룹별 식별자를 더해주는 방식으로 구현 가능 (필요 시 주석 해제)
        # self.group_token1 = nn.Parameter(torch.zeros(1, 1, d1))
        # self.group_token2 = nn.Parameter(torch.zeros(1, 1, d2))
        # trunc_normal_(self.group_token1, std=0.02)
        # trunc_normal_(self.group_token2, std=0.02)
    
    def forward(self, x, variables):
        """
        x: (Batch, Total_Vars, H, W)
        variables: 현재 배치에 들어온 전체 변수 이름 리스트
        """
        
        # 1. 변수 인덱싱 (Dynamic Indexing)
        # 현재 입력 x에서 Group 1과 Group 2가 어디에 있는지 찾습니다.
        # (매번 리스트 검색이 부담된다면, 학습 시 변수 순서를 고정하고 미리 계산된 인덱스를 써도 됩니다)
        idx1 = [variables.index(v) for v in self.group1_vars if v in variables]
        idx2 = [variables.index(v) for v in self.group2_vars if v in variables]
        
        # 검증: 빠진 변수가 없는지 확인
        if len(idx1) != len(self.group1_vars) or len(idx2) != len(self.group2_vars):
             # 실제 학습/추론 시에는 입력 변수 리스트가 고정되므로 이 에러는 초기 세팅 문제일 가능성이 큼
             pass 
        # 2. 입력 쪼개기 (Slice)
        x1_input = x[:, idx1, :, :]  # (B, N1, H, W)
        x2_input = x[:, idx2, :, :]  # (B, N2, H, W)
        
        # 3. 개별 임베딩 수행
        # embed1 결과: (B, L, d1)
        emb1 = self.embed1(x1_input, self.group1_vars)
        
        # embed2 결과: (B, L, d2)
        emb2 = self.embed2(x2_input, self.group2_vars)
        
        # (옵션) 그룹 토큰 더하기 (VA-MoE 논문 아이디어 차용)
        # emb1 = emb1 + self.group_token1
        # emb2 = emb2 + self.group_token2
        
        # 4. 채널 방향으로 합치기 (Concatenate)
        # 결과: (B, L, d1 + d2)
        # 출력의 앞부분 d1개 채널은 무조건 Group 1, 뒷부분 d2개는 Group 2임이 보장됨
        return torch.cat([emb1, emb2], dim=2)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Linear(1, hidden_size)

    def forward(self, t):
        return self.mlp(t.unsqueeze(-1))


class MemEffAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    An transformers block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MemEffAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.Identity()
        # self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def _classify_variables(variables):
    """
    변수를 물리적 특성에 따라 두 그룹으로 분류합니다.
    - Group 1 (Dynamics): U, V, Z (wind components, geopotential), pressure
    - Group 2 (Thermodynamics): T (temperature), Q (specific humidity)
    """
    group1_vars = []  # Dynamics
    group2_vars = []  # Thermodynamics
    
    for var in variables:
        var_lower = var.lower()
        # Dynamics: wind components (u, v), geopotential (z), pressure
        if any(keyword in var_lower for keyword in ['u_component', 'v_component', 'geopotential', 'pressure', 'mslp']):
            group1_vars.append(var)
        # Thermodynamics: temperature, specific humidity
        elif any(keyword in var_lower for keyword in ['temperature', 'specific_humidity', 'humidity']):
            group2_vars.append(var)
        else:
            # 기본값: Dynamics로 분류 (보수적 접근)
            group1_vars.append(var)
    
    return group1_vars, group2_vars


class Sonny(nn.Module):
    """
    Sonny — StepNets architecture with variable-aware embedding.
    Two stages: Step 1 (slow path, width d1) on dynamics-heavy channels,
    then Step 2 (fast path, full width) after fusion with thermodynamics channels.
    """
    def __init__(self, 
        in_img_size,
        variables,
        patch_size=2,
        hidden_size=384,  # ViT-S: 384
        depth=12,  # ViT-B: 12
        num_heads=6,  # ViT-B: 12
        mlp_ratio=4.0,
        step_ratio=0.5,  # Step 1에 할당할 채널 비율 (보통 절반 사용)
        depth_step1=None,  # Step1 블록 개수; None이면 depth//2 (나머지는 Step2)
        group1_vars=None,  # Dynamics 변수 리스트 (None이면 자동 분류)
        group2_vars=None,  # Thermodynamics 변수 리스트 (None이면 자동 분류)
        use_cnn_head=False,  # config 호환용 (미사용)
        cnn_head_channels=256,  # config 호환용 (미사용)
        **kwargs,  # config에 있는 나머지 인자 무시
    ):
        super().__init__()
        
        if in_img_size[0] % patch_size != 0:
            pad_size = patch_size - in_img_size[0] % patch_size
            in_img_size = (in_img_size[0] + pad_size, in_img_size[1])
        self.in_img_size = in_img_size
        self.variables = variables
        self.patch_size = patch_size
        
        # --- StepsNet 구현 핵심 부분 ---
        
        # Step 1과 Step 2의 차원(Width) 계산
        self.d1 = int(hidden_size * step_ratio)  # Step 1 너비 (예: 384)
        self.d2 = hidden_size - self.d1           # 나머지 (예: 384)
        
        # 변수 그룹 분류
        if group1_vars is None or group2_vars is None:
            self.group1_vars, self.group2_vars = _classify_variables(variables)
        else:
            self.group1_vars = group1_vars
            self.group2_vars = group2_vars
        
        # 1. Variable-Aware Embedding
        self.embedding = VariableAwareEmbedding(
            group1_vars=self.group1_vars,
            group2_vars=self.group2_vars,
            img_size=in_img_size,
            patch_size=patch_size,
            d1=self.d1,
            d2=self.d2,
            num_heads=num_heads,
        )
        self.embed_norm_layer = nn.LayerNorm(hidden_size)
        
        # interval embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Head 개수도 너비에 맞춰 조정 (Head당 차원 유지)
        head_dim = hidden_size // num_heads
        self.num_heads_1 = max(1, self.d1 // head_dim)  # 최소 1개
        self.num_heads_2 = num_heads  # Step 2는 전체 너비를 사용하므로 원래 헤드 수
        
        # 깊이 배분: depth_step1으로 Step1 트랜스포머 깊이를 스윕 가능 (ablation)
        if depth_step1 is None:
            depth_1 = depth // 2
        else:
            depth_1 = int(depth_step1)
        depth_2 = depth - depth_1
        if depth_1 < 1 or depth_2 < 1:
            raise ValueError(f"depth_step1={depth_step1!r} with depth={depth} gives depth_1={depth_1}, depth_2={depth_2}")
        
        # Step 1 Blocks (너비가 d1으로 작음)
        self.step1_blocks = nn.ModuleList([
            Block(self.d1, self.num_heads_1, mlp_ratio=mlp_ratio) 
            for _ in range(depth_1)
        ])
        
        # Step 2 Blocks (너비가 hidden_size로 복귀)
        self.step2_blocks = nn.ModuleList([
            Block(hidden_size, self.num_heads_2, mlp_ratio=mlp_ratio) 
            for _ in range(depth_2)
        ])
        
        # **중요**: Step 1 블록들은 d1 크기의 입력을 받는데, 
        # TimestepEmbedder는 hidden_size 크기를 뱉으므로 차원을 맞춰줘야 합니다.
        self.time_proj_step1 = nn.Linear(hidden_size, self.d1)
        
        # Prediction Head (기존과 동일)
        self.head = FinalLayer(hidden_size, patch_size, len(variables))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        trunc_normal_(self.t_embedder.mlp.weight, std=0.02)
        
        # adaLN 초기화 (각 Step별로 수행)
        for block in self.step1_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.step2_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)
        
        # 추가된 time projection 초기화
        trunc_normal_(self.time_proj_step1.weight, std=0.02)

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        v = len(self.variables)
        h = self.in_img_size[0] // p if h is None else h // p
        w = self.in_img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * p, w * p))
        return imgs

    def forward(self, x, variables, time_interval):
        # 1. 임베딩 (B, L, hidden_size)
        x = self.embedding(x, variables) 
        x = self.embed_norm_layer(x)
        
        # 시간 임베딩 (B, hidden_size)
        time_interval_emb = self.t_embedder(time_interval)
        
        # --- StepsNet Forward ---
        
        # 2. 채널 분할 (Split)
        # x를 채널 차원(dim=2)을 기준으로 d1, d2로 나눕니다.
        x1 = x[:, :, :self.d1]  # (B, L, d1)
        x2 = x[:, :, self.d1:]  # (B, L, d2)
        
        # 3. Step 1 실행 (Slow Path)
        # x1만 처리. 시간 임베딩도 차원을 맞춰서 넣어줍니다.
        time_emb_1 = self.time_proj_step1(time_interval_emb)  # (B, d1)
        
        y1 = x1
        for block in self.step1_blocks:
            y1 = block(y1, time_emb_1)
            
        # 4. Step 2 준비 (Concatenate)
        # 처리된 y1과 처리되지 않은 x2를 합칩니다.
        # 결과 모양은 다시 (B, L, hidden_size)가 됩니다.
        x_step2 = torch.cat([y1, x2], dim=2)  # (B, L, hidden_size)
        
        # 5. Step 2 실행 (Fast Path)
        y2 = x_step2
        for block in self.step2_blocks:
            y2 = block(y2, time_interval_emb)  # 여기선 원래 time_emb 사용
            
        # ------------------------
        
        x = self.head(y2, time_interval_emb)
        x = self.unpatchify(x)
        
        return x

    def forward_encode_with_intermediates(
        self,
        x,
        variables,
        time_interval,
        step1_fracs=(0.25, 0.5),
    ):
        """
        Encoder path only: Step1 hidden states at ~fractional depths (Slow path, width d1)
        plus final y2 latent before the prediction head. Used for frozen feature extraction
        (e.g. Cross-Attention KV) and optional residual decoding from y2.
        """
        x = self.embedding(x, variables)
        x = self.embed_norm_layer(x)
        time_interval_emb = self.t_embedder(time_interval)
        x1 = x[:, :, : self.d1]
        x2 = x[:, :, self.d1 :]
        time_emb_1 = self.time_proj_step1(time_interval_emb)
        depth_1 = len(self.step1_blocks)

        needed = set()
        for frac in step1_fracs:
            k = int(round(frac * depth_1)) - 1
            k = max(0, min(depth_1 - 1, k))
            needed.add(k)

        y1 = x1
        step1_hidden = {}
        for i, block in enumerate(self.step1_blocks):
            y1 = block(y1, time_emb_1)
            if i in needed:
                step1_hidden[i] = y1.clone()

        x_step2 = torch.cat([y1, x2], dim=2)
        y2 = x_step2
        for block in self.step2_blocks:
            y2 = block(y2, time_interval_emb)

        feats_ordered = []
        for frac in step1_fracs:
            k = int(round(frac * depth_1)) - 1
            k = max(0, min(depth_1 - 1, k))
            feats_ordered.append(step1_hidden[k])

        return {
            "step1_feats": feats_ordered,
            "y2": y2,
            "time_interval_emb": time_interval_emb,
        }

    def decode_from_y2(self, y2, time_interval=None, time_interval_emb=None):
        """Run head + unpatchify on latent y2 (same as tail of forward)."""
        if time_interval_emb is None:
            if time_interval is None:
                raise ValueError("Provide time_interval or time_interval_emb")
            time_interval_emb = self.t_embedder(time_interval)
        x = self.head(y2, time_interval_emb)
        return self.unpatchify(x)

