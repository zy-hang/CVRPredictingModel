import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizationMachine(nn.Module):
    """FM层，用于交互特征并输出最终预估分数"""

    def __init__(self, input_dim, embedding_dim=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.v = nn.Parameter(torch.randn(input_dim, embedding_dim))
        nn.init.xavier_normal_(self.v)
        self.bias = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        linear_term = self.linear(x)
        square_of_sum = (torch.mm(x, self.v) ** 2).sum(1, keepdim=True)
        sum_of_square = torch.mm(x ** 2, self.v ** 2).sum(1, keepdim=True)
        interaction_term = 0.5 * (square_of_sum - sum_of_square)
        output = linear_term + interaction_term + self.bias
        return self.sigmoid(output)


class Expert1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.fc(x))


class Expert3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = input_dim * 4
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


class Expert2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self, x):
        return self.fc2(self.layernorm(self.fc1(x) * x))


class Tower(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.tower = FactorizationMachine(input_dim)

    def forward(self, x):
        return self.tower(x)


class GateNetwork(nn.Module):
    def __init__(self, input_dim, expert_count):
        super().__init__()
        self.gate = nn.Linear(input_dim, expert_count)

    def forward(self, x):
        return F.softmax(self.gate(x), dim=-1)

class LHUC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim/4)),
            nn.SiLU(),
            nn.Linear(int(dim/4), dim),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.net(x)


class ExtractionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_specific_experts, num_shared_experts, depth):
        super().__init__()
        self.depth = depth
        # 各任务专属专家 & 共享专家
        if depth == 1:
            self.task_a_experts = nn.ModuleList([Expert1(input_dim, output_dim) for _ in range(num_specific_experts)])
            self.task_b_experts = nn.ModuleList([Expert1(input_dim, output_dim) for _ in range(num_specific_experts)])
            self.task_c_experts = nn.ModuleList([Expert1(input_dim, output_dim) for _ in range(num_specific_experts)])
            self.shared_experts = nn.ModuleList([Expert1(input_dim, output_dim) for _ in range(num_shared_experts)])
        elif depth == 2:
            self.task_a_experts = nn.ModuleList([Expert2(input_dim, output_dim) for _ in range(num_specific_experts)])
            self.task_b_experts = nn.ModuleList([Expert2(input_dim, output_dim) for _ in range(num_specific_experts)])
            self.task_c_experts = nn.ModuleList([Expert2(input_dim, output_dim) for _ in range(num_specific_experts)])
            self.shared_experts = nn.ModuleList([Expert2(input_dim, output_dim) for _ in range(num_shared_experts)])
        else:
            self.task_a_experts = nn.ModuleList([Expert3(input_dim, output_dim) for _ in range(num_specific_experts)])
            self.task_b_experts = nn.ModuleList([Expert3(input_dim, output_dim) for _ in range(num_specific_experts)])
            self.task_c_experts = nn.ModuleList([Expert3(input_dim, output_dim) for _ in range(num_specific_experts)])
            self.shared_experts = nn.ModuleList([Expert3(input_dim, output_dim) for _ in range(num_shared_experts)])

        # 每个任务门控网络要选择的专家数 = 专属 + 共享
        experts_per_task = num_specific_experts + num_shared_experts
        # 共享门控要选择的专家数 = 所有专属 + 共享
        total_experts = num_specific_experts * 3 + num_shared_experts

        # 门控网络
        self.task_a_gate = GateNetwork(input_dim, experts_per_task)
        self.task_b_gate = GateNetwork(input_dim, experts_per_task)
        self.task_c_gate = GateNetwork(input_dim, experts_per_task)
        self.shared_gate = GateNetwork(input_dim, total_experts)

        if depth == 3:
            self.LHUC_a = LHUC(input_dim)
            self.LHUC_b = LHUC(input_dim)
            self.LHUC_c = LHUC(input_dim)

    def forward(self, A, B, C, S):
        # --- 1. 专家输出 ---
        a_outs = [e(A) for e in self.task_a_experts]
        b_outs = [e(B) for e in self.task_b_experts]
        c_outs = [e(C) for e in self.task_c_experts]
        s_outs = [e(S) for e in self.shared_experts]
        # --- 2. 门控权重 ---
        a_w = self.task_a_gate(A)  # [batch, num_specific+num_shared]
        b_w = self.task_b_gate(B)
        c_w = self.task_c_gate(C)
        s_w = self.shared_gate(S)  # [batch, all_experts]
        # 获取专属专家和共享专家的权重数量
        num_specific = len(self.task_a_experts)
        num_shared = len(self.shared_experts)

        # --- 3. 计算特定专家和共享专家的贡献 ---
        def weighted_expert_sum(experts, weights, specific_count):
            # 特定专家部分
            specific_experts = experts[:specific_count]
            specific_weights = weights[:, :specific_count]

            specific_sum = torch.zeros_like(specific_experts[0])
            for i, expert in enumerate(specific_experts):
                specific_sum += expert * specific_weights[:, i].unsqueeze(1)

            # 共享专家部分
            shared_experts = experts[specific_count:]
            shared_weights = weights[:, specific_count:]

            shared_sum = torch.zeros_like(shared_experts[0])
            for i, expert in enumerate(shared_experts):
                shared_sum += expert * shared_weights[:, i].unsqueeze(1)

            return specific_sum, shared_sum

        # 计算每个任务的特定专家和共享专家部分
        a_specific, a_shared = weighted_expert_sum(a_outs + s_outs, a_w, num_specific)
        b_specific, b_shared = weighted_expert_sum(b_outs + s_outs, b_w, num_specific)
        c_specific, c_shared = weighted_expert_sum(c_outs + s_outs, c_w, num_specific)
        # 如果是第3层，应用LHUC机制
        if self.depth == 3:
            gating_A = self.LHUC_a(S)
            gating_B = self.LHUC_b(S)
            gating_C = self.LHUC_c(S)

            # 特定专家部分与gating相乘
            a_specific = a_specific * gating_A
            b_specific = b_specific * gating_B
            c_specific = c_specific * gating_C

        # --- 4. 合并特定专家和共享专家部分 ---
        a_feat = a_specific + a_shared
        b_feat = b_specific + b_shared
        c_feat = c_specific + c_shared

        # 共享特征的计算
        all_experts = a_outs + b_outs + c_outs + s_outs
        s_feat = torch.zeros_like(all_experts[0])
        for i, expert in enumerate(all_experts):
            s_feat += expert * s_w[:, i].unsqueeze(1)
        return a_feat, b_feat, c_feat, s_feat


class PLE(nn.Module):
    def __init__(self,
                 input_dim=615,
                 expert_dim=128,
                 num_specific_experts=2,
                 num_shared_experts=1,
                 num_levels=3):
        super().__init__()
        self.extraction_layers = nn.ModuleList()
        # 第一层
        self.extraction_layers.append(
            ExtractionLayer(input_dim, expert_dim, num_specific_experts, num_shared_experts, depth=1)
        )
        # 后续层
        for i in range(1, num_levels):
            depth = min(i + 1, 3)  # 限制depth最大为3
            self.extraction_layers.append(
                ExtractionLayer(expert_dim, expert_dim, num_specific_experts, num_shared_experts, depth=depth)
            )

        # 三个任务塔
        self.task_a_tower = Tower(expert_dim)
        self.task_b_tower = Tower(expert_dim)
        self.task_c_tower = Tower(expert_dim)

    def forward(self, x):
        # 第一层，输入全用 x
        a, b, c, s = self.extraction_layers[0](x, x, x, x)
        # 后续层，按上一级输出递归
        for layer in self.extraction_layers[1:]:
            a, b, c, s = layer(a, b, c, s)

        # 三塔分别做最终预测
        y1 = self.task_a_tower(a)
        y2 = self.task_b_tower(b)
        y3 = self.task_c_tower(c)
        return y1, y1 * y3 + (1 - y1) * y2
