import torch.nn as nn
import torch
# torch.autograd.set_detect_anomaly(True)
class GATLayer(nn.Module):
    def __init__(self, in_feature, out_feature, alpha):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.A = nn.Parameter(torch.empty(size=(2 * out_feature, 1)))
        nn.init.xavier_uniform_(self.A.data, nn.init.calculate_gain('relu'))
        # torch.nn.init.uniform(self.A, a=0., b=1.)
        self.alpha = alpha

    def forward(self, input, adj):
        h = input
        h1 = torch.matmul(h, self.A[self.out_feature:, :])
        h2 = torch.matmul(h, self.A[:self.out_feature, :])
        e = h1 + h2.T
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=-1)
        ah = torch.matmul(attention, h)
        return ah


class GAT(nn.Module):
    def __init__(self, args, in_feature, hid_feature=16, out_feature=16, alpha=0.1, dropout=0):
        super().__init__()
        self.args = args
        self.in_feature = in_feature
        self.hid_feature = hid_feature
        self.out_feature = out_feature
        self.drop = nn.Dropout(p=dropout)
        self.in2hidden = GATLayer(in_feature, hid_feature, alpha).to(args.device)
        self.hidden2out = GATLayer(hid_feature, out_feature, alpha).to(args.device)

    @staticmethod
    def compute_ls(f_t, f_s):
        total_sim = 0
        for fs in f_s:
            with torch.no_grad():
                sim = (torch.cosine_similarity(fs, f_t, dim=0) + 1) / 2
                total_sim += sim
        F_s = 0
        for fs in f_s:
            sim = (torch.cosine_similarity(fs, f_t, dim=0) + 1) / 2
            F_s += sim * fs / total_sim
        loss = torch.norm(f_t - F_s) ** 2
        return loss

    @staticmethod
    def compute_lm(f_t, f_s):
        loss = 0
        for fs in f_s:
            loss += torch.nn.functional.mse_loss(fs, f_t)
        return loss

    def sampling_RL(self, environment, user_input, item_input, agent, Random=True):
        environment.reset_state(item_input.shape[0])
        # high_state = self.env.get_overall_state()
        # high_prob = agent.get_prob('high', high_state)
        # high_action = self._get_high_action(high_prob, Random)

        low_state = environment.get_state(user_input, item_input)
        low_prob = agent.get_prob('low_model', low_state)
        low_action = environment.get_low_action(low_prob, Random)
        environment.update_state(low_action, low_state, low_prob)
        return low_action * item_input  # 这里面会有负零

    def _sampling_RL(self, environment, user_input, item_input, agent, Random=True):
        environment.reset_state(item_input.shape[0])
        # high_state = self.env.get_overall_state()
        # high_prob = agent.get_prob('high', high_state)
        # high_action = self._get_high_action(high_prob, Random)
        for i in range(self.args.embedding_size):
            low_state = environment._get_state(user_input, item_input, i)
            low_prob = agent.get_prob('low_model', low_state)
            low_action = environment.get_low_action(low_prob, Random)
            environment.update_state(low_action, low_state, low_prob, i)
        environment.prob_matrix = torch.cat(environment.prob_matrix, dim=-1)
        return environment.action_matrix * item_input  # 这里面会有负零

    def forward(self, x, is_transfer_stage=False, agent=None, transfer_vec=None, environment=None):
        ls, lm = 0, 0
        alpha, beta = 0.01, 0.01
        intermediate_embedding = []
        adj = torch.eye(len(x), device=x.device)
        adj[:, 0] = 1.
        adj[0, :] = 1.
        x = self.in2hidden(x, adj)

        intermediate_embedding.append(x[0].data)
        if is_transfer_stage:
            # ls = alpha / 2 * self.compute_ls(x[0], transfer_vec)
            # lm = beta / 2 * self.compute_lm(x[0], transfer_vec)
            transfer_vec = torch.stack(transfer_vec)
            # scaling_factor = torch.norm(x[0]) / torch.norm(transfer_vec, dim=1) / self.args.num_domain
            if agent is not None:
                x_hat = self._sampling_RL(environment, x[0], transfer_vec, agent, False)
                x = torch.cat((x, x_hat))
            else:
                x = torch.cat((x, transfer_vec))
            adj = torch.eye(len(x), device=x.device)
            adj[:, 0] = 1.
            adj[0, :] = 1.
        else:
            if agent is not None:
                x_hat = self._sampling_RL(environment, x[0], x[1:], agent)
                x_hat = self.hidden2out(torch.cat((x[0].unsqueeze(dim=0), x_hat)), adj)
                x = self.hidden2out(x, adj)
                return x, x_hat  # 这种写法太丑了，重构的时候要改下agent训练阶段接受的返回值

        x = self.hidden2out(x, adj)
        return x, intermediate_embedding, ls, lm


class MLP(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.L1 = nn.Linear(in_feature, 2*in_feature)
        self.L2 = nn.Linear(2*in_feature, in_feature)
        self.f = nn.Tanh()
        self.k = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.k * self.f(self.L1(x))
        x = self.k * self.f(self.L2(x))  # 先验知识，knowledge的大小在0~1之间
        return x
