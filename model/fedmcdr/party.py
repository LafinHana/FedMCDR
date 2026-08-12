import copy
import random
import time
from tqdm import tqdm
from .model import GAT
from torch.nn.functional import sigmoid, binary_cross_entropy
import torch
from tqdm import tqdm
import numpy as np
import math
from .rl_parts import Agent, Environment


class Server:
    def __init__(self, id, d_name, num_m, total_clients, clients, evaluate_data, user_dic, args):
        self.id = id
        self.domain_name = d_name
        self.clients = clients
        self.total_clients = total_clients
        self.num_items = num_m
        self.num_users = len(clients)
        self.V = torch.randn(num_m, args.embedding_size, device=args.device)
        self.U = torch.randn(self.num_users, args.embedding_size, device=args.device)
        torch.nn.init.uniform(self.U, a=0., b=1.)
        torch.nn.init.uniform(self.V, a=0., b=1.)
        self.evaluate_data = torch.tensor(evaluate_data).to(args.device)
        self.item_gat = GAT(args, args.embedding_size, args.embedding_size, args.embedding_size)
        self.domain_attention = torch.randn(1, args.num_domain, device=args.device)
        self.user_embedding_with_attention = torch.zeros_like(self.U)
        self.item_embedding_with_attention = torch.zeros_like(self.V)
        self.lg10 = torch.Tensor([math.log(2) / math.log(i + 2) for i in range(10)]).to(args.device)
        self.lg5 = torch.Tensor([math.log(2) / math.log(i + 2) for i in range(5)]).to(args.device)
        self.user_dic = user_dic
        self.args = args
        self.mlp = None
        self.agent = Agent(args)

    def mf_train(self):
        batch_num = math.ceil(self.num_users / self.args.user_batch)
        ids = copy.deepcopy(self.clients)
        np.random.shuffle(ids)
        for bt in tqdm(range(batch_num)):
            grads, p = [], []
            item_interact_table = torch.zeros(self.num_items).to(self.args.device)
            s, t = bt * self.args.user_batch, min((bt+1) * self.args.user_batch, self.args.num_users)
            batch_user = ids[s:t]
            for it in batch_user:
                if len(self.total_clients[it].train_data[self.id]) == 0:
                    continue
                map_id = self.user_dic[it][self.domain_name]
                grad, items = self.total_clients[it].train(self.id, map_id, self.U, self.V)
                grads.append(grad)
                item_interact_table[items] += 1
            item_interact_table[item_interact_table == 0] = 1
            for it, vl in enumerate(grads):
                u_grad, i_grad = vl[0], vl[1]
                map_id = self.user_dic[batch_user[it]][self.domain_name]
                self.U[map_id] -= u_grad
                self.V -= i_grad / item_interact_table.unsqueeze(1)

    def metric_at_k(self, test_predictions, k, epoch_id, output_file):
        length = int(len(test_predictions) / 100)
        test_predictions = test_predictions.reshape(length, 100)
        values, indices = torch.topk(test_predictions, k, dim=1, largest=True)
        loc = indices == 99
        hr = torch.sum(loc).item() / length
        if k == 10:
            ndcg = torch.sum(self.lg10 * loc).item() / length
        else:
            ndcg = torch.sum(self.lg5 * loc).item() / length
        return hr, ndcg

    def test(self, U, V, epoch_id, output_file):
        test_data = self.evaluate_data
        with torch.no_grad():
            test_user, test_item = test_data[:, 0], test_data[:, 1]
            test_predictions = sigmoid(torch.sum(torch.multiply(U[test_user], V[test_item]), dim=-1))
            hr_5, ndcg_5 = self.metric_at_k(test_predictions, 5, epoch_id, output_file)
            hr_10, ndcg_10 = self.metric_at_k(test_predictions, 10, epoch_id, output_file)
            return hr_5, ndcg_5, hr_10, ndcg_10

    def test_gat(self, epoch_id, output_file):
        self.item_gat.eval()
        return self.test(self.user_embedding_with_attention, self.V, epoch_id, output_file)

    def test_agent(self, epoch_id, output_file):

        for it in tqdm(self.clients):
            map_id = self.user_dic[it][self.domain_name]
            self.user_embedding_with_attention[map_id] = self.total_clients[it].test(self.agent, self.id)
        return self.test(self.user_embedding_with_attention, self.V, epoch_id, output_file)

    def test_mf(self, epoch_id, output_file):
        return self.test(self.U, self.V, epoch_id, output_file)

    def train_mlp(self, batch):
        s, t = batch * self.args.user_batch, min((batch + 1) * self.args.user_batch, self.args.num_users)
        selected_clients = [i for i in range(s, t)]
        grads = []
        for it in selected_clients:
            grads.append(self.total_clients[it].train_mlp(self.mlp))
        p = 1 / len(self.total_clients)
        for it in grads:
            for d in range(self.args.num_domain - 1):
                gd = it[d]
                for i, vl in enumerate(self.mlp[d].parameters()):
                    vl.data -= p * gd[i]

    def kt_stage(self, tf_flag=False):
        batch_num = math.ceil(self.num_users / self.args.user_batch)
        ids = copy.deepcopy(self.clients)
        np.random.shuffle(ids)
        for bt in tqdm(range(batch_num)):
            grads_model, p, grads_embedding, grads_kt = [], [], [], []
            total_item_interact_table = torch.zeros(self.num_items).to(self.args.device)
            s, t = bt * self.args.user_batch, min((bt + 1) * self.args.user_batch, self.num_users)
            batch_user = ids[s:t]
            no_trans = self.args.user_batch * 1
            for i, it in enumerate(batch_user):
                if len(self.total_clients[it].train_data[self.id]) == 0:
                    continue
                if tf_flag is False or i >= no_trans:
                    pk, grad_gat, grad_emb, grad_kt = self.total_clients[it].train_gat(self.id, self.user_dic,
                                                                                       self.item_gat, self.U, self.V)
                else:
                    pk, grad_gat, grad_emb, grad_kt = self.total_clients[it].knowledge_transfer(self.id, self.mlp,
                                                                                                self.user_dic,
                                                                                                self.item_gat, self.U,
                                                                                                self.V, self.agent)
                    grads_kt.append(grad_kt)
                total_items = grad_emb[3]
                total_item_interact_table[total_items] += 1
                p.append(pk)
                grads_model.append(grad_gat)
                grads_embedding.append(grad_emb)

            p = torch.Tensor(p)
            p = p / torch.sum(p)
            for i, it in enumerate(grads_model):
                if tf_flag and i < no_trans:
                    for mid, mlp in enumerate(self.mlp):
                        for pid, para in enumerate(mlp.parameters()):  # 看看global里面有没有grad
                            para.data -= p[i] * grads_kt[i][mid][pid]

                for j, vl in enumerate(self.item_gat.parameters()):
                    vl.data -= p[i] * it[j]
            total_item_interact_table[total_item_interact_table == 0] = 1
            for grad in grads_embedding:
                uid, u_emb, u_grad, total_items, total_grads = grad[0], grad[1], grad[2], grad[3], grad[4]
                map_id = self.user_dic[uid][self.domain_name]
                self.user_embedding_with_attention[map_id] = u_emb
                self.U[map_id] = u_grad
                self.V[total_items] -= total_grads / total_item_interact_table[total_items].unsqueeze(1)

    def distribute_model(self):
        for client in self.clients:
            self.total_clients[client].get_global_model(self.id, self.item_gat, self.user_dic, self.U, self.V)

    def train_agent(self):
        batch_num = math.ceil(self.num_users / self.args.user_batch)
        ids = copy.deepcopy(self.clients)
        np.random.shuffle(ids)
        for bt in tqdm(range(batch_num)):
            grads_model, p = [], []
            s, t = bt * self.args.user_batch, min((bt + 1) * self.args.user_batch, self.num_users)
            batch_user = ids[s:t]
            for i, it in enumerate(batch_user):
                if len(self.total_clients[it].train_data[self.id]) == 0:
                    continue
                pk, grads = self.total_clients[it].train_agent(self.agent, self.id, self.user_dic, self.U, self.V)
                p.append(pk)
                grads_model.append(grads)
            p = torch.Tensor(p)
            p = p / torch.sum(p)
            for i, it in enumerate(grads_model):
                for j, vl in enumerate(self.agent.low_model.parameters()):
                    vl.data -= p[i] * it[j]


class Client:
    def __init__(self, id, train_data, num_m, rating_mean, domain_names, args):
        self.id = id
        self.rating_mean = rating_mean
        self.train_data = [torch.tensor(train_data[i], device=args.device) for i in range(args.num_domain)]
        self.batch_size = [len(train_data[i]) for i in range(args.num_domain)]
        self.items = train_data
        self.gat = None
        self.knowledge = [[] for _ in range(args.num_domain)]
        self.num_items = num_m
        self.unselected = []
        self.mlp = []
        self.transfer_vec = []
        self.args = args
        self.delta = torch.tensor(args.delta, device=args.device)
        self.sensitivity = torch.sqrt(torch.tensor(1, device=args.device))
        self.domain_names = domain_names
        # rl params
        self.sample_times = 3
        self.env = Environment(None, 2 * args.embedding_size + 2, None, args)
        self.agent = Agent(args)
        self.frozen_user_embedding = None
        self.frozen_item_embedding = None

    def reset(self, input):
        output = torch.clone(input).detach()
        output.requires_grad = True
        output.grad = torch.zeros_like(output)  # 感觉可以去掉
        return output

    @staticmethod
    def sample_negative(data, num):
        neg = torch.randint(0, num, (4 * len(data), 1), device=data.device, dtype=torch.int64).squeeze()
        rating = torch.cat((torch.ones(len(data), device=data.device),
                            torch.zeros(len(neg), device=data.device)), dim=0)
        neg = torch.cat((data, neg), dim=0)
        return neg, rating

    def train(self, domain_id, map_id, user_embedding, item_embedding):
        #怎么对clip等操作求梯度？
        domain_items, domain_ratings = self.sample_negative(self.train_data[domain_id], self.num_items[domain_id])
        item_emb = self.reset(item_embedding)
        user_emb = self.reset(user_embedding[map_id])
        optimizer = torch.optim.AdamW([user_emb, item_emb], lr=self.args.lr_mf,weight_decay=self.args.weight_decay)
        for _ in range(self.args.local_epoch):
            optimizer.zero_grad()
            # predict = torch.sigmoid(torch.matmul(user_emb, total_select.t())) * 4 + 1
            predict = torch.sum(torch.multiply(user_emb, item_emb[domain_items]), dim=1)
            predict = sigmoid(predict)
            loss = binary_cross_entropy(predict, domain_ratings)
            loss.backward()
            optimizer.step()
        grads = [user_embedding[map_id].detach() - user_emb.detach(), item_embedding.detach() - item_emb.detach()]
        return grads, domain_items  #检查一下grads是否在之后过程中产生了grad

    def train_gat(self, domain_id, user_dic, model_item, global_user_embedding, global_item_embedding, transfer=False,
                  agent=None, transfer_vec=None):

        grads_gat, grad_emb, grad_kt, temp_vec = [], [], [], [0 for _ in range(self.args.num_domain)]
        length = len(self.items[domain_id])
        self.gat = copy.deepcopy(model_item)
        user_embedding = self.reset(global_user_embedding[user_dic[self.id][self.domain_names[domain_id]]])
        item_embedding = self.reset(global_item_embedding)
        paras = [user_embedding, item_embedding] + [para for para in self.gat.parameters()]
        non_w_paras = []
        if transfer:
            mlps = copy.deepcopy(self.mlp)
            for mlp in mlps:
                for k, v in mlp.named_parameters():
                    if k != 'k': paras.append(v)
                    else: non_w_paras.append(v)
            # paras.append(local_a)
        # optimizer = torch.optim.AdamW(paras, lr=self.args.lr_gat, weight_decay=self.args.weight_decay)
        optimizer = torch.optim.AdamW([
            {'params': paras, 'weight_decay': self.args.weight_decay},
            {'params': non_w_paras, 'weight_decay': self.args.weight_decay_k}], lr=self.args.lr_gat)
        total_item, ratings = self.sample_negative(self.train_data[domain_id], self.num_items[domain_id])
        for epoch in range(self.args.local_epoch):
            optimizer.zero_grad()
            if transfer:
                for i in range(self.args.num_domain):
                    temp_vec[i] = mlps[i](transfer_vec[i])
            h_i, intermediate_emb, ls, lm = self.gat(
                torch.cat((user_embedding.reshape(1, self.args.embedding_size), item_embedding[self.items[domain_id]])),
                transfer, agent, temp_vec, self.env)
            user_emb = h_i[0]
            h_i = item_embedding[total_item]
            predict = sigmoid(torch.sum(torch.multiply(user_emb, h_i), dim=1))
            loss = binary_cross_entropy(predict, ratings) + ls + lm
            loss.backward()
            optimizer.step()

        local_para = [para.data for para in self.gat.parameters()]
        global_para = [para.data for para in model_item.parameters()]
        for i in range(len(local_para)):
            grads_gat.append(global_para[i] - local_para[i])
        with torch.no_grad():
            user_emb, self.knowledge[domain_id], ls, lm = self.gat(
                torch.cat((user_embedding.reshape(1, self.args.embedding_size), item_embedding[self.items[domain_id]])),
                transfer, agent, transfer_vec, self.env)
        grad_emb.append(self.id)
        grad_emb.append(user_emb[0].detach())
        grad_emb.append(user_embedding.detach())
        grad_emb.append(total_item)
        grad_emb.append(global_item_embedding[grad_emb[-1]].detach() - item_embedding[grad_emb[-1]].detach())
        if transfer:
            for i in range(self.args.num_domain):
                local_para = [para.data for para in mlps[i].parameters()]
                global_para = [para.data for para in self.mlp[i].parameters()]
                para_grad = []
                for pid in range(len(local_para)):
                    para_grad.append(global_para[pid] - local_para[pid])
                grad_kt.append(para_grad)
        return length, grads_gat, grad_emb, grad_kt

    @staticmethod
    def l2_clip(x, s):
        norm = torch.norm(x)
        if norm > s:
            return s * (x / norm)
        else:
            return x

    def inject_comprehensive_perturbation(self, h, miu=0.0, dropout_rate=0.1):
        """
        对传入的原始特征向量 h 注入综合扰动池（CPP）噪声。
        包含：乘性特征丢失（Dropout）、方差对齐的高斯、拉普拉斯、均匀复合加性噪声（系数随机且和为1）。
        
        参数:
            h (Tensor): 原始特征向量或矩阵，shape 可以是 (embedding_size,) 或 (batch_size, embedding_size)
            miu (float): 高斯噪声的均值，默认为 0.0
            dropout_rate (float): 乘性特征丢失率，默认为 0.1
            force_dropout (bool): 是否在测试集（eval模式）下也强制进行特征丢失。
                                默认仅在训练期（self.training=True）进行。
                                
        返回:
            h_corrupted (Tensor): 注入综合扰动后的特征，shape 与输入 h 完全一致
        """
        device = self.args.device
        sigma = self.args.sigma

        # 1. 乘性特征丢失 (Multiplicative Dropout)
        # 推荐系统做负迁移仿真测试时，如果想在测试/评估阶段也施加维度丢失，可将 force_dropout 设为 True
        
        dropout_mask = (torch.rand_like(h) >= dropout_rate).float().to(device)
        h_dropped = h * dropout_mask
        

        # 2. 随机生成 3 个加和严格为 1 的噪声系数 (alpha_g + alpha_l + alpha_u = 1.0)
        raw_coeffs = torch.rand(3).to(device)
        alphas = raw_coeffs / raw_coeffs.sum()  # 归一化
        alpha_g, alpha_l, alpha_u = alphas[0].item(), alphas[1].item(), alphas[2].item()

        # 3. 生成三种方差对齐的加性噪声
        # (1) 高斯噪声
        epsilon_gaussian = torch.normal(miu, sigma, size=h.shape).to(device)

        # (2) 拉普拉斯噪声 (方差对齐: Var = 2 * b^2 = sigma^2  => b = sigma / sqrt(2))
        b_laplacian = sigma / (2.0 ** 0.5)
        laplace_dist = torch.distributions.laplace.Laplace(
            torch.zeros_like(h).to(device),
            torch.ones_like(h).to(device) * b_laplacian
        )
        epsilon_laplacian = laplace_dist.sample()

        # (3) 均匀分布噪声 (方差对齐: Var = a^2 / 3 = sigma^2  => a = sigma * sqrt(3))
        a_uniform = sigma * (3.0 ** 0.5)
        epsilon_uniform = (torch.rand_like(h).to(device) * 2.0 - 1.0) * a_uniform

        # 4. 融合随机系数，生成最终的复合加性噪声
        negative_noise = alpha_g * epsilon_gaussian + alpha_l * epsilon_laplacian + alpha_u * epsilon_uniform

        # 5. 最终合成被复合污染后的特征
        h_corrupted = h_dropped + negative_noise

        return h_corrupted

    def prepare_transfer(self, domain_id, low_model):
        self.agent.low_model = low_model
        std = self.sensitivity * torch.sqrt(2 * torch.log(1.25 / self.delta)) * 1 / self.args.eps
        for j in range(self.args.num_domain):
            if j == domain_id:
                self.transfer_vec.append(torch.zeros(self.args.embedding_size, device=self.args.device))
            else:
                if len(self.knowledge[j]) == 0:
                    temp_vec = torch.zeros(self.args.embedding_size, device=self.args.device)
                else:
                    temp_vec = Client.l2_clip(torch.tensor(self.knowledge[j][0], device=self.args.device),
                                              self.sensitivity)
                if self.args.miu > 0:
                    miu = random.choice((-1, 1)) * random.uniform(self.args.miu-1, self.args.miu)
                else:
                    miu = 0

                if self.args.dp:
                    if self.args.mix_noise and self.args.sigma > 0:
                        noise = torch.normal(mean=0, std=torch.sqrt(std ** 2),
                                     size=(self.args.embedding_size,)).to(self.args.device)
                        self.transfer_vec.append(self.inject_comprehensive_perturbation(temp_vec + noise))
                    else: 
                        noise = torch.normal(mean=miu, std=torch.sqrt(std ** 2 + self.args.sigma ** 2),
                                     size=(self.args.embedding_size,)).to(self.args.device)
                        self.transfer_vec.append(temp_vec + noise)
                else:
                    self.transfer_vec.append(temp_vec)

    def knowledge_transfer(self, domain_id, mlps, user_dic, item_gat, user_embedding, item_embedding, agent):
        self.mlp = mlps
        return self.train_gat(domain_id, user_dic, item_gat, user_embedding, item_embedding, True, agent,
                              self.transfer_vec)

    @staticmethod
    def _get_high_action(prob, Random):
        """
        判断是不是要处理序列，random起到的作用是改变阈值，等于false时prob固定大于0.5才要改，否则每个都是随机数，不太明白为什么引入随机数，会有什么好处？
        """
        batch_size = prob.shape[0]
        if Random:
            random_number = np.random.rand(batch_size)
            return np.where(random_number < prob, np.ones(batch_size, dtype=np.int), np.zeros(batch_size, dtype=np.int))
        else:
            return np.where(prob >= 0.5, np.ones(batch_size, dtype=np.int), np.zeros(batch_size, dtype=np.int))

    def test(self, global_agent, domain_id):
        with torch.no_grad():
            _, h_hat = self.gat(
                torch.cat((self.frozen_user_embedding.reshape(1, self.args.embedding_size),
                           self.frozen_item_embedding[self.items[domain_id]])),
                False, global_agent, None, self.env)

        return h_hat[0].data

    def get_global_model(self, domain_id, model, user_dic, global_user_embedding, global_item_embedding):
        self.gat = copy.deepcopy(model)
        for para in self.gat.parameters():
            para.requires_grad = False
        self.frozen_user_embedding = global_user_embedding[user_dic[self.id][self.domain_names[domain_id]]]
        self.frozen_item_embedding = global_item_embedding

    def train_agent(self, global_agent, domain_id, user_dic, global_user_embedding, global_item_embedding):
        grads = []
        length = len(self.items[domain_id])
        self.agent = copy.deepcopy(global_agent)
        total_item, ratings = self.sample_negative(self.train_data[domain_id], self.num_items[domain_id])
        for epoch in range(self.args.local_epoch):
            model = self.agent.low_model
            optim = torch.optim.AdamW(params=model.parameters(), lr=self.args.lr_agent,
                                      weight_decay=self.args.weight_decay)
            optim.zero_grad()
            h, h_hat = self.gat(
                torch.cat((self.frozen_user_embedding.reshape(1, self.args.embedding_size),
                           self.frozen_item_embedding[self.items[domain_id]])),
                False, self.agent, None, self.env)
            user_emb = h[0]
            user_emb_hat = h_hat[0]
            item_emb = self.frozen_item_embedding[total_item]
            predict = sigmoid(torch.sum(torch.multiply(user_emb, item_emb), dim=1))
            predict_hat = sigmoid(torch.sum(torch.multiply(user_emb_hat, item_emb), dim=1))
            likelihood = binary_cross_entropy(predict, ratings)
            likelihood_hat = binary_cross_entropy(predict_hat, ratings)
            reward = -torch.sum(likelihood_hat - likelihood)

            prob = self.env.prob_matrix
            action = self.env.action_matrix
            pi = action * prob + (1 - action) * (1 - prob)
            loss = -1 * torch.mean(torch.log(pi) * reward)

            loss.backward()
            optim.step()

        with torch.no_grad():
            local_para = [para.data for para in self.agent.low_model.parameters()]
            global_para = [para.data for para in global_agent.low_model.parameters()]
            for i in range(len(local_para)):
                grads.append(global_para[i] - local_para[i])
        return length, grads
