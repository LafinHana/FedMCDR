import copy

import torch
import torch.nn as nn
import math
import numpy as np

class Agent_Model(nn.Module):
    def __init__(self, state_dim, weight_dim, embedding_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, weight_dim)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.h = nn.Parameter(torch.empty(size=(weight_dim, 1)))
        nn.init.trunc_normal_(self.h, 0.0, math.sqrt(weight_dim / 2))

    def forward(self, input_state):
        temp = self.fc(input_state)
        temp = self.ReLU(temp)
        temp = torch.matmul(temp, self.h)
        temp = self.Sigmoid(temp)
        temp = torch.clip(temp, 1e-5, 1 - 1e-5)
        return temp


class Agent:
    def __init__(self, args):
        self.global_step = 0  # use for what?
        self.lr = args.lr_agent
        self.tau = None  # args.agent_pretrain_tau
        # self.high_state_size = args.high_state_size
        # self.low_state_size = args.embedding_size + 3
        self.low_state_size = 5
        self.weight_size = args.agent_weight_size
        self.high_model = None
        self.target_high_model = None
        self.low_model = Agent_Model(self.low_state_size, self.weight_size, args.embedding_size).to(args.device)
        # self.target_low_model = Agent_Model(self.low_state_size, self.weight_size, args.embedding_size).to(args.device)

    def train(self, name, state, action, reward):
        # name 只能是high或low，对应计算target high prob和target low prob；看源代码并没有去计算high prob或low prob
        model = getattr(self, name)
        optim = torch.optim.Adam(params=model.parameters(), lr=self.lr)
        optim.zero_grad()
        prob = model(state)
        pi = action * prob + (1 - action) * (1 - prob)
        loss = -1 * torch.sum(torch.sqrt(pi) * reward)
        loss.backward()
        optim.step()

    def get_prob(self, name, state):
        model = getattr(self, name)
        prob = model(state)
        return prob

    def update(self, name):
        if name == 'high':
            for para, para_tar in zip(self.high_model.parameters(), self.target_high_model.parameters()):
                para_tar.data = para.data * self.tau + para_tar.data * (1 - self.tau)
        else:
            for para, para_tar in zip(self.low_model.parameters(), self.target_low_model.parameters()):
                para_tar.data = para.data * self.tau + para_tar.data * (1 - self.tau)

    def assign(self, name):
        if name == 'high':
            for para, para_tar in zip(self.high_model.parameters(), self.target_high_model.parameters()):
                para.data = para_tar.data
        else:
            for para, para_tar in zip(self.low_model.parameters(), self.target_low_model.parameters()):
                para.data = para_tar.data


class Environment:
    def __init__(self, high_state_size, low_state_size, batch_size, args):
        self.gamma = 0.5
        self.high_state_size = high_state_size
        self.low_state_size = low_state_size
        # self.padding_number = padding_number
        self.origin_train_rewards = None  # 这里的reward一直是初始的吗，目前看他代码里没有对这里进行过修改
        self.origin_test_rewards = None
        self.origin_rewards = None
        self.embedding_size = args.embedding_size  # 16
        self.set_train_original_rewards()
        self.batch_size = batch_size
        self.args = args

    def get_low_action(self, prob, random=False):
        """
        同上，唯一的区别是处理padding
        """
        # if self.args.train_agent == 0:
        #     random = False
        matrix_zeros = torch.zeros(prob.shape, dtype=torch.int, device=self.args.device)
        matrix_ones = torch.ones(prob.shape, dtype=torch.int, device=self.args.device)
        if random:
            random_number = torch.rand(prob.shape, device=self.args.device)
            return torch.where(random_number < prob, matrix_ones, matrix_zeros)
        else:
            return torch.where(prob >= self.args.random_rate, matrix_ones, matrix_zeros)

    def get_reward(self, user_embedding, item_embedding, negative=False):
        reward = torch.nn.functional.sigmoid(torch.matmul(user_embedding, item_embedding))
        if negative:
            reward = 1 - reward
        return reward
    def set_train_original_rewards(self):
        self.origin_rewards = self.origin_train_rewards

    def set_test_original_rewards(self):
        self.origin_rewards = self.origin_test_rewards

    def reset_state(self, batch_size):
        self.batch_size = batch_size
        # 涉及到batch size的都要改，感觉应该改成embedding_size
        self.origin_prob = torch.zeros((self.batch_size, 1))

        self.dot_product_sum = torch.zeros((self.batch_size, 1))
        self.dot_product_mean = torch.zeros((self.batch_size, 1))

        self.element_wise_mean = torch.zeros((self.batch_size, self.embedding_size))
        self.element_wise_sum = torch.zeros((self.batch_size, self.embedding_size))

        self.vector_sum = torch.zeros((self.batch_size, 1))
        self.num_selected = torch.zeros(self.batch_size, dtype=torch.int)
        self.action_matrix = torch.zeros((self.batch_size, self.embedding_size), device=self.args.device)
        self.prob_matrix = []
        self.state_matrix = torch.zeros((self.batch_size, self.embedding_size, 5), device=self.args.device)
        # self.selected_input = torch.full((self.batch_size, self.max_course_num), self.padding_number)

    def get_overall_state(self, user_embedding, item_embedding):
        """
        该函数计算了每个用户的课程序列，与目标课程的平均余弦相似度以及平均哈达玛积，需要修改成embedding的，可能相当于课程只有一个的特殊情况
        """
        def _mask(i):
            return [True] * i[0] + [False] * (self.max_course_num - i[0])

        origin_prob = torch.reshape(self.origin_rewards[self.batch_index], (-1, 1))  # (batch_size, 1)
        self.num_idx = torch.reshape(self.num_idx, (-1, 1))

        average_sim = self.rank_dot_product_bymatrix(user_embedding, item_embedding)
        element_wise = self.rank_element_wise_bymatrix(user_embedding, item_embedding)
        mask_mat = torch.tensor(np.array(list(map(_mask, torch.reshape(self.num_idx, (self.batch_size, 1))))))
        average_sim = torch.reshape(torch.sum(average_sim * mask_mat, 1), (-1, 1)) / self.num_idx
        mask_mat = torch.tensor(np.repeat(torch.reshape(mask_mat, (self.batch_size, self.max_course_num, 1)), self.embedding_size, 2))  # 这里的repeat，np和torch有没有区别
        element_wise = torch.sum(element_wise * mask_mat, 1) / self.num_idx

        return torch.concatenate((average_sim, element_wise, origin_prob), 1)

    def get_state(self, user_input, item_input):
        # self.origin_prob = torch.reshape(self.origin_rewards[self.batch_index], (-1, 1))  # (batch_size, 1)
        # self.dot_product = self.rank_dot_product(user_input, item_input, step_index) #  计算第step index个物品跟target的余弦相似度
        # self.element_wise_current = self.rank_element_wise(user_input, item_input, step_index) #  计算第step index个物品跟target的点积平均值
        # self.vector_current = item_input[:, step_index]
        # self.vector_target = user_input[item_input]
        # self.vector_current = torch.abs(self.vector_current - self.vector_target)  #  计算第step index个物品跟target的曼哈顿距离
        self.similarity = torch.cosine_similarity(user_input, item_input).reshape(-1,1)
        self.norm_difference = (torch.norm(user_input) - torch.norm(item_input, dim=1)).reshape(-1,1)  # 长度差值很高，很多都是23，会有什么影响？
        self.abs = torch.abs(user_input - item_input)
        self.avg_abs = torch.mean(self.abs, dim=1).reshape(-1, 1)
        return torch.cat((self.similarity, self.norm_difference, self.abs, self.avg_abs), 1)

    def _get_state(self, user_input, item_input, dim):
        user_input = copy.deepcopy(user_input.data)
        item_input = copy.deepcopy(item_input.data)
        revised_vector = copy.deepcopy(item_input)
        revised_vector[:, dim] = 0

        abs_value = torch.abs(item_input[:, dim] - user_input[dim]).reshape(-1, 1) #第k维差值
        k_dim_mean_value = torch.mean(item_input[:, dim]).repeat(len(item_input))
        k_dim_mean_diff = torch.abs(item_input[:, dim] - k_dim_mean_value).reshape(-1, 1) #
        mean_value = torch.mean(item_input, dim=-1).reshape(-1, 1)
        revised_mean = torch.mean(revised_vector, dim=-1).reshape(-1, 1)
        target_mean_value = torch.mean(user_input)
        mean_diff = torch.abs(torch.abs(target_mean_value - mean_value) - torch.abs(target_mean_value - revised_mean))
        std_value = torch.std(item_input, dim=-1).reshape(-1, 1)
        revised_std = torch.std(revised_vector, dim=-1).reshape(-1, 1)
        target_std_value = torch.std(user_input)
        std_diff = torch.abs(torch.abs(target_std_value - std_value) - torch.abs(target_std_value - revised_std))
        similarity = torch.abs(torch.cosine_similarity(user_input, item_input, dim=-1) - torch.cosine_similarity(user_input, revised_vector, dim=-1)).reshape(-1, 1)

        return torch.cat((abs_value, k_dim_mean_diff, mean_diff, std_diff, similarity), 1)

    def rank_element_wise(self, batched_user_input, item_input, step_index):
        self.train_item_ebd = self.course_embedding_user[batched_user_input[:, step_index]]
        self.test_item_ebd = torch.reshape(self.course_embedding_item[item_input], (self.batch_size, self.embedding_size))
        return torch.multiply(self.train_item_ebd, self.test_item_ebd)  # (batch_size, embedding_size)

    def rank_dot_product(self, user_input, item_input, step_index):
        self.train_item_ebd = user_input[step_index]
        self.test_item_ebd = torch.reshape(self.course_embedding_item[item_input], (self.batch_size, self.embedding_size))
        norm_user = torch.sqrt(torch.sum(torch.multiply(self.train_item_ebd, self.train_item_ebd), 1))
        norm_item = torch.sqrt(torch.sum(torch.multiply(self.test_item_ebd, self.test_item_ebd), 1))
        norm = torch.multiply(norm_user, norm_item)
        dot_prod = torch.sum(torch.multiply(self.train_item_ebd, self.test_item_ebd), 1)
        cos_similarity = torch.tensor(np.where(norm != 0, dot_prod / norm, dot_prod)) # possible bug
        return torch.reshape(cos_similarity, (-1, 1))  # (batch_size, 1)

    def rank_element_wise_bymatrix(self, batched_user_input, item_input):  # 计算user的实际交互序列（正样本）中，每一个物品跟target物品的embedding点积，不是很能理解这个state有什么用
        self.train_item_ebd = self.course_embedding_user[
            torch.reshape(batched_user_input, (-1, 1))]  # (batch_size, embedding_size)
        self.test_item_ebd = self.course_embedding_item[
            torch.reshape(torch.tile(item_input, (1, self.max_course_num)), (-1, 1))]  # (batch_size, embedding_size)
        return torch.reshape(torch.multiply(self.train_item_ebd, self.test_item_ebd),
                          (-1, self.max_course_num, self.embedding_size))  # (batch_size, embedding_size)

    def rank_dot_product_bymatrix(self, batched_user_input, item_input):  # 计算user的实际交互序列（正样本）中，每一个物品跟target物品的embedding相似度
        self.train_item_ebd = self.course_embedding_user[
            torch.reshape(batched_user_input, (-1,))]  # (batch_size, embedding_size)
        self.test_item_ebd = self.course_embedding_item[
            torch.reshape(torch.tile(item_input, (1, self.max_course_num)), (-1,))]  # (batch_size, embedding_size)
        # print self.train_item_ebd.shape, self.test_item_ebd.shape
        norm_user = torch.sqrt(torch.sum(torch.multiply(self.train_item_ebd, self.train_item_ebd), 1))
        norm_item = torch.sqrt(torch.sum(torch.multiply(self.test_item_ebd, self.test_item_ebd), 1))
        norm = torch.multiply(norm_user, norm_item)
        dot_prod = torch.sum(torch.multiply(self.train_item_ebd, self.test_item_ebd), 1)
        cos_similarity = torch.tensor(np.where(norm != 0, dot_prod / norm, dot_prod))
        return torch.reshape(cos_similarity, (-1, self.max_course_num))  # (batch_size, 1)

    def update_state(self, low_action, low_state, low_prob, dim):
        self.action_matrix[:, dim] = low_action.squeeze()
        self.state_matrix[:, dim] = low_state.squeeze()
        self.prob_matrix.append(low_prob)

        # self.num_selected = self.num_selected + low_action
        # self.vector_sum = self.vector_sum + torch.multiply(torch.reshape(low_action, (-1, 1)), self.vector_current)
        # self.element_wise_sum = self.element_wise_sum + torch.multiply(torch.reshape(low_action, (-1, 1)),
        #                                                             self.element_wise_current)
        # self.dot_product_sum = self.dot_product_sum + torch.multiply(torch.reshape(low_action, (-1, 1)), self.dot_product)
        # num_selected_array = torch.reshape(self.num_selected, (-1, 1))
        # self.element_wise_mean = torch.tensor(np.where(num_selected_array != 0, self.element_wise_sum / num_selected_array,
        #                                   self.element_wise_sum))
        # self.vector_mean = torch.tensor(np.where(num_selected_array != 0, self.vector_sum / num_selected_array, self.vector_sum))
        # self.dot_product_mean = torch.tensor(np.where(num_selected_array != 0, self.dot_product_sum / num_selected_array,
        #                                  self.dot_product_sum))

    def get_action_matrix(self):
        return self.action_matrix

    def get_state_matrix(self):
        return self.state_matrix

    def get_selected_courses(self, input_matrix):
        revised_matrix = torch.where(self.action_matrix > 0, input_matrix, self.action_matrix)
        return revised_matrix

    def get_reward(self, recommender, batch_index, high_actions, selected_user_input, batched_num_idx,
                   batched_item_input, batched_label_input):
        batch_size = selected_user_input.shape[0]

        # difference between likelihood
        loglikelihood = recommender.get_reward(selected_user_input, torch.reshape(self.num_selected, (-1, 1)),
                                               batched_item_input, batched_label_input)
        old_likelihood = self.origin_rewards[batch_index]
        likelihood_diff = loglikelihood - old_likelihood
        likelihood_diff = torch.where(high_actions == 1, likelihood_diff, torch.zeros(batch_size))

        # difference between average dot_product
        dot_product = self.rank_dot_product_bymatrix(selected_user_input, batched_item_input)
        # print dot_product
        new_dot_product = torch.sum(torch.multiply(dot_product, self.action_matrix), 1) / self.num_selected
        old_dot_product = torch.sum(dot_product, 1) / batched_num_idx

        dot_product_diff = new_dot_product - old_dot_product
        reward1 = likelihood_diff + self.gamma * dot_product_diff

        return reward1, old_dot_product, likelihood_diff
