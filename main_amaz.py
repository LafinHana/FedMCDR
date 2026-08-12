import copy
import random
import numpy as np
import pandas as pd
import torch
import os
import json
import importlib
from sort import sort_training_log
import math
import argparse
import warnings
import datetime
import utility

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='args for fedgcdr')
parser.add_argument('--dataset', choices=['amazon', 'douban'], default='amazon')
parser.add_argument('--round_gat', type=int, default=200)
parser.add_argument('--round_ft', type=int, default=300)
parser.add_argument('--num_domain', type=int, default=7)
parser.add_argument('--device', type=str, default='cuda:3')
parser.add_argument('--target_domain', type=int, default=0)
parser.add_argument('--lr_mf', type=float, default=0.005)
parser.add_argument('--lr_gat', type=float, default=0.001)
parser.add_argument('--embedding_size', type=int, default=8)
parser.add_argument('--local_epoch', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=1)
parser.add_argument('--num_negative', type=int, default=4)
parser.add_argument('--user_batch', type=int, default=16)
parser.add_argument('--model', type=str, default='fedmcdr')
parser.add_argument('--random_seed', type=int, default=7)
parser.add_argument('--weight_decay_k', type=float, default=0.1)
parser.add_argument('--knowledge', type=int, default=0)
parser.add_argument('--only_ft', type=int, default=0)
parser.add_argument('--eps', type=float, default=8)
parser.add_argument('--dp', type=bool, default=True)
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--num_users', type=int, default=55518)
parser.add_argument('--zdes', type=str, default='no weight decay use clip')
parser.add_argument('--full_overlap', type=bool, default=True)
parser.add_argument('--num_mlp', type=int, default=0)
# plus parameters
parser.add_argument('--only_target', type=int, default=0)  # used for training an environment
parser.add_argument('--train_agent', type=int, default=0)
# parser.add_argument('--use_agent', type=int, default=0)
parser.add_argument('--agent_weight_size', type=int, default=4)
parser.add_argument('--sigma', type=float, default=4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--lr_agent', type=float, default=0.001)
parser.add_argument('--random_rate', type=float, default=0.5)  # fixed threshold for action
parser.add_argument('--embedding_file', type=str, default=None)

#rebuttal

parser.add_argument('--miu', type=int, default=0) # 均值实验

parser.add_argument('--mix_noise', type=int, default=0) # 混合噪声实验

parser.add_argument('--mask', type=int, default=0) # partial_overlap实验

parser.add_argument('--simulate', type=int, default=1) # 工业场景实验

args = parser.parse_args()

Server = importlib.import_module('model.' + args.model + '.party').Server
Client = importlib.import_module('model.' + args.model + '.party').Client
MLP = importlib.import_module('model.' + args.model + '.model').MLP

# random seed
seed = args.random_seed
random.seed(17 + seed)
np.random.seed(707 + seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(2001 + seed)
torch.cuda.manual_seed_all(70506 + seed)

device = torch.device(args.device)

now = datetime.datetime.now()
formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S").replace(' ', '_').replace(':', "_")
formatted_date = now.strftime("%Y-%m-%d")
output_file = 'output/' + args.model + '/' + formatted_date_time + '_tar_' + str(
    args.target_domain) + '_num_' + str(args.num_domain) + '_local_'+str(args.local_epoch)+'_s_'+str(args.sigma)+'_w_'+str(args.weight_decay)+'_r_'+str(args.random_rate)+ '_' + str(
    args.random_seed) + '.out'

with open(output_file, 'w') as f:
    f.write(str(args) + '\n')
print(args)

domain_user, dic, domain_names = utility.set_dataset(args)
client_train_data, server_evaluate_data, num_items, num_users, user_dic = dic['client_train_data'], dic[
    'server_evaluate_data'], dic['num_items'], dic['num_users'], dic['user_dic']
clients = [Client(i, client_train_data[i], num_items, 0, domain_names, args) for i in range(args.num_users)]
server = [
    Server(i, domain_names[i], num_items[i], clients, domain_user[domain_names[i]], server_evaluate_data[i], user_dic,
           args) for i in range(args.num_domain)]
MLPs = [MLP(args.embedding_size).to(device) for _ in range(args.num_domain-args.num_mlp)]

# eval pre-train model
for it in server:
    it.test_mf(0, output_file)

tar_domain = args.target_domain
k_dic, emb_dic, save_dic = {}, {}, {}

# load knowledge_hr
if args.knowledge:
    path = 'knowledge_hr/' + str(args.num_domain) + 'domains.json'
    if args.full_overlap:
        if args.dataset == 'douban':
            path = 'knowledge_hr/full_overlap/douban.json'
        else:
            if args.mask:
                path = 'knowledge_hr/full_overlap/mask_' + str(args.num_domain) + 'domains.json'
                print("partial scene")
            if args.simulate:
                path = 'knowledge_hr/full_overlap/sim_' + str(args.num_domain) + 'domains.json'
                print("simulate scene")
            else:
                path = 'knowledge_hr/full_overlap/' + str(args.num_domain) + 'domains.json'
    with open(path, 'r') as f:
        k_dic = json.load(f)
    for i in range(args.num_users):
        clients[i].knowledge = k_dic[str(i)]
else:
    if args.only_target == 0:
        order = [i for i in range(args.num_domain)]
    else:
        order = [args.target_domain]
    for it in order:
        max_hr, max_ndcg, epoch_id, no_improve = 0, 0, 0, 0
        knowledge = [1] * args.num_users
        for i in range(args.round_gat):
            print(f'{server[it].domain_name} gat round {i}: ' + formatted_date_time)
            server[it].kt_stage()
            hr_5, ndcg_5, hr_10, ndcg_10 = server[it].test_gat(i, output_file)
            with open(output_file, 'a') as f:
                f.write(
                    f'[{server[it].domain_name} GAT Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = {hr_10:.4f},'
                    f' ndcg_10 = {ndcg_10:.4f}\n')
            print(
                f'[{server[it].domain_name} GAT Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = {hr_10:.4f},'
                f' ndcg_10 = {ndcg_10:.4f}\n')
            if hr_10 > max_hr or (hr_10 == max_hr and ndcg_10 > max_ndcg):
                no_improve = 0
                epoch_id = i
                max_hr = hr_10
                max_ndcg = ndcg_10
                # if it == tar_domain and args.knowledge_hr:
                for client in clients:
                    knowledge[client.id] = client.knowledge[it]
                save_dic[domain_names[tar_domain]] = [server[tar_domain].U.data.tolist(),
                                                      server[tar_domain].V.data.tolist()]
            else:
                no_improve += 1
            # if no_improve > 100:
            #     break
        for client in clients:
            client.knowledge[it] = knowledge[client.id]

    # save_knowledge
    for i in range(args.num_users):
        for kl in clients[i].knowledge:
            if len(kl) != 0:
                kl[0] = kl[0].tolist()
        k_dic[i] = clients[i].knowledge

    if args.only_target:
        if args.full_overlap:
            if args.dataset == 'douban':
                path = 'GAT/douban/' + str(args.num_domain)
                if not os.path.exists(path):
                    os.mkdir(path)
                with open(path + '/'+str(args.target_domain)+'embeddings' + '_' + str(formatted_date_time) + '_' + str(args.target_domain) + '.json', 'w') as f:
                    json.dump(save_dic, f)
                torch.save(server[tar_domain].item_gat, path + '/'+str(args.target_domain)+'model' + '_' + str(formatted_date_time) + '_' + str(args.target_domain) + '.pt')
            else:
                path = 'GAT/full_overlap/' + str(args.num_domain)
                if not os.path.exists(path):
                    os.mkdir(path)
                with open(path + '/'+str(args.target_domain)+'embeddings'+ '_' + str(formatted_date_time)+'.json', 'w') as f:
                    json.dump(save_dic, f)
                torch.save(server[tar_domain].item_gat, path + '/'+str(args.target_domain)+'model'+ '_' + str(formatted_date_time)+'.pt')
        else:
            path = 'GAT/' + str(args.num_domain)
            if not os.path.exists(path):
                os.mkdir(path)
            with open(path + '/'+str(args.target_domain)+'embeddings' + '_' + str(formatted_date_time) + '.json', 'w') as f:
                json.dump(save_dic, f)
            torch.save(server[tar_domain].item_gat, path + '/'+str(args.target_domain)+'model' + '_' + str(formatted_date_time) + '.pt')
    else:

        path = 'knowledge_hr/' + str(args.num_domain) + 'domains' + '_' + formatted_date + '.json'
        if args.dataset == 'douban':
            path = 'knowledge_hr/douban.json'
        if args.full_overlap:
            path = 'knowledge_hr/full_overlap/' + str(args.num_domain) + 'domains' + '_' + formatted_date + '.json'
            if args.dataset == 'douban':
                path = 'knowledge_hr/full_overlap/douban.json'
        with open(path, 'w') as f:
            json.dump(k_dic, f)
        print('finish knowledge extract')
server[tar_domain].mlp = MLPs


if args.train_agent:
    if args.full_overlap:
        if args.knowledge == 0 and args.only_target == 1:
            if args.dataset == 'amazon':
                with open('GAT/full_overlap/' + str(args.num_domain) + '/'+str(args.target_domain)+'embeddings'+ '_' + str(formatted_date_time) +'.json', 'r') as f:
                    save_dic = json.load(f)
                temp_gat = torch.load('GAT/full_overlap/' + str(args.num_domain) + '/'+str(args.target_domain)+'model'+ '_' + str(formatted_date_time) +'.pt', map_location=args.device)
            else:
                with open('GAT/douban/' + str(args.num_domain) + '/'+str(args.target_domain)+'embeddings' + '_' + str(
                        formatted_date_time) + '_' + str(args.target_domain) + '.json', 'r') as f:
                    save_dic = json.load(f)
                temp_gat = torch.load(
                    'GAT/douban/' + str(args.num_domain) + '/'+str(args.target_domain)+'model' + '_' + str(formatted_date_time) + '_' + str(args.target_domain) + '.pt',
                    map_location=args.device)
        else:
            if args.dataset == 'amazon':
                with open('GAT/full_overlap/' + str(args.num_domain) + '/'+str(args.target_domain)+'embeddings.json', 'r') as f:
                    save_dic = json.load(f)
                temp_gat = torch.load('GAT/full_overlap/' + str(args.num_domain) + '/'+str(args.target_domain)+'model.pt', map_location=args.device)
            else:
                with open('GAT/douban/' + str(args.num_domain) + '/'+str(args.target_domain)+'embeddings'+ '_' + str(args.target_domain) + '.json', 'r') as f:
                    save_dic = json.load(f)
                temp_gat = torch.load('GAT/douban/' + str(args.num_domain) + '/'+str(args.target_domain)+'model'+ '_' + str(args.target_domain) + '.pt', map_location=args.device)
    else:
        with open('GAT/' + str(args.num_domain) + '/'+str(args.target_domain)+'embeddings.json', 'r') as f:
            save_dic = json.load(f)
        temp_gat = torch.load('GAT/' + str(args.num_domain) + '/'+str(args.target_domain)+'model.pt', map_location=args.device)
    server[tar_domain].U = torch.tensor(save_dic[domain_names[tar_domain]][0], device=args.device)
    server[tar_domain].V = torch.tensor(save_dic[domain_names[tar_domain]][1], device=args.device)

    server[tar_domain].item_gat.in2hidden = copy.deepcopy(temp_gat.in2hidden)
    server[tar_domain].item_gat.hidden2out = copy.deepcopy(temp_gat.hidden2out)
    server[tar_domain].distribute_model()
    max_hr, max_ndcg, epoch_id, no_improve = 0, 0, 0, 0
    for i in range(args.round_gat):
        print(f'{server[args.target_domain].domain_name} agent round {i}: ' + formatted_date_time)
        server[args.target_domain].train_agent()
        hr_5, ndcg_5, hr_10, ndcg_10 = server[args.target_domain].test_agent(i, output_file)
        with open(output_file, 'a') as f:
            f.write(
                f'[{server[args.target_domain].domain_name} agent Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = {hr_10:.4f},'
                f' ndcg_10 = {ndcg_10:.4f}\n')
        print(
            f'[{server[args.target_domain].domain_name} agent Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = {hr_10:.4f},'
            f' ndcg_10 = {ndcg_10:.4f}\n')
        if hr_10 > max_hr or (hr_10 == max_hr and ndcg_10 > max_ndcg):
            no_improve = 0
            epoch_id = i
            max_hr = hr_10
            max_ndcg = ndcg_10
            if args.dataset == 'amazon':
                torch.save(server[tar_domain].agent.low_model, 'Agent/' + str(args.num_domain) + '/'+str(args.target_domain)+'model'+'_'+str(formatted_date_time)+'.pt')
            else:
                torch.save(server[tar_domain].agent.low_model,
                           'Agent/douban/'+str(args.target_domain)+'model' + '_' + str(formatted_date_time) + '.pt')




# ASYNC
if args.only_ft == 0:
    max_hr, max_ndcg, epoch_id, no_improve = 0, 0, 0, 0
    if args.train_agent == 1:
        if args.dataset == 'amazon':
            low_model = torch.load('Agent/' + str(args.num_domain) + '/'+str(args.target_domain)+'model'+'_'+str(formatted_date_time)+'.pt',
                               map_location=args.device)
        else:
            low_model = torch.load(
                'Agent/douban/' + str(args.target_domain) + 'model' + '_' + str(
                    formatted_date_time) + '.pt',
                map_location=args.device)
    else:
        if args.dataset == 'amazon':
            low_model = torch.load('Agent/'+str(args.target_domain)+'model.pt', map_location=args.device)
        else:
            low_model = torch.load('Agent/douban/model_'+str(args.target_domain)+'.pt', map_location=args.device)
    for para in low_model.parameters():
        para.requires_grad = False
    if server[tar_domain].agent is not None:
        server[tar_domain].agent.low_model = low_model
    for client in clients:
        client.prepare_transfer(tar_domain, low_model)  # hr需要写一下这个函数

    for i in range(args.round_gat):
        print(f'{server[tar_domain].domain_name} gat round {i}: ' + formatted_date_time + f' [{args.zdes}]')
        # if server[tar_domain].id == tar_domain and args.knowledge_hr:
        try:
            server[tar_domain].kt_stage(True)
            hr_5, ndcg_5, hr_10, ndcg_10 = server[tar_domain].test_gat(i, output_file)
        except Exception as e:
            print(e)
            emb_dic['parser'] = vars(args)
            if args.full_overlap:
                with open('embedding/' + args.model + '/full_overlap/' + str(args.num_domain) + 'dp' + str(
                        args.dp) + '_' + args.dataset + '_' +
                          domain_names[tar_domain] + '_' + str(args.gamma) + '_' + str(args.sigma) + '.json', 'w') as f:
                    json.dump(emb_dic, f)
            else:
                with open('embedding/' + args.model + '/' + str(args.num_domain) + 'dp' + str(
                        args.dp) + '_' + args.dataset + '_' +
                          domain_names[tar_domain] + '_' + str(args.gamma) + '_' + str(args.sigma) + '.json', 'w') as f:
                    json.dump(emb_dic, f)


        with open(output_file, 'a') as f:
            f.write(
                f'[{server[tar_domain].domain_name} GAT Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = {hr_10:.4f},'
                f' ndcg_10 = {ndcg_10:.4f}\n')
        print(
            f'[{server[tar_domain].domain_name} GAT Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = {hr_10:.4f},'
            f' ndcg_10 = {ndcg_10:.4f}\n')
        if hr_10 > max_hr or (hr_10 == max_hr and ndcg_10 > max_ndcg):
            no_improve = 0
            epoch_id = i
            max_hr = hr_10
            max_ndcg = ndcg_10
            # if tar_domain == tar_domain and args.knowledge_hr:
            emb_dic[domain_names[tar_domain]] = [server[tar_domain].user_embedding_with_attention.data.tolist(),
                                                 server[tar_domain].V.data.tolist()]
        else:
            no_improve += 1
        # if no_improve > 100:
        #     break
    failed_count = sort_training_log(output_file)
    print(f"Sorting complete. {failed_count} line(s) failed and appended at the end.")

    server[tar_domain].U = torch.tensor(emb_dic[domain_names[tar_domain]][0], device=args.device)
    server[tar_domain].V = torch.tensor(emb_dic[domain_names[tar_domain]][1], device=args.device)
    emb_dic['parser'] = vars(args)
    if args.full_overlap:
        with open('embedding/' + args.model + '/full_overlap/' + str(args.num_domain) + 'dp' + str(
                args.dp) + '_' + args.dataset + '_' +
                  domain_names[tar_domain] + '_' + str(args.gamma) + '_' + str(args.sigma) + '.json', 'w') as f:
            json.dump(emb_dic, f)
    else:
        with open('embedding/' + args.model + '/' + str(args.num_domain) + 'dp' + str(args.dp) + '_' + args.dataset +
                  '_' + domain_names[tar_domain] + '_' + str(args.gamma) + '_' + str(args.sigma) + '.json', 'w') as f:
            json.dump(emb_dic, f)
# load embedding
else:
    if args.embedding_file is not None:
        file_name = args.embedding_file
    else:
        file_name = 'embedding/' + args.model + '/' + str(args.num_domain) + 'dp' + str(args.dp) + '_' + args.dataset +\
                    '_' + domain_names[tar_domain] + '_' + str(args.gamma) + '.json'
    with open(file_name, 'r') as f:
        dic = json.load(f)
        tar_name = domain_names[args.target_domain]
        server[tar_domain].U.data, server[tar_domain].V.data = torch.tensor(dic[tar_name][0], device=args.device), \
            torch.tensor(dic[tar_name][1], device=args.device)

max_hr, max_ndcg, epoch_id, no_improve = 0, 0, 0, 0
max_hr_5, max_hr_10, max_ndcg_5, max_ndcg_10 = 0, 0, 0, 0
for i in range(args.round_ft):
    print(f'{server[tar_domain].domain_name} fine-tuning round {i} ' + formatted_date_time)
    server[tar_domain].mf_train()
    hr_5, ndcg_5, hr_10, ndcg_10 = server[tar_domain].test_mf(i, output_file)
    with open(output_file, 'a') as f:
        f.write(f'[{server[tar_domain].domain_name} Fine-tuning Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, '
                f'hr_10 ={hr_10:.4f}, ndcg_10 = {ndcg_10:.4f}\n')
    print(f'[{server[tar_domain].domain_name} Fine-tuning Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = '
          f'{hr_10:.4f}, ndcg_10 = {ndcg_10:.4f}\n')
    max_hr_5 = max(max_hr_5, hr_5)
    max_hr_10 = max(max_hr_10, hr_10)
    max_ndcg_5 = max(max_ndcg_5, ndcg_5)
    max_ndcg_10 = max(max_ndcg_10, ndcg_10)
    if hr_10 > max_hr or (hr_10 == max_hr and ndcg_10 > max_ndcg):
        no_improve = 0
        epoch_id = i
        max_hr = hr_10
        max_ndcg = ndcg_10
    else:
        no_improve += 1

with open(output_file, 'a') as f:
    f.write(str(epoch_id) + '\n')
    f.write(f'hr_5 = {max_hr_5}, ndcg_5 = {max_ndcg_5}, hr_10 = {max_hr_10}, ndcg_10 = {max_ndcg_10}')
print(epoch_id)
print(f'hr_5 = {max_hr_5}, ndcg_5 = {max_ndcg_5}, hr_10 = {max_hr_10}, ndcg_10 = {max_ndcg_10}')
print(f'save to {output_file}')
