# encoding: utf-8
import argparse
from utils import *

from CIFAR10_models import *
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
import copy
import os
import numpy
import argparse
import logging
from StrategyNet import *
from torch.nn.utils import clip_grad_norm_
logger = logging.getLogger(__name__)
CUDA_LAUNCH_BLOCKING = 1
from tensorboardX import SummaryWriter
from collections import OrderedDict
from collections import Counter


def _get_sub_policies(attack_id_list, espilon_id_list, attack_iters_id_list, step_size_id_list, prob_id_list, args):
    policies = []
    attack_id_list = attack_id_list[0].cpu().numpy()
    espilon_id_list = espilon_id_list[0].cpu().numpy()
    attack_iters_id_list = attack_iters_id_list[0].cpu().numpy()
    step_size_id_list = step_size_id_list[0].cpu().numpy()

    for n in range(args.subpolicy_num):
        sub_policy = {}
        for i in range(args.op_num_pre_subpolicy):
            all_policy = {}
            # print(n+i)
            # print(args.epsilon_types)
            # print(espilon_id_list[n+i])
            all_policy['attack'] = args.attack_types[attack_id_list[n + i]]
            all_policy['epsilon'] = args.epsilon_types[espilon_id_list[n + i]]
            all_policy['attack_iters'] = args.attack_iters_types[attack_iters_id_list[n + i]]
            all_policy['step_size'] = args.step_size_types[step_size_id_list[n + i]]

            sub_policy[i] = all_policy
        policies.append(sub_policy)
    return policies


def _get_all_policies(attack_id_list, espilon_id_list, attack_iters_id_list, step_size_id_list, args):
    policies = []
    # print(attack_id_list)
    attack_id_list = attack_id_list[0].cpu().numpy()
    espilon_id_list = espilon_id_list[0].cpu().numpy()
    attack_iters_id_list = attack_iters_id_list[0].cpu().numpy()
    step_size_id_list = step_size_id_list[0].cpu().numpy()
    # prob_id_list=prob_id_list[0].cpu().numpy()
    for n in range(len(attack_id_list)):
        sub_policy = {}

        all_policy = {}
        # print(n+i)
        # print(args.epsilon_types)
        # print(espilon_id_list[n+i])
        all_policy['attack'] = args.attack_types[attack_id_list[n]]
        all_policy['epsilon'] = args.epsilon_types[espilon_id_list[n]]

        all_policy['attack_iters'] = args.attack_iters_types[attack_iters_id_list[n]]

        all_policy['step_size'] = args.step_size_types[step_size_id_list[n]]
        # all_policy['prob'] = args.prob_types[prob_id_list[n]]
        sub_policy[n] = all_policy
        policies.append(sub_policy)

    return policies


def select_action(policy_model, state):
    outputs = policy_model(state)
    attack_id_list = []
    espilon_id_list = []
    attack_iters_id_list = []
    step_size_id_list = []
    prob_list = []
    action_list = []

    max_attack_id_list = []
    max_espilon_id_list = []
    max_attack_iters_id_list = []
    max_step_size_id_list = []
    # max_prob_list = []
    # max_action_list = []
    temp_saved_log_probs = []
    for id in range(4):

        logits = outputs[id]
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.data.clone()
        m = Categorical(probs)

        prob_list.append(m)
        action = m.sample()

        max_action = max_probs.max(1)[1]
        # print(action.shape)
        mode = id % 5
        if mode == 0:
            attack_id_list.append(action)
            max_attack_id_list.append(max_action)
        elif mode == 1:
            espilon_id_list.append(action)
            max_espilon_id_list.append(max_action)
        elif mode == 2:
            attack_iters_id_list.append(action)
            max_attack_iters_id_list.append(max_action)
        elif mode == 3:
            step_size_id_list.append(action)
            max_step_size_id_list.append(max_action)
        temp_saved_log_probs.append(m.log_prob(action))
    # policy_model.saved_log_probs.append(temp_saved_log_probs)
    curpolicy = _get_all_policies(attack_id_list, espilon_id_list, attack_iters_id_list, step_size_id_list, args)
    max_curpolicy = _get_all_policies(max_attack_id_list, max_espilon_id_list, max_attack_iters_id_list,
                                      max_step_size_id_list, args)
    action_list.append(attack_id_list)
    action_list.append(espilon_id_list)
    action_list.append(attack_iters_id_list)
    action_list.append(step_size_id_list)

    return action_list, curpolicy, prob_list, max_curpolicy


def get_args():
    parser = argparse.ArgumentParser('LAS_AT')
    # target model
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../dataset', type=str)
    parser.add_argument('--out-dir', default='LAS_AT', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--target_model_lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--target_model_lr_scheduler', default='multistep', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--target_model_lr_min', default=0., type=float)
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model', default='PreActResNest18', type=str, help='model name')

    parser.add_argument('--path', default='LAS_AT', type=str, help='model name')

    ## search
    parser.add_argument('--attack_types', type=list, default=['IFGSM'], help='all searched policies')
    parser.add_argument('--epsilon_types', type=int, nargs="*", default=range(3, 15))
    parser.add_argument('--attack_iters_types', type=int, nargs="*", default=range(3, 14))
    parser.add_argument('--step_size_types', type=int, nargs="*", default=range(1, 5))

    ## policy Hyperparameters
    parser.add_argument('--policy_model_lr', type=float, default=0.0001)
    parser.add_argument('--policy_model_lr_scheduler', default='multistep', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--policy_model_lr_min', default=0., type=float)
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')

    parser.add_argument('--interval_num', type=int, default=20)
    parser.add_argument('--exp_iter', type=int, default=1)

    parser.add_argument('--tensor-path', default='runs', type=str, help='tensorboardX name')

    parser.add_argument('--policy_optimizer', default='SGD_without_momentum', type=str, help='policy_optimizer')
    parser.add_argument('--factor', default=0.7, type=float, help='Label Smoothing')

    parser.add_argument('--a', default=1, type=float)
    parser.add_argument('--b', default=1, type=float)
    parser.add_argument('--c', default=1, type=float)

    parser.add_argument('--R2_param', default=4, type=float)
    parser.add_argument('--R3_param', default=2, type=float)
    parser.add_argument('--clip_grad_norm', default=1.0, type=float)

    arguments = parser.parse_args()
    return arguments


def strategy_counter(actions):
    epsilon = actions[1][0].cpu().numpy().tolist()
    attack_iter = actions[2][0].cpu().numpy().tolist()
    step_size = actions[3][0].cpu().numpy().tolist()
    step_size_list = []
    epsilon_list = []
    attack_iters_list = []

    for n in range(len(epsilon)):
        epsilon_list.append(args.epsilon_types[epsilon[n]])
        attack_iters_list.append(args.attack_iters_types[attack_iter[n]])
        step_size_list.append(args.step_size_types[step_size[n]])

    epsilon = Counter(epsilon_list)
    attack_iter = Counter(attack_iters_list)
    step_size = Counter(step_size_list)
    return epsilon, attack_iter, step_size

args = get_args()
out_dir = os.path.join(args.out_dir, 'model_' + args.model)
out_dir = os.path.join(out_dir, 'epochs_' + str(args.epochs))

out_dir = os.path.join(out_dir, 'epsilon_types_' + str(min(args.epsilon_types)) + '_' + str(max(args.epsilon_types)))
out_dir = os.path.join(out_dir, 'attack_iters_types_' + str(min(args.attack_iters_types)) + '_' + str(
    max(args.attack_iters_types)))
out_dir = os.path.join(out_dir,
                       'step_size_types_' + str(min(args.step_size_types)) + '_' + str(max(args.step_size_types)))
train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
for current_epoch in ['0', '20', '40', '60', '80', '100', '120']:
    policy_model_path = os.path.join(out_dir, f'{current_epoch}_policy_model_ckpt.t7')
    policy_model_checkpoint = torch.load(policy_model_path)
    Strategy_model = ResNet18_Strategy(args)
    try:
        Strategy_model.load_state_dict(policy_model_checkpoint['net'])
    except:
        new_state_dict = OrderedDict()
        for k, v in policy_model_checkpoint['net'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        Strategy_model.load_state_dict(new_state_dict, False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    epsilon_counter = Counter()
    attack_iter_counter = Counter()
    step_size_counter = Counter()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        Strategy_model.to(device)
        Strategy_model.eval()
        action_list, policy_outputs, policy_prob, max_policy_outputs = select_action(Strategy_model, inputs)
        a, b, c = strategy_counter(action_list)
        epsilon_counter += a
        attack_iter_counter += b
        step_size_counter += c
    print(f'{current_epoch} epochs epsilon：', epsilon_counter)
    print(f'{current_epoch} epochs attack_iter：', attack_iter_counter)
    print(f'{current_epoch} epochs step_size：', step_size_counter)
    print('\n\n')
