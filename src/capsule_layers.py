'''Capsule in PyTorch
TBD
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

#### Capsule Layer ####
class CapsuleFC(nn.Module):
    r"""Applies as a capsule fully-connected layer.
    TBD
    """
    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, n_rank, dp, dim_pose_to_vote, uniform_routing_coefficient=False,\
            act_type='EM', small_std=False):
        super(CapsuleFC, self).__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.n_rank = n_rank
        
        self.weight_init_const = np.sqrt(out_n_capsules/(in_d_capsules*in_n_capsules)) 
        self.w = nn.Parameter(self.weight_init_const* \
                                          torch.randn(in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules))


        self.dropout_rate = dp
        if small_std:
            self.nonlinear_act = nn.Sequential()
        else:
            print("layer norm will destroy interpretability, thus not available")
            assert False
        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1. / (out_d_capsules ** 0.5)
        
        self.act_type = act_type

        if act_type == 'EM':
            self.beta_u = nn.Parameter(torch.randn(out_n_capsules))
            self.beta_a = nn.Parameter(torch.randn(out_n_capsules))
        elif act_type == 'Hubert':
            self.alpha = nn.Parameter(torch.ones(out_n_capsules))

            self.beta = nn.Parameter(np.sqrt(1./(in_n_capsules*in_d_capsules)) * \
                                    torch.randn(in_n_capsules, in_d_capsules, out_n_capsules))
        self.uniform_routing_coefficient = uniform_routing_coefficient
    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, n_rank{}, \
            weight_init_const={}, dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules, self.n_rank,
            self.weight_init_const, self.dropout_rate
        )        
    def forward(self, input, current_act, num_iter, next_capsule_value=None, next_act=None, uniform_routing=False):
        # b: batch size
        # n: num of capsules in current layer
        # a: dim of capsules in current layer
        # r: num of rank in low rank approx.
        # m: num of capsules in next layer
        # d: dim of capsules in next layer

        # input: current_capsule_value, (bs, 7, pc_dim)
        # current_act: (bs, 7, 1)
        # next_capsule_value: (bs, 7, mc_dim)
        # next_act: (bs, 7)

        # vote is w times input
        # query_key: routing coefficient

        current_act = current_act.view(current_act.shape[0], -1)
        w = self.w # 7, 64, 7, 64, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules
        if next_capsule_value is None:
            query_key = torch.zeros(self.in_n_capsules, self.out_n_capsules).type_as(input) # 7, 7; in_n_capsules, out_n_capsules
            query_key = F.softmax(query_key, dim=1) # turns into proability
            next_capsule_value = torch.einsum('nm, bna, namd->bmd', query_key, input, w)
        else:
            if uniform_routing:
                query_key = torch.zeros(input.shape[0], self.in_n_capsules, self.out_n_capsules).type_as(input) # 7, 7; in_n_capsules, out_n_capsules
                _query_key = torch.zeros(input.shape[0], self.in_n_capsules, self.out_n_capsules).type_as(input) # 7, 7; in_n_capsules, out_n_capsules
                query_key = F.softmax(query_key, dim=1) # turns into proability
            else:
                _query_key = torch.einsum('bna, namd, bmd->bnm', input, w, next_capsule_value)
                _query_key.mul_(self.scale)
                query_key = F.softmax(_query_key, dim=2)
                query_key = torch.einsum('bnm, bm->bnm', query_key, next_act)
                query_key = query_key / (torch.sum(query_key, dim=2, keepdim=True) + 1e-10)
            next_capsule_value = torch.einsum('bnm, bna, namd, bn->bmd', query_key, input, 
                                                  w, current_act)
        if self.act_type == 'ONES':
            next_act = torch.ones(next_capsule_value.shape[0:2]).type_as(next_capsule_value)
        next_capsule_value = self.drop(next_capsule_value)
        if not next_capsule_value.shape[-1] == 1:
            next_capsule_value = self.nonlinear_act(next_capsule_value)
        return next_capsule_value, next_act, query_key

