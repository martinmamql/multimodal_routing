#import torch.nn as nn
#import torch.nn.functional as F
#
#from torch.nn.modules.loss import _Loss
#import torch
#
#def squash(s):
#    s_norm = s.norm(dim=-1, keepdim=True)
#    v = s_norm / (1. + s_norm**2) * s
#    return v    
#
#class SpreadLoss(_Loss):
#
#    def __init__(self, m_min=0.2, m_max=0.9):
#        super(SpreadLoss, self).__init__()
#        self.m_min = m_min
#        self.m_max = m_max
# 
#    def forward(self, x, target, r):
#        b, E = x.shape
#        margin = self.m_min + (self.m_max - self.m_min)*r
#
#        at = torch.FloatTensor(b).fill_(0)
#        at = at.type_as(x)
#        for i, lb in enumerate(target):
#            at[i] = x[i][lb]
#        at = at.view(b, 1).repeat(1, E)
#
#        zeros = x.new_zeros(x.shape)
#        loss = torch.max(margin - (at - x), zeros)
#        loss = loss**2
#        loss = loss.sum() / b - margin**2
#
#        return loss
#
## Defining loss function
#def def_loss_func(loss_type):
#    def loss_func(T, v, lambda_param=0.5, m_plus=0.9, m_minus=0.1):     
#        v_norm = v.norm(dim=2, keepdim=False)
#        return (T*F.relu(m_plus - v_norm)**2 + \
#                lambda_param * (1-T)*F.relu(v_norm - m_minus)**2).sum(1).mean() 
#    if loss_type == 'Capsule_Loss':
#        return loss_func
#    elif loss_type == 'CrossEntropy':
#        return nn.CrossEntropyLoss()
#    elif loss_type == 'SpreadLoss':
#        return SpreadLoss()
#    elif loss_type == 'BCEWithLogitsLoss':
#        return nn.BCEWithLogitsLoss()
