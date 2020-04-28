from src import capsule_layers, capsule_utils
import torch.nn as nn
import torch.nn.functional as F
import torch
from src.multimodal_transformer import mult_model
# Capsule model
class CapsModel(nn.Module):
    def __init__(self,
                 dataset,
                 backbone,
                 loss_type,
                 act_type,
                 num_routing,
                 dp,
                 layer_norm,
                 d_mult,
                 num_heads,
                 transformer_layers,
                 self_transformer_layers,
                 multimodal_transformer_layer,
                 attn_dropout,
                 attn_dropout_a,
                 attn_dropout_v,
                 relu_dropout,
                 res_dropout,
                 out_dropout,
                 embed_dropout,
                 pc_dim,
                 mc_caps_dim,
                 dim_pose_to_vote):
        
        super(CapsModel, self).__init__()
        #### Parameters
        self.act_type = act_type
        self.loss_type = loss_type
        self.d_mult = d_mult
        self.multimodal_transformer_layer = multimodal_transformer_layer
        if self.multimodal_transformer_layer:
            ## Multimodal Transformer Layer
            self.mult = mult_model.MULTModel(orig_d_l=300,
                                         orig_d_a=74, 
                                         orig_d_v=35, 
                                         d_l=self.d_mult, # different from MulT 
                                         d_a=self.d_mult, # different from MulT
                                         d_v=self.d_mult, # different from MulT
                                         vonly=True, 
                                         aonly=True, 
                                         lonly=True, 
                                         num_heads=num_heads, 
                                         layers=transformer_layers, 
                                         self_layers=self_transformer_layers,
                                         attn_dropout=attn_dropout,
                                         attn_dropout_a=attn_dropout_a, 
                                         attn_dropout_v=attn_dropout_v, 
                                         relu_dropout=relu_dropout, 
                                         res_dropout=res_dropout, 
                                         out_dropout=out_dropout, 
                                         embed_dropout=embed_dropout, 
                                         attn_mask=True 
            ) 
            # self.t_in_dim = self.v_in_dim = self.a_in_dim = self.d_mult * 2
            self.t_in_dim = self.v_in_dim = self.a_in_dim = self.d_mult
        else:
            self.t_in_dim = 300
            self.a_in_dim = 74
            self.v_in_dim = 35

        ## Primary Capsule Layer (pc in the code, a single FNN)
        self.pc_dim = pc_dim # 16, 32, ...
        
        # projection to primary capsules
        self.pc_t = nn.Linear(self.t_in_dim, self.pc_dim + 1)
        self.pc_v = nn.Linear(self.v_in_dim, self.pc_dim + 1)
        self.pc_a = nn.Linear(self.a_in_dim, self.pc_dim + 1) 
        self.pc_ta = nn.Linear(self.a_in_dim, self.pc_dim + 1) 
        self.pc_av = nn.Linear(self.a_in_dim, self.pc_dim + 1) 
        self.pc_vt = nn.Linear(self.a_in_dim, self.pc_dim + 1) 
        self.pc_tav = nn.Linear(self.a_in_dim, self.pc_dim + 1) 
        
        ## Decision Capsule (mc in the code) Layers 
        if dataset == 'mosei_senti':
            self.mc_num_caps = 7 
        elif dataset == 'iemocap':
            self.mc_num_caps = 8
        elif dataset == 'mosei_emo':
            self.mc_num_caps = 6
        else:
            print("currently support mosei senti and iemocap only")
            assert False

        self.mc_caps_dim = mc_caps_dim
        self.mc = capsule_layers.CapsuleFC(in_n_capsules=7,
                in_d_capsules=self.pc_dim,
                out_n_capsules=self.mc_num_caps,
                out_d_capsules=self.mc_caps_dim,
                n_rank=None,
                dp=dp,
                act_type=act_type,
                small_std=not layer_norm,
                dim_pose_to_vote=dim_pose_to_vote
                )
        self.embedding = nn.Parameter(torch.zeros(self.mc_num_caps, self.mc_caps_dim))
        if layer_norm:
            print("Using layer norm.")
        else: # Layer Norm
            print("Not using layer norm")
            self.nonlinear_act = nn.Sequential()
        self.num_routing = num_routing

        self.bias = nn.Parameter(torch.zeros(self.mc_num_caps))

    def forward(self, text, audio, vision):
        #### Forward Pass
        # shape: text: (bs, 50, t_in_dim=300); audio: (bs, 50, a_in_dim=74); vision: (bs, 50, v_in_dim=35)
        if self.multimodal_transformer_layer:
            # Multimodal transformer as feature extractor
            # shape for each extracted feature: (bs, d_mult)
            text, audio, vision, ta, av, vt, tav = self.mult(text, audio, vision)
        # primary capsule: just fnn
        u_t = self.pc_t(text).unsqueeze(1) # shape: (bs, 1, pc_dim+1)
        u_a = self.pc_a(audio).unsqueeze(1) 
        u_v = self.pc_v(vision).unsqueeze(1)
        u_ta = self.pc_ta(ta).unsqueeze(1)
        u_av = self.pc_av(av).unsqueeze(1)
        u_vt = self.pc_vt(vt).unsqueeze(1)
        u_tav = self.pc_tav(tav).unsqueeze(1)

        ## Primary Capsules Input, shape: [bs, 7, pc_dim+1]
        pc_input = torch.cat([u_t, u_a, u_v, u_ta, u_av, u_vt, u_tav], dim=1) # dim_1 is pc_num_caps
        # Pose
        init_capsule_pose = pc_input[:,:,:self.pc_dim]
        # Activation Value
        init_capsule_act = torch.sigmoid(pc_input[:,:,self.pc_dim:])  
        # First routing
        decision_pose, decision_act, _ = self.mc(init_capsule_pose, init_capsule_act, 0)
        # second to last routing
        for n in range(self.num_routing):
            decision_pose, decision_act, routing_coefficient = self.mc(init_capsule_pose, init_capsule_act, n, \
                                decision_pose, decision_act)    
        decision_logits = torch.einsum('bcd, cd ->bc', decision_pose, self.embedding)
        return decision_logits, init_capsule_act.squeeze(-1), routing_coefficient
