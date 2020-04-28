import torch.nn as nn
import torch.nn.functional as F
import torch
from src.multimodal_transformer import mult_model

class Transformer(nn.Module):
    def __init__(self, early_fusion, d_model, n_head, dim_feedforward, dropout, num_layers, layer_norm, embed_dropout, output_dim, out_dropout, multimodal_transformer=True):
        super(Transformer, self).__init__()
        self.multimodal_transformer = multimodal_transformer
        if self.multimodal_transformer:
            self.d_mult = d_model
            self.mult = mult_model.MULTModel(orig_d_l=300,
                                         orig_d_a=74, 
                                         orig_d_v=35, 
                                         d_l=self.d_mult, # different from MulT 
                                         d_a=self.d_mult, # different from MulT
                                         d_v=self.d_mult, # different from MulT
                                         vonly=True, 
                                         aonly=True, 
                                         lonly=True, 
                                         num_heads=n_head, 
                                         layers=num_layers, 
                                         attn_dropout=0.1,
                                         attn_dropout_a=0.0, 
                                         attn_dropout_v=0.0, 
                                         relu_dropout=0.1, 
                                         res_dropout=0.1, 
                                         out_dropout=out_dropout, 
                                         embed_dropout=embed_dropout, 
                                         attn_mask=True 
            ) 
            self.t_in_dim = self.v_in_dim = self.a_in_dim = self.d_mult * 2
            combined_dim = 6 * d_model

        else:    
            self.early_fusion = early_fusion
            'Only late fusion implemented; early fusion will be implemented later'
            # Late fusion
            if not self.early_fusion:
                self.orig_d_t = 300 
                self.orig_d_a = 74
                self.orig_d_v = 35
                self.d_t = self.d_a = self.d_v = d_model
                # Temporal convolutional layers
                self.proj_t = nn.Conv1d(self.orig_d_t, self.d_t, kernel_size=1, padding=0, bias=False)
                self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
                self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

                # Transformer Layers
                self.encoder_layer_t = nn.TransformerEncoderLayer(d_model, nhead=n_head, dim_feedforward=dim_feedforward)
                self.encoder_layer_a = nn.TransformerEncoderLayer(d_model, nhead=n_head, dim_feedforward=dim_feedforward)
                self.encoder_layer_v = nn.TransformerEncoderLayer(d_model, nhead=n_head, dim_feedforward=dim_feedforward)

                'Remember to implement layer norm option here'
                if layer_norm: 
                    print("layer norm not implemented yet for vanilla transformer")
                    assert False
                else:
                    self.transformer_encoder_t = nn.TransformerEncoder(self.encoder_layer_t, num_layers=num_layers)
                    self.transformer_encoder_a = nn.TransformerEncoder(self.encoder_layer_a, num_layers=num_layers)
                    self.transformer_encoder_v = nn.TransformerEncoder(self.encoder_layer_v, num_layers=num_layers)

                self.embed_dropout = embed_dropout

                'Change here for other dataset since number of modalities might be different'
                combined_dim = 3 * d_model

        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        self.out_dropout = out_dropout



    def forward(self, text, audio, vision):
        if self.multimodal_transformer:
            h_ts, h_as, h_vs = self.mult(text, audio, vision) 
            h_ts, h_as, h_vs = h_ts.transpose(0, 1), h_as.transpose(0, 1), h_vs.transpose(0, 1)
        else:
            if not self.early_fusion:
                text = F.dropout(text.transpose(1, 2), p=self.embed_dropout, training=self.training)
                audio = audio.transpose(1, 2)
                vision = vision.transpose(1, 2)
                proj_x_t = self.proj_t(text)
                proj_x_a = self.proj_a(audio)
                proj_x_v = self.proj_v(vision)

                proj_x_t = proj_x_t.permute(2, 0, 1)
                proj_x_a = proj_x_a.permute(2, 0, 1)
                proj_x_v = proj_x_v.permute(2, 0, 1)

                # Self attentions
                h_ts = self.transformer_encoder_t(proj_x_t)
                h_as = self.transformer_encoder_a(proj_x_a)
                h_vs = self.transformer_encoder_v(proj_x_v)
        last_h_t = h_ts[-1] # take the last output for prediction
        last_h_a = h_as[-1] 
        last_h_v = h_vs[-1] 
        last_hs = torch.cat([last_h_t, last_h_a, last_h_v], dim=1)
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output
            
