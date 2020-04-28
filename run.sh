#!/bin/bash
DATASET_PATH='.'
DATASET='mosei_senti'
CUDA_VISIBLE_DEVICES=0 python3 -u main.py --data-path $DATASET_PATH --aligned --arch multimodal_capsule --epoch 20 --batch-size 32 --lr 0.0001 --weight-decay 1e-4 --clip 1 --act-type ONES --num-routing 1 --dp 0.5 --patience 5 --d_mult 200 --transformer_layers 7 --num_heads 20 --attn_dropout 0.0 --attn_dropout_a 0.0 --attn_dropout_v 0.0 --relu_dropout 0.0 --res_dropout 0.0 --out_dropout 0.0 --embed_dropout 0.0 --pc_dim 64 --mc_caps_dim 64 --dataset $DATASET

