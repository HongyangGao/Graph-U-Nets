#!/bin/bash

DATA=NCI1

gm=loopy_bp

LV=3
CONV_SIZE=64
FP_LEN=0
n_hidden=64
bsize=50
num_epochs=10000
learning_rate=0.001
fold=1

python main.py \
    -seed 1 \
    -data $DATA \
    -learning_rate $learning_rate \
    -num_epochs $num_epochs \
    -hidden $n_hidden \
    -max_lv $LV \
    -latent_dim $CONV_SIZE \
    -out_dim $FP_LEN \
    -batch_size $bsize \
    $@
