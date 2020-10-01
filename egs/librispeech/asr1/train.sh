#!/bin/bash
# general configuration
. ./path.sh || exit 1
. ./cmd.sh || exit 1

ngpu=4
debugmode=1
verbose=0
resume=

workdir=$(pwd)
config=${workdir}/conf/train_transformer_transducer.yaml

expname=transformer_transducer
expdir=exp
exppath=${expdir}/${expname}

mkdir -p ${exppath}


${cuda_cmd} --gpu ${ngpu} ${exppath}/train.log \
    asr_train.py \
    --config ${config} \
    --preprocess-conf ${workdir}/conf/specaug.yaml \
    --ngpu ${ngpu} \
    --outdir ${exppath}/results \
    --train-json dump/train_960/deltafalse/data_unigram5000.json \
    --valid-json dump/dev/deltafalse/data_unigram5000.json \
    --resume ${resume} \
    --dict data/lang_char/train_960_unigram5000_units.txt

