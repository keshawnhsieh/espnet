#!/bin/bash

#ffmpeg -i /export/a15/vpanayotov/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac /export/a15/vpanayotov/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.wav
# cp 1272-128104-0000.wav 1272-128104-0001.wav ....


stage=0
stop_stage=100
ngpu=0
do_delta=false
cmvn=data/train_960/cmvn.ark


. ./path.sh
. utils/parse_options.sh || exit 1;
. ./cmd.sh




wav=/export/a15/vpanayotov/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.wav
decode_dir=mini_decode
base=$(basename $wav .wav)
decode_dir=${decode_dir}/${base}



if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    mkdir -p ${decode_dir}/data
    echo "$base $wav" > ${decode_dir}/data/wav.scp
    echo "X $base" > ${decode_dir}/data/spk2utt
    echo "$base X" > ${decode_dir}/data/utt2spk
    echo "$base X" > ${decode_dir}/data/text
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"

    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 1 --write_utt2num_frames true \
        ${decode_dir}/data ${decode_dir}/log ${decode_dir}/fbank

    feat_recog_dir=${decode_dir}/dump; mkdir -p ${feat_recog_dir}
    dump.sh --cmd "$train_cmd" --nj 1 --do_delta ${do_delta} \
        ${decode_dir}/data/feats.scp ${cmvn} ${decode_dir}/log \
        ${feat_recog_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Json Data Preparation"

    dict=${decode_dir}/dict
    echo "<unk> 1" > ${dict}
    feat_recog_dir=${decode_dir}/dump
    data2json.sh --feat ${feat_recog_dir}/feats.scp \
        ${decode_dir}/data ${dict} > ${feat_recog_dir}/data.json
    rm -f ${dict}
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Decoding"
    if ${use_lang_model}; then
        recog_opts="--rnnlm ${lang_model}"
    else
        recog_opts=""
    fi
    feat_recog_dir=${decode_dir}/dump

    ${decode_cmd} ${decode_dir}/log/decode.log \
        asr_recog.py \
        --config conf/decode_default.yaml \
        --ngpu ${ngpu} \
        --backend pytorch \
        --recog-json ${feat_recog_dir}/data.json \
        --result-label ${decode_dir}/result.json \
        --model exp/transformer_transducer/results/model.last5.avg.best \

    echo ""
    recog_text=$(grep rec_text ${decode_dir}/result.json | sed -e 's/.*: "\(.*\)".*/\1/' | sed -e 's/<eos>//')
    echo "Recognized text: ${recog_text}"
    echo ""
    echo "Finished"
fi