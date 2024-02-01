#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_sizes=("2p8B" "7B")
batch_sizes=("1" "4")

model_name='DCFormerLlama';use_dcmha="--use_dcmha";q_chunk_sizes=("128" "256")
#model_name="Llama";use_dcmha="";q_chunk_sizes=("512" "2048")

compile="--compile"

for model_size in "${model_sizes[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for q_chunk_size in "${q_chunk_sizes[@]}"; do
            echo "$model_name $model_size batchsize$batch_size q_chunk${q_chunk_size} start"
            time python train.py --model_size ${model_size} ${use_dcmha} --query_chunk_size ${q_chunk_size} --batch_size ${batch_size} ${compile} > logs/training_${model_name}_${model_size}_qchunk${q_chunk_size}_batchsize${batch_size}_compiled.log 2>&1 
            echo "$model_name $model_size batchsize$batch_size q_chunk${q_chunk_size} done"
        done
    done
done
