############# Transformer++ vs DCFormer++ 

model_names=("DCFormerLlama")
model_sizes=("7B")

#model_names=("Llama" "DCFormerLlama")
#model_sizes=("2p8B" "7B" "13B" "33B")

batch_sizes=(1)
compile_prefill=""
full_cache="" ;full_cache_log_name="full_cache0"
window_size="--window_size 256";window_size_str="window_size256"
#window_size="";window_size_str="window_sizeNone"
#--checkpoint_path $MODEL_REPO/model.pth

for model_size in "${model_sizes[@]}"; do
    for model_name in "${model_names[@]}"; do
        for bs in "${batch_sizes[@]}"; do
            echo "$model_name $model_size batchsize$bs $full_cache_log_name $window_size_str start"
            python -u generate.py --compile  ${compile_prefill} ${full_cache} ${window_size} --fake_prompt --max_batch_size ${bs} --model_name ${model_name} --model_size ${model_size}  --max_new_tokens 128 --num_samples 3 --temperature 0.5 > logs/v3_inference_performance_${model_name}_${model_size}_batchsize${bs}_compiled_${full_cache_log_name}_${window_size_str}.log 2>&1 
            echo "$model_name $model_size batchsize$bs $full_cache_log_name $window_size_str done"
        done
    done
done
