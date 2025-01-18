############# Transformer++ vs DCFormer++ 

#model_names=("DCFormerLlama")
#model_names=("Llama")
#model_names=("DynamicDenseFormer")
model_names=("MUDDFormer")
#model_names=("DDFormer" "DenseFormer" "MUDDFormer")
#model_names=("DDFormer")
#model_names=("DenseFormer")
#model_sizes=("13B")
#model_sizes=("7B")
#model_sizes=("2p8B")
model_sizes=("1p4Bb")

#model_sizes=("7B" "13B")

#model_names=("Llama" "DCFormerLlama")
#model_sizes=("1p4Bb" "2p8B" "7B" "13B")

batch_sizes=(1)
compile_prefill=""

full_cache="--full_cache" ;full_cache_log_name="full_cache1"
#full_cache="" ;full_cache_log_name="full_cache0"
#window_size="--window_size 256";window_size_str="window_size256"
window_size="";window_size_str="window_sizeNone"
window_type="default"
#window_type="LG"
#window_type="LGLL"
#window_type="LGL6"
#query_wise="--query_wise";query_wise_str="qw1"
query_wise="";query_wise_str="qw0"
compile="--compile"; compile_str="compile1"
#compile=""; compile_str="compile0"
#window_type="LG"
#--checkpoint_path $MODEL_REPO/model.pth
profile="--profile perfs/"

for model_size in "${model_sizes[@]}"; do
    for model_name in "${model_names[@]}"; do
        for bs in "${batch_sizes[@]}"; do
            echo "$model_name $model_size batchsize$bs $full_cache_log_name $window_size_str window${window_type} ${query_wise_str} start"
            logname="logs/v5_inference_performance_${model_name}_${model_size}_batchsize${bs}_compiled_${full_cache_log_name}_${window_size_str}_window${window_type}_${query_wise_str}_${compile_str}.log" 
            # v4 log for evaluating DynamicDenseFormer
            echo ${logname} 
            #TORCH_COMPILE_DEBUG=0 
            echo "TORCH_COMPILE_DEBUG=1 python -u generate.py ${compile}  ${compile_prefill} ${full_cache} ${window_size} --fake_prompt ${profile} --max_batch_size ${bs} --model_name ${model_name} --model_size ${model_size} --max_new_tokens 128 --num_samples 3 --temperature 0.5 --window_type ${window_type} ${query_wise} > ${logname} 2>&1"
            #TORCH_COMPILE_DEBUG=1 python -u generate.py ${compile}  ${compile_prefill} ${full_cache} ${window_size} --fake_prompt ${profile} --max_batch_size ${bs} --model_name ${model_name} --model_size ${model_size} --max_new_tokens 128 --num_samples 3 --temperature 0.5 --window_type ${window_type} ${query_wise} > ${logname} 2>&1 
            echo "$model_name $model_size batchsize$bs $full_cache_log_name $window_size_str window${window_type} ${query_wise_str} done"
        done
    done
done
