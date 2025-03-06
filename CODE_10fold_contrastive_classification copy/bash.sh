echo "CHA dataset"
target='volume'
dataset_name="cha_infertility"

for SEED in 2024 2035
do
    echo "Processing SEED: $SEED"
    for m_num in 0 1 2 3
    do  
        gpu_idx=$((m_num % 2 + 3))
        echo "Running on GPU $gpu_idx | Seed: $SEED | Model: $m_num"
        CUDA_VISIBLE_DEVICES=$gpu_idx python 3.Infertility/CODE_5fold_regressor/TotalMain.py \
        --dataset_name=$dataset_name --cuda_num=$gpu_idx --seed=$SEED --model_num=$m_num & 
    done
done 
wait
echo "CHA dataset End"