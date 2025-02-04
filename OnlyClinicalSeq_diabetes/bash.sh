echo "MIMIC dataset"
SEED=2028
target='mortality'

for m_num in 0
do  
    for imputation in 'median_median' 'simpleimputer_flagmedian'
    do 
        gpu_idx=1
        CUDA_VISIBLE_DEVICES=$gpu_idx python 2.ChronicDisease/OnlyClinical/TotalMain.py \
        --cuda_num=$gpu_idx --seed=$SEED --model_num=$m_num --target=$target --imputation=$imputation &
    done
done
echo "MIMIC dataset End"