seed=2024
CUDA_VISIBLE_DEVICES=0 python main.py --dataset ImageNet-C --data_corruption .../datasets/ImageNet-C \
--exp_type normal --method dpal --model vitbase_timm --output ./output/debug  \
--seed $seed --test_batch_size 64 --prompt_deep