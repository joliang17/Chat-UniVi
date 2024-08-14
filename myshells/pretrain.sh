
export WANDB_PROJECT="videovlm_motion"
export WANDB_NAME="base_pretrain"

# --pretrain_mm_masking ${motion_ckpt} \

stage1_save_path="/mnt/bn/videovlm/code/themis/checkpoints/base_pretrain/stage1"
motion_ckpt="/mnt/bn/videovlm/ckpt/video_reconstruction/checkpoints/v2_vatex_smaller_largerclip/checkpoint_epoch_1_iter1000.pth"
LLM_path="/mnt/bn/themis/data/LLM/vicuna-7b-v1.5"

deepspeed \
--include localhost:0 \
--master_port=29602 \
ChatUniVi/train/train_mem.py \
--deepspeed scripts/zero3.json \
--model_name_or_path ${LLM_path} \
--version v1 \
--model_use PRETUNE_MOT \
--dataset_use FINETUNE \
--vision_tower openai/clip-vit-large-patch14-336 \
--tune_mm_mlp_adapter True \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir ${stage1_save_path} \
--num_train_epochs 1 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 24000 \
--save_total_limit 1 \
--learning_rate 2e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to wandb