
export WANDB_PROJECT="videovlm_motion"
export WANDB_NAME="motions_finetune"

LLM_path="/mnt/bn/themis/data/LLM/vicuna-7b-v1.5"
motion_ckpt="/mnt/bn/videovlm/ckpt/video_reconstruction/checkpoints/v2_vatex_smaller/checkpoint_epoch_1_iter2000.pth"
stage1_save_path="/mnt/bn/videovlm/code/themis/checkpoints/base_pretrain/stage1_4"
stage2_save_path="/mnt/bn/videovlm/code/themis/checkpoints/motions_finetune/stage2"

deepspeed \
--include localhost:0 \
--master_port=29601 \
ChatUniVi/train/train_mem.py \
--deepspeed scripts/zero2.json \
--model_name_or_path ${LLM_path} \
--version v1 \
--model_use FINETUNE_MOT \
--dataset_use FINETUNE \
--vision_tower openai/clip-vit-large-patch14-336 \
--pretrain_mm_mlp_adapter ${stage1_save_path}/mm_projector.bin \
--pretrain_mm_masking ${motion_ckpt} \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir ${stage2_save_path} \
--num_train_epochs 2 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 2e-5 \
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