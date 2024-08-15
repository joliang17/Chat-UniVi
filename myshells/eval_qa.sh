

git config --global --add safe.directory /mnt/bn/yijun-multimodal/Chat-UniVi


LLM_model="/mnt/bn/videovlm/code/themis/checkpoints/motion_finetune_padding/stage2/checkpoint-15000/"
saved_file="/mnt/bn/yijun-multimodal/themis/themis/eval/answers/activitynet_answer_motion_pad.json"
final_folder="activity_qa_net_motion_pad"
final_json="activity_qa_net_motion_pad.json"
API_KEY=""

python3 ChatUniVi/eval/model_video_qa.py \
    --model-path=${LLM_model} \
    --answers-file=${saved_file} \

python3 ChatUniVi/eval/evaluate/evaluate_video_qa.py \
    --pred_path=${saved_file} \
    --output_dir=${final_folder} \
    --output_json=${final_json} \
    --api_key=${API_KEY}