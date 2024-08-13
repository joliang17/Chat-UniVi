root_path = '/mnt/bn/themis/data/LLM/Chat-UniVi-Instruct/'

Pretrain = {
    "chat_path": f"{root_path}/Pre-training/CC3M-595K/chat.json",
    "CC3M": f"{root_path}/Pre-training/CC3M-595K/image_files/",
}

VIT = {
    "chat_path": f"{root_path}/Fine-tuning/VIT/llava_instruct_150k.json",
    "COCO2017": f"{root_path}/Fine-tuning/VIT/train2017/",
}

MIMIC_imageonly_cgd = {
    "chat_path": f"{root_path}/Fine-tuning/MIMIC_imageonly/MIMIC-IT-imageonly_CGD.json",
    "CDG": f"{root_path}/Fine-tuning/MIMIC_imageonly/CGD/images",
}

MIMIC_imageonly_la = {
    "chat_path": f"{root_path}/Fine-tuning/MIMIC_imageonly/MIMIC-IT-imageonly_LA.json",
    "LA": f"{root_path}/Fine-tuning/MIMIC_imageonly/LA/images",
}

MIMIC_imageonly_sd = {
    "chat_path": f"{root_path}/Fine-tuning/MIMIC_imageonly/MIMIC-IT-imageonly_SD.json",
    "SD": f"{root_path}/Fine-tuning/MIMIC_imageonly/SD/images",
}

# MIMIC_imageonly = {
#     "chat_path": f"{root_path}/MIMIC-IT-imageonly.json",
#     "CDG": "${root_path}/CGD/images",
#     "LA": "${root_path}/LA/images",
#     "SD": "${root_path}/SD/images",
# }

COCO_CAP = {
    "chat_path": f"{root_path}/Pre-training/COCO/coco_cap_chat.json",
    "COCO2014": f"{root_path}/Pre-training/COCO/train2014",
}

COCO_REG = {
    "chat_path": f"{root_path}/Pre-training/COCO/coco_reg_chat.json",
    "COCO2014": f"{root_path}/Pre-training/COCO/train2014",
}

COCO_REC = {
    "chat_path": f"{root_path}/Pre-training/COCO/coco_rec_chat.json",
    "COCO2014": f"{root_path}/Pre-training/COCO/train2014",
}

VIDEO = {
    "chat_path": f"{root_path}/Fine-tuning/VIDEO/video_chat.json",
    "VIDEO": f"{root_path}/Fine-tuning/VIDEO/Activity_Videos",
}

SQA = {
    "chat_path": f"{root_path}/ScienceQA_tuning/llava_train_QCM-LEA.json",
    "ScienceQA": f"{root_path}/ScienceQA_tuning/train",
}

Pretrain_valley_llava = {
    "chat_path": "${root_path}/valley_llavaimage.json",
    "valley": "${root_path}/Data",
    "llava": "${root_path}/Data",  # from llava v1.5
}

LLaVA = {
    "chat_path": "${root_path}/llavaimage_tune.json",
    "llava": "${root_path}/Data",  # from llava v1.5
}