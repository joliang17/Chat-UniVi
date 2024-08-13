from .dataset_config import *
from .model_config import *


ModelConfig = {
    "PRETUNE": model_config_pretune,
    "PRETUNE_MOT": model_config_pretune_motion,
    "FINETUNE": model_config_finetune,
    "FINETUNE_MOT": model_config_finetune_motion,
}


DataConfig = {
    "Pretrain": [Pretrain, COCO_CAP, COCO_REG, COCO_REC],
    "SQA": [SQA],
    # "FINETUNE": [VIT, MIMIC_imageonly, VIDEO],
    "FINETUNE": [VIT, MIMIC_imageonly_cgd, MIMIC_imageonly_la, MIMIC_imageonly_sd, VIDEO],
    "Pretrainv1.5": [Pretrain, Pretrain_valley_llava],
    "FINETUNEv1.5": [VIT, VIDEO, LLaVA],
}