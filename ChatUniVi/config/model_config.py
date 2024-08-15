model_config_pretune = {
    "use_cluster": True,
    "use_masking": False,
    "freeze": False,
    "vision_tune": False,

    "spatial_cluster_rate0": 64,  # 0.25
    "spatial_cluster_rate1": 32,  # 0.5
    "spatial_cluster_rate2": 16,  # 0.5

    "temporal_cluster_rate": 1/16,
}


model_config_finetune = {
    "use_cluster": True,
    "use_masking": False,
    "freeze": False,
    "mm_tune": True,
    "vision_tune": False,

    "spatial_cluster_rate0": 64,  # 0.25
    "spatial_cluster_rate1": 32,  # 0.5
    "spatial_cluster_rate2": 16,  # 0.5

    "temporal_cluster_rate": 1/16,
}

model_config_pretune_motion = {
    "use_masking": True,
    "use_cluster": False,
    "use_ada": False,
    "freeze": False,
    "vision_tune": False,

    "num_patches": 576, 
    "num_layers": 2,
    "num_head": 16, 
    "mask_ratio": 0.9,
    "use_learnable_pos_emb": False, 
}

model_config_finetune_motion = {
    "use_masking": True,
    "use_cluster": False,
    "use_ada": False,
    "freeze": False,
    "mm_tune": True,
    "vision_tune": False,
    "padding_to_full": False,

    "num_patches": 576, 
    "num_layers": 2,
    "num_head": 16, 
    "mask_ratio": 0.9,
    "use_learnable_pos_emb": True, 
}

model_config_finetune_motion_padding = {
    "use_masking": True,
    "use_cluster": False,
    "use_ada": False,
    "padding_to_full": True,
    "freeze": False,
    "mm_tune": True,
    "vision_tune": False,

    "num_patches": 576, 
    "num_layers": 2,
    "num_head": 16, 
    "mask_ratio": 0.9,
    "use_learnable_pos_emb": True, 
}