import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)



class MHAMasking(nn.Module):
    """
    Find masking probability with cross attention on f_{t-1} & f_{t}
    """

    def __init__(self, embed_dim: int, n_patches: int, n_layers: int = 6,
                 n_head: int = 16, mask_ratio: float = 0.9, use_learnable_pos_emb=False, ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_patches = n_patches
        self.n_layers = n_layers
        self.n_head = n_head

        # TODO: fixed ratio or not
        self.mask_ratio = mask_ratio
        self.visible_patches = int(n_patches * (1 - mask_ratio))
        print("No. of visible patches selected for pre-training: {}".format(self.visible_patches))

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(n_patches, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=n_head, batch_first=True)
        self.gen_token_feat = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Probability prediction network
        self.pos_embed_probs = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        self.get_token_probs = nn.Sequential(nn.Linear(embed_dim, 1), torch.nn.Flatten(start_dim=1), )
        self.softmax = nn.Softmax(dim=-1)

    def get_mask(self, prev_img, next_img):
        """
        prev_img: [B*(T-1), N, D], 0 -- T-1
        next_img: [B*(T-1), N, D], 1 -- T
        """
        # add position embedding to each patch
        # pos_embed_probs.shape: [1, N, D]
        next_img = next_img + self.pos_embed_probs.type_as(next_img).to(next_img.device).clone()

        # prev_img: Q, next_img: K, V
        # frame_diff shape: (B*T-1), N, D
        # TODO:
        frame_diff = self.gen_token_feat(tgt=next_img, memory=prev_img)

        # generate patch probability
        logits = self.get_token_probs(frame_diff)
        logits = torch.nan_to_num(logits)
        prob_patch = self.softmax(logits)

        # generate visible mask
        # vis_idx.shape: (B*T-1), N
        vis_idx = torch.multinomial(prob_patch, num_samples=self.visible_patches, replacement=False)
        mask = torch.ones((next_img.shape[0], next_img.shape[1])).to(next_img.device, non_blocking=True)
        mask.scatter_(dim=-1, index=vis_idx.long(), value=0.0)
        mask = mask.flatten(1).to(torch.bool)
        return prob_patch, vis_idx, mask

    def forward(self, image_feat: torch.Tensor, ):
        """
        image_feat: [B, T, N, D]
        """
        B, T, N, D = image_feat.shape

        # prev_img shape: B, T-1, N, D
        # next_img shape: B, T-1, N, D
        prev_img = image_feat[:, :-1].reshape(-1, N, D)
        next_img = image_feat[:, 1:].reshape(-1, N, D)

        # find masked id
        # mask.shape: B * (T - 1), N
        # prob_patch.shape: [B * (T - 1), N]
        prob_patch, vis_idx, mask = self.get_mask(prev_img, next_img)

        # generate masked features
        # pos_embed.shape: 1, N, D
        next_img = next_img + self.pos_embed.type_as(next_img).to(next_img.device).clone().detach()

        BT, _, D1 = next_img.shape
        # TODO: ~mask means visible shape: (B * T - 1), N2, N2=N * mask_ratio
        next_img_vis = next_img[~mask].reshape(BT, -1, D1)

        next_img_vis = rearrange(next_img_vis, '(b t) k d -> b t k d', b=B, t=T - 1)
        mask = rearrange(mask, '(b t) n -> b t n', b=B, t=T - 1)

        return prob_patch, next_img_vis, mask
