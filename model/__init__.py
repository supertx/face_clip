'''
Author: supermantx
Date: 2024-09-06 16:38:51
LastEditTime: 2024-09-12 16:31:12
Description: 
'''
from model.mbf import MBF as MobileFaceNet
from model.transformer import Transformer
from model.clip import CLIP

from model.criterion.clip_loss import ClipLoss

support_models = ["mobilefacenet", "transformer", "clip"]


def build_model(cfg, **arg):
    assert cfg.model_name.lower() in support_models, f"model name should be one of {__all__}"

    if cfg.model_name.lower() == "mobilefacenet":
        model = MobileFaceNet(stages_channel=cfg.stages_channels,
                              stages=cfg.stages,
                              inner_scale=cfg.inner_scale,
                              num_features=arg.get("num_features"))
    elif cfg.model_name.lower() == "transformer":
        model = Transformer(layers=cfg.layers,
                            heads=cfg.heads,
                            attn_mask=None,
                            width=arg.get("width"))
    elif cfg.model_name.lower() == "clip":
        model = CLIP(embed_dim=cfg.clip.embedding_size,
                     context_length=cfg.clip.context_length,
                     vocab_size=cfg.clip.vocab_size,
                     vision_model=build_model(
                         cfg.vision_model,
                         num_features=cfg.clip.embedding_size,
                         **arg),
                     text_model=build_model(
                        cfg.text_model,
                        width=cfg.clip.embedding_size,
                        **arg))
    if arg.get("use_cuda"):
        return model.cuda()
    else:
        return model
