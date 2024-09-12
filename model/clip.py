import torch
from torch import nn
import numpy as np

from model import MobileFaceNet


class CLIP(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            vision_model: nn.Module,
            text_model: nn.Module,
            context_length: int,
            vocab_size: int):
        super().__init__()

        self.context_length = context_length

        # use redesigned mobilefacenet for vision encoder
        self.visual_model = vision_model

        self.text_model = text_model

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, self.text_model.width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, self.text_model.width))
        self.ln_final = nn.LayerNorm(self.text_model.width)

        self.text_projection = nn.Parameter(
            torch.empty(self.text_model.width, embed_dim))


        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.text_model.width ** -0.5) * (
                (2 * self.text_model.layers) ** -0.5)
        attn_std = self.text_model.width ** -0.5
        fc_std = (2 * self.text_model.width) ** -0.5
        for block in self.text_model.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection,
                            std=self.text_model.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image):
        return self.visual_model(image)

    def encode_text(self, text):
        x = self.token_embedding(text).type(
            self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)[:text.shape[1]]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_model(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]),
        text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        # return logits_per_image, logits_per_text
        return image_features, text_features
