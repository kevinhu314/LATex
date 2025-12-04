# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import build_backbone, BACKBONE_REGISTRY

from .resnet import build_resnet_backbone
from .osnet import build_osnet_backbone
from .resnest import build_resnest_backbone
from .resnext import build_resnext_backbone
from .regnet import build_regnet_backbone, build_effnet_backbone
from .shufflenet import build_shufflenetv2_backbone
from .mobilenet import build_mobilenetv2_backbone
from .mobilenetv3 import build_mobilenetv3_backbone
from .repvgg import build_repvgg_backbone
from .vision_transformer import build_vit_backbone
from .vision_transformer_multiview_onebranch import build_multiview_vit_backbone_onebranch
from .clip import build_clip

from .ViT_clip_prompt_demo import build_clip_prompt_demo

from .ViT_clip_prompt import build_clip_prompt
from .ViT_clip_prompt_cargo import build_clip_prompt_cargo
from .ViT_clip_prompt_v2 import build_clip_prompt_v2

from .ViT_clip_prompt_full import build_clip_prompt_full
from .ViT_clip_prompt_cargo_full import build_clip_prompt_cargo_full
from .ViT_clip_prompt_v2_full import build_clip_prompt_v2_full


from .ViT_clip_prompt_forvis import build_clip_prompt_forvis

