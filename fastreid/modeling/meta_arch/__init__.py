# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import META_ARCH_REGISTRY, build_model


# import all the meta_arch, so they will be registered
from .baseline import Baseline
from .mgn import MGN
from .pcb import PCB
from .moco import MoCo
from .distiller import Distiller
from .baseline_multi_view import Baseline_multiview

from .baseline_clip_demo import Baseline_clip_demo
from .baseline_clip_visual_text import Baseline_clip_visual_text
from .baseline_clip_visual_text_cargo import Baseline_clip_visual_text_cargo
from .baseline_clip_visual_text_v2 import Baseline_clip_visual_text_v2


from .baseline_clip_visual_text_forvis import Baseline_clip_visual_text_forvis
from .baseline_clip_visual_text_attnvis import Baseline_clip_visual_text_attnvis

from .baseline_clip_visual_text_share4 import Baseline_clip_visual_text_share4
from .baseline_clip_visual_text_share16 import Baseline_clip_visual_text_share16
from .baseline_clip_visual_text_share2 import Baseline_clip_visual_text_share2
from .baseline_clip_visual_text_share12 import Baseline_clip_visual_text_share12
from .baseline_clip_visual_text_woView import Baseline_clip_visual_text_woView
