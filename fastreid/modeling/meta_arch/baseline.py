# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import pdb

import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone

        # head
        self.heads = heads

        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        return {
            'backbone': backbone,
            'heads': heads,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        camids = batched_inputs['camids']

        features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)
            return losses
        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        return loss_dict


    def flops(self):
        from fvcore.nn import flop_count
        import copy
        # shape = self.__input_shape__[1:]
        # if self.image_size[0] != shape[1] or self.image_size[1] != shape[2]:
        #     shape = (3, self.image_size[0], self.image_size[1])
        # For vehicle reid, the input shape is (3, 128, 256)
        supported_ops = self.give_supported_ops()
        model = copy.deepcopy(self)
        model.cuda().eval()
        # input_r = torch.randn((1, *shape), device=next(model.parameters()).device)
        # input_n = torch.randn((1, *shape), device=next(model.parameters()).device)
        # input_t = torch.randn((1, *shape), device=next(model.parameters()).device)
        # input = {"RGB": input_r, "NI": input_n, "TI": input_t}

        # input:{'images':[B,3,256,128],torch.float32,
        #       'targets':[B,],torch.int64,
        #       'camids':[B,],torch.int64,
        #       'viewids':[B,77],torch.float32
        #       }
        B = 2
        input_img = torch.randint(0, 35, (B, 3, 256, 128), device=next(model.parameters()).device).to(torch.float32)
        input_t = torch.randint(1, 807, (B,), device=next(model.parameters()).device).to(torch.int64)
        input_c = torch.randint(0, 2, (B,), device=next(model.parameters()).device).to(torch.int64)
        input_v = torch.zeros((B, ), device=next(model.parameters()).device).to(torch.int64)
        input_a = torch.zeros((B, 88), device=next(model.parameters()).device).to(torch.int64)
        input = {'images': input_img, 'targets': input_t, 'camids': input_c, 'viewids': input_v,'attributes':input_a}

        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Drop_path is not included because it defaults to 0 during testing, so please disregard the message.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        del model, input
        return sum(Gflops.values()) * 1e9 / B

    def give_supported_ops(self):
        from fvcore.nn.jit_handles import elementwise_flop_counter
        return {
            "aten::silu": elementwise_flop_counter(0, 1),
            "aten::gelu": elementwise_flop_counter(0, 1),
            "aten::neg": elementwise_flop_counter(0, 1),
            "aten::exp": elementwise_flop_counter(0, 1),
            "aten::flip": elementwise_flop_counter(0, 1),
            "aten::mul": elementwise_flop_counter(0, 1),
            "aten::div": elementwise_flop_counter(0, 1),
            "aten::softmax": elementwise_flop_counter(0, 2),
            "aten::sigmoid": elementwise_flop_counter(0, 1),
            "aten::add": elementwise_flop_counter(0, 1),
            "aten::add_": elementwise_flop_counter(0, 1),
            "aten::radd": elementwise_flop_counter(0, 1),
            "aten::sub": elementwise_flop_counter(0, 1),
            "aten::sub_": elementwise_flop_counter(0, 1),
            "aten::rsub": elementwise_flop_counter(0, 1),
            "aten::mul_": elementwise_flop_counter(0, 1),
            "aten::rmul": elementwise_flop_counter(0, 1),
            "aten::div_": elementwise_flop_counter(0, 1),
            "aten::rdiv": elementwise_flop_counter(0, 1),
            "aten::cumsum": elementwise_flop_counter(0, 1),
            "aten::ne": elementwise_flop_counter(0, 1),
            "aten::silu_": elementwise_flop_counter(0, 1),
            "aten::dropout_": elementwise_flop_counter(0, 1),
            "aten::log_softmax": elementwise_flop_counter(0, 2),
            "aten::argmax": elementwise_flop_counter(0, 1),
            "aten::one_hot": elementwise_flop_counter(0, 1),
            "aten::flatten": elementwise_flop_counter(0, 0),
            "aten::unflatten": elementwise_flop_counter(0, 0),
            "aten::mean": elementwise_flop_counter(1, 0),
            "aten::sum": elementwise_flop_counter(1, 0),
            "aten::abs": elementwise_flop_counter(0, 1),
            "aten::tanh": elementwise_flop_counter(0, 1),
            "aten::relu": elementwise_flop_counter(0, 1),
            "aten::where": elementwise_flop_counter(0, 1),
            "aten::le": elementwise_flop_counter(0, 1),
            "aten::topk": elementwise_flop_counter(1, 1),
            "aten::sort": elementwise_flop_counter(1, 1),
            "aten::argsort": elementwise_flop_counter(1, 1),
            "aten::scatter": elementwise_flop_counter(1, 1),
            "aten::gather": elementwise_flop_counter(1, 1),
            "aten::adaptive_max_pool2d": elementwise_flop_counter(1, 0),
        }

