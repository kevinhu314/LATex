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
class Baseline_multiview(nn.Module):
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
            view_heads,
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
        self.view_heads = view_heads

        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)

        cfg0 = cfg.clone()
        if cfg0.is_frozen(): cfg0.defrost()
        cfg0.MODEL.HEADS.NUM_CLASSES = 2
        view_heads = build_heads(cfg0)
        cfg0 = cfg.clone()

        return {
            'backbone': backbone,
            'heads': heads,
            'view_heads': view_heads,
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
                        'scale': cfg.MODEL.LOSSES.CE.SCALE,
                        'view_id': cfg.MODEL.LOSSES.CE.VIEW_ID,
                        'view_oreg': cfg.MODEL.LOSSES.CE.VIEW_OREG,
                        'view_lambda': cfg.MODEL.LOSSES.CE.VIEW_LAMBDA,
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
                    },

                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        camids = batched_inputs['camids']

        view = batched_inputs['viewids']
        view1_index = [index for index, content in enumerate(view) if content == 'Aerial']
        view2_index = [index for index, content in enumerate(view) if content == 'Ground']

        global_feats, view_feats = self.backbone(images, camids)
        features = global_feats - view_feats

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            temp = torch.zeros((targets.shape[0])).long().to(targets.device)
            temp[view1_index] = 1
            targets_view = temp

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            outputs_global = self.heads(global_feats, targets)
            view_outputs = self.view_heads(view_feats, targets_view)
            losses = self.losses(outputs, outputs_global, view_outputs, targets, targets_view)
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

    def losses(self, outputs, outputs_global, outputs_view, gt_labels, view_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs = outputs['cls_outputs']
        pred_features = outputs['features']
        view_pred_class_logits = outputs_view['pred_class_logits'].detach()
        view_cls_outputs = outputs_view['cls_outputs']
        view_pred_features = outputs_view['features']
        global_pred_class_logits = outputs_global['pred_class_logits'].detach()
        global_cls_outputs = outputs_global['cls_outputs']
        global_pred_features = outputs_global['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        view_kwargs = self.loss_kwargs.get('ce')
        view_id_flag = view_kwargs.get('view_id')
        view_oreg_flag = view_kwargs.get('view_oreg')
        view_lambda = view_kwargs.get('view_lambda')

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls_id'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

            if view_id_flag:
                loss_dict['loss_cls_view'] = cross_entropy_loss(
                    view_cls_outputs,
                    view_labels,
                    ce_kwargs.get('eps'),
                    ce_kwargs.get('alpha')
                ) * ce_kwargs.get('scale') * view_lambda

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet_id'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        # calc oreg loss part
        if view_oreg_flag:
            loss_dict['loss_oreg'] = torch.cosine_similarity(pred_features, view_pred_features).abs().mean() * view_lambda

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
        supported_ops = self.give_supported_ops()
        model = copy.deepcopy(self)
        model.cuda().eval()
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
