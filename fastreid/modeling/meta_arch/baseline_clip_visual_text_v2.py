# encoding: utf-8
import pdb

import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY

from fvcore.nn.jit_handles import elementwise_flop_counter
from fvcore.nn import flop_count
import copy

# from fastreid.modeling.backbones.clip.simple_tokenizer import SimpleTokenizer
from fastreid.modeling.backbones.clip.clip import tokenize

import torch.nn.functional as F
from fastreid.modeling.losses.cross_entroy_loss import cross_entropy_loss
import json
import logging

import numpy as np
import os
from sklearn import manifold
import matplotlib.pyplot as plt


@META_ARCH_REGISTRY.register()
class Baseline_clip_visual_text_v2(nn.Module):
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
            # heads,
            heads_v,
            heads_t,
            pixel_mean,
            pixel_std,
            loss_kwargs=None,
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
        self.visual = backbone.visual
        self.text = backbone.text_encoder

        # head
        self.heads_v = heads_v
        self.heads_t = heads_t


        # l2
        self.num_features_v = 512
        self.bottleneck_v = nn.BatchNorm1d(self.num_features_v)
        self.bottleneck_v.bias.requires_grad_(False)
        self.bottleneck_v.apply(self.weights_init_kaiming)

        self.num_features_t = 512
        self.bottleneck_t = nn.BatchNorm1d(self.num_features_t)
        self.bottleneck_t.bias.requires_grad_(False)
        self.bottleneck_t.apply(self.weights_init_kaiming)

        # classfier
        self.num_CF_feature = 512 * 2
        self.num_attribute = 15
        self.CFs = AttrClassifier_all(input_size=self.num_CF_feature, loss_eps=loss_kwargs.get('atrce').get('eps'),
                                      CF_num=self.num_attribute,view_num=None)

        self.proj = nn.ModuleList(
            Mlp_proj(in_features=1024, hidden_features=512, out_features=512) for _ in range(self.num_attribute))  # MLP

        # loss
        self.loss_kwargs = loss_kwargs

        self.prompt_learner = PromptLearner(backbone.dtype, backbone.token_embedding)


        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

        self.logger = logging.getLogger(__name__)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        # heads = build_heads(cfg)

        heads_v = build_heads(cfg)
        heads_t = build_heads(cfg)

        return {
            'backbone': backbone,
            # 'heads': heads,
            'heads_v': heads_v,
            'heads_t': heads_t,

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
                    },
                    'atrce': {
                        'scale': cfg.MODEL.LOSSES.ATTRCE.SCALE,
                        'eps': cfg.MODEL.LOSSES.ATTRCE.EPS,
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        # self.text_sup = self.text_sup.to('cpu')
        images = self.preprocess_image(batched_inputs)
        camids = batched_inputs['camids']
        # texts = batched_inputs['viewids'].to(torch.int64)
        viewids = batched_inputs['viewids']
        attributes = batched_inputs['attributes']
        targets = batched_inputs["targets"]

        B = attributes.shape[0]
        attributes_new = self.trans_attribute(B, attributes)  # [B,attributes_num]

        viewids = viewids.unsqueeze(1)

        features_v, prompt_instance,_ = self.visual(images, camids)
        features_v = self.bottleneck_v(features_v.squeeze())
        features_v = features_v.unsqueeze(-1).unsqueeze(-2)

        features = []
        attributes_instance = []

        visual_squeeze = features_v.squeeze()

        prompt_instance_squeeze = prompt_instance.squeeze()

        for i in range(self.num_attribute):
            temp = torch.cat((visual_squeeze, prompt_instance_squeeze[:, i, :]), dim=1)
            features.append(temp)
            attributes_instance.append(self.proj[i](temp))


        features = torch.stack(features, dim=1)
        attributes_instance = torch.stack(attributes_instance,dim=1)

        if self.training:
            loss_cf, attributes_acc, tokens = self.CFs(features, prompt_instance_squeeze, visual_squeeze, attributes_new, targets)
        else:
            _, attributes_acc, tokens = self.CFs(features, prompt_instance_squeeze, visual_squeeze, attributes_new, targets)

        prompts = self.prompt_learner(attributes_instance,viewids)

        features_t,_ = self.text(prompts, self.prompt_learner.tokenized_prompts)
        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            if targets.sum() < 0: targets.zero_()

            # 图像损失
            outputs_v = self.heads_v(features_v, targets)
            losses_v = self.losses(outputs_v, targets)

            # 文本损失
            features_t = self.bottleneck_t(features_t.squeeze())
            features_t = features_t.unsqueeze(-1).unsqueeze(-2)


            outputs_t = self.heads_t(features_t, targets)
            losses_t = self.losses(outputs_t, targets)


            losses = {}
            for i1, i2, in zip(losses_t.items(), losses_v.items()):
                (k1, v1) = i1
                (k2, v2) = i2
                if k1 != k2:
                    print("ERROR!")
                losses[k1] = v1 + v2

            cf_wight = self.loss_kwargs.get('atrce').get('scale')
            losses['loss_cf'] = loss_cf * cf_wight

            return losses
        else:
            outputs_v = self.heads_v(features_v)
            outputs_t = self.heads_t(features_t)
            outputs = torch.cat((outputs_v, outputs_t), dim=1)

            return outputs


    def lock_net(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            p.requires_grad = False

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
        cls_outputs = outputs['cls_outputs']
        pred_features = outputs['features']
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

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)

        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def flops(self):
        shape = self.__input_shape__[1:]
        supported_ops = self.give_supported_ops()
        model = copy.deepcopy(self)
        model.cuda().eval()
        B = 2
        input_img = torch.randint(0, 35, (B, 3, 256, 128), device=next(model.parameters()).device).to(torch.float32)
        input_t = torch.randint(1, 807, (B,), device=next(model.parameters()).device).to(torch.int64)
        input_c = torch.randint(0, 2, (B,), device=next(model.parameters()).device).to(torch.int64)
        # input_v = torch.randint(1,3970,(1,77), device=next(model.parameters()).device).to(torch.float32)
        input_v = torch.zeros((B, 88), device=next(model.parameters()).device).to(torch.int64)
        input = {'images': input_img, 'targets': input_t, 'camids': input_c, 'viewids': input_v}

        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Drop_path is not included because it defaults to 0 during testing, so please disregard the message.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        del model, input
        return sum(Gflops.values()) * 1e9 / B

    def give_supported_ops(self):
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

    def trans_attribute(self, B, attributes):
        attribute_new = torch.empty((B, 15))
        for i in range(B):
            attr = attributes[i]

            Gender = torch.where(attr[0:3] == 1)[0]
            if Gender.numel() == 0:
                Gender = torch.tensor(len(attr[0:3]) - 1)

            Age = torch.where(attr[3:7] == 1)[0]
            if Age.numel() == 0:
                Age = torch.tensor(len(attr[3:7]) - 1)

            Height = torch.where(attr[7:12] == 1)[0]
            if Height.numel() == 0:
                Height = torch.tensor(len(attr[7:12]) - 1)

            Weight = torch.where(attr[12:16] == 1)[0]
            if Weight.numel() == 0:
                Weight = torch.tensor(len(attr[12:16]) - 1)

            Ethnicity = torch.where(attr[16:21] == 1)[0]
            if Ethnicity.numel() == 0:
                Ethnicity = torch.tensor(len(attr[16:21]) - 1)

            Hair_Color = torch.where(attr[21:28] == 1)[0]
            if Hair_Color.numel() == 0:
                Hair_Color = torch.tensor(len(attr[21:28]) - 1)

            Hairstyle = torch.where(attr[28:34] == 1)[0]
            if Hairstyle.numel() == 0:
                Hairstyle = torch.tensor(len(attr[28:34]) - 1)

            Beard = torch.where(attr[34:37] == 1)[0]
            if Beard.numel() == 0:
                Beard = torch.tensor(len(attr[34:37]) - 1)

            Moustache = torch.where(attr[37:40] == 1)[0]
            if Moustache.numel() == 0:
                Moustache = torch.tensor(len(attr[37:40]) - 1)

            Glasses = torch.where(attr[40:44] == 1)[0]
            if Glasses.numel() == 0:
                Glasses = torch.tensor(len(attr[40:44]) - 1)

            Head_Accessories = torch.where(attr[44:49] == 1)[0]
            if Head_Accessories.numel() == 0:
                Head_Accessories = torch.tensor(len(attr[44:49]) - 1)

            Upper_body_clothing = torch.where(attr[49:62] == 1)[0]
            if Upper_body_clothing.numel() == 0:
                Upper_body_clothing = torch.tensor(len(attr[49:62]) - 1)

            Lower_body_clothing = torch.where(attr[62:72] == 1)[0]
            if Lower_body_clothing.numel() == 0:
                Lower_body_clothing = torch.tensor(len(attr[62:72]) - 1)

            Feet = torch.where(attr[72:79] == 1)[0]
            if Feet.numel() == 0:
                Feet = torch.tensor(len(attr[72:79]) - 1)

            Accessories = torch.where(attr[79:89] == 1)[0]
            if Accessories.numel() == 0:
                Accessories = torch.tensor(len(attr[79:89]) - 1)

            attr_new = torch.tensor(
                [Gender, Age, Height, Weight, Ethnicity, Hair_Color, Hairstyle, Beard, Moustache, Glasses,
                 Head_Accessories, Upper_body_clothing, Lower_body_clothing, Feet, Accessories])

            attribute_new[i] = attr_new

        return attribute_new.to(torch.int64).cuda()


class AttrClassifier_all(nn.Module):
    def __init__(self, input_size, loss_eps=0.1, CF_num=16, view_num=None):
        super().__init__()
        self.loss_eps = loss_eps
        self.CF_num = CF_num
        self.map_num_list = [3, 4, 5, 4, 5, 7, 6, 3, 3, 4, 5, 13, 10, 7, 9]

        if view_num is not None:
            self.map_num_list += [view_num]

        self.input_size = input_size
        CF_config = [{'input_size': self.input_size, 'num_classes': num, 'loss_eps': self.loss_eps} for num in
                     self.map_num_list]
        self.CF_list = nn.ModuleList([AttrClassifier_one(**config) for config in CF_config])

    def forward(self, prompts, patches, visual_squeeze, atttribute, label_):
        loss = 0.0
        attributes_pre = torch.empty_like(atttribute)
        token_list = []

        # for get acc
        atttribute_list = ['Gender', 'Age', 'Height', 'Weight', 'Ethnicity', 'Hair Color', 'Hairstyle',
                           'Beard',
                           'Moustache', 'Glasses', 'Head Accessories', 'Upper Body Clothing', 'Lower Body Clothing',
                           'Feet',
                           'Accessories',
                            'total']
        attributes_acc = {
            'Gender': 0, 'Age': 0, 'Height': 0, 'Weight': 0, 'Ethnicity': 0, 'Hair Color': 0, 'Hairstyle': 0,
            'Beard': 0,
            'Moustache': 0, 'Glasses': 0, 'Head Accessories': 0, 'Upper Body Clothing': 0, 'Lower Body Clothing': 0,
            'Feet': 0,
            'Accessories': 0,
            'total': 0
        }

        acc_total = 0
        total = 0

        for i in range(self.CF_num):
            if self.training:
                loss_temp, prediction,token = self.CF_list[i](patches, prompts[:, i], visual_squeeze, atttribute[:, i], label_ ,i)
                loss = loss + loss_temp
            else:
                _, prediction,token = self.CF_list[i](patches, prompts[:, i], visual_squeeze, atttribute[:, i], label_ , i)

            attributes_pre[:, i] = prediction

            correct = (prediction == atttribute[:,i]).sum().item()
            acc = correct / atttribute[:, i].shape[0]
            attributes_acc[atttribute_list[i]] = acc

            total = total + atttribute[:,i].shape[0]
            acc_total = acc_total + correct

            token_list.append(token)


        attributes_acc[atttribute_list[-1]] = acc_total / total

        token_list = torch.stack(token_list, dim=1)
        if self.training:
            return loss, attributes_acc,token_list
        else:
            return None, attributes_acc,token_list


class AttrClassifier_one(nn.Module):
    def __init__(self, input_size, num_classes, loss_eps):
        super().__init__()
        input_size = input_size

        self.input_size = input_size
        self.hidden_size = input_size // 2
        self.numclasses = num_classes

        self.fc1 = nn.Linear(self.input_size,512)

        self.attn = nn.MultiheadAttention(embed_dim=512,num_heads=8)
        self.channel_fc = nn.Linear(14,1)

        self.ffn = nn.Sequential(QuickGELU(),
                                 nn.BatchNorm1d(512),
                                 nn.Linear(512, self.numclasses))
        self.loss_eps = loss_eps



    def forward(self, patches, prompt, visual, attr, label_, i):
        f1 = self.fc1(prompt)

        x = torch.cat([patches[:, :i], patches[:, i + 1:]], dim=1)
        y = visual.unsqueeze(1).expand(-1,14,-1)
        f2 = self.attn(x,y,y)[0].permute(0,2,1)

        f2 = self.channel_fc(f2).squeeze()

        token = f1 + f2

        score = self.ffn(token)

        probabilities = F.softmax(score, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

        if self.training:
            loss_one = cross_entropy_loss(score, attr, self.loss_eps)
            return loss_one, predicted_class,token
        else:
            return None, predicted_class,token


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., qkv_bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=qkv_bias)
        self.act = QuickGELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=qkv_bias)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(hidden_features)


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Mlp_proj(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., qkv_bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=qkv_bias)
        self.act = QuickGELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=qkv_bias)
        self.drop = nn.Dropout(drop)

        self.bn = nn.BatchNorm1d(hidden_features)

    def forward(self, x):
        x = self.fc1(x)

        if self.training:
            x = self.bn(x)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PromptLearner(nn.Module):
    def __init__(self, dtype, token_embedding):
        super().__init__()
        ctx_init_A = "A UAV view photo of a Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y X X X X X X X X X X X X X X X X person"  # 16+15
        ctx_init_G = "A CCTV view photo of a Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y X X X X X X X X X X X X X X X X person"  # 16+15
        ctx_init_W = "A Wearable view photo of a Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y X X X X X X X X X X X X X X X X person"  # 16+15

        ctx_init_A = ctx_init_A.replace("_", " ")
        ctx_init_G = ctx_init_G.replace("_", " ")
        ctx_init_W = ctx_init_W.replace("_", " ")

        n_ctx = 6

        tokenized_prompts_A = tokenize(ctx_init_A)
        tokenized_prompts_G = tokenize(ctx_init_G)
        tokenized_prompts_W = tokenize(ctx_init_W)
        with torch.no_grad():
            embedding_A = token_embedding(tokenized_prompts_A).type(dtype)
            embedding_G = token_embedding(tokenized_prompts_G).type(dtype)
            embedding_W = token_embedding(tokenized_prompts_W).type(dtype)

        self.tokenized_prompts = tokenized_prompts_A  # 只是为了定位EOS的索引，由于A、G的长度相同，不需要差异化处理
        ctx_dim = 512

        n_share = 16

        share_vectors = torch.empty(n_share, ctx_dim, dtype=dtype)
        nn.init.normal_(share_vectors, std=0.02)
        self.share_vectors = nn.Parameter(share_vectors)

        n_attribute = 15
        n_cls_ctx = n_share + n_attribute
        # n_cls_ctx = n_attribute

        token_prefix_list = [embedding_A[:, :n_ctx + 1, :],
                             embedding_G[:, :n_ctx + 1, :],
                             embedding_W[:, :n_ctx + 1, :],]  # 0:UAV,1:CCTV,2:Wearable

        token_suffix_list = [embedding_A[:, n_ctx + 1 + n_cls_ctx:, :],
                             embedding_G[:, n_ctx + 1 + n_cls_ctx:, :],
                             embedding_W[:, n_ctx + 1 + n_cls_ctx:, :],] # 0:UAV,1:CCTV,2:Wearable

        token_prefix_list = torch.stack(token_prefix_list,dim=0)
        token_suffix_list = torch.stack(token_suffix_list,dim=0)

        self.register_buffer("token_prefix", token_prefix_list)
        self.register_buffer("token_suffix", token_suffix_list)


    def forward(self, attribute,viewids):
        B = attribute.shape[0]
        prefix = self.token_prefix[viewids].squeeze()
        suffix = self.token_suffix[viewids].squeeze()

        share = self.share_vectors.expand(B, -1, -1)


        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                share,
                attribute,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts.cuda()
