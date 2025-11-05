import torch
import os
import torchvision.models as models
import timm
from timm import create_model
import models_mamba
from torch import nn
from Normalize import TfNormalize
import warnings
from torch_nets import (
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    tf_adv_inception_v3,
    )
from collections import OrderedDict
# from tf_adv_inception_v3 import KitModel
warnings.filterwarnings("ignore")
from models_mamba import vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
from models_mamba import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
def get_vim_model(type="small"):
    if type == "small":
        model = vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False,
                                                                                                  num_classes=1000)

        checkpoint_path = "./vim_s_midclstok_80p5acc.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    elif type == "tiny":
        model = vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False,
                                                                                                 num_classes=1000)
        checkpoint_path = "./checkpoints/vim_s_midclstok_80p5acc.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        pass
    else:
        raise ValueError("Unsupported model type")

    model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    return model


def model_selection1(name):
    model_dir = "./checkpoints/"
    model_path = os.path.join(model_dir, name + '.npy')
    if name == 'tf2torch_adv_inception_v3':
        model = tf_adv_inception_v3
    elif name == 'tf2torch_ens3_adv_inc_v3':
        model = tf_ens3_adv_inc_v3
    elif name == 'tf2torch_ens4_adv_inc_v3':
        model = tf_ens4_adv_inc_v3
    elif name == 'tf2torch_ens_adv_inc_res_v2':
        model = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')
    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        TfNormalize('tensorflow'),
        model.KitModel(model_path).eval().cuda(), )
    return model.cuda()
def model_selection(name):
    if name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif name == "resnet34":
        model = models.resnet34(pretrained=True)
    elif name == "resnet152":
        model = models.resnet152(pretrained=True)
    elif name == "googlenet":
        model = models.googlenet(pretrained=True)
    elif name == "den121":
        model = models.densenet121(pretrained=True)
    elif name == "den201":
        model = models.densenet201(pretrained=True)
    elif name == "convnext":
        model = models.convnext_base(pretrained=True)
    elif name == "swin":
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    elif name == "vim_s_midclstok_80p5acc":
        model = get_vim_model("small")
    elif name == "vim-tiny":
        model = get_vim_model("tiny")
    elif name == "vgg19":
        model = models.vgg19(pretrained=True)
    elif name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif name == "vgg11":
        model = models.vgg11(pretrained=True)
    elif name == "vgg13":
        model = models.vgg13(pretrained=True)
    elif name == "mobile_v2":
        model = models.mobilenet_v2(pretrained=True)
    elif name == "mobile_v3l":
        model = models.mobilenet_v3_large(pretrained=True)
    elif name == "mobile_v3s":
        model = models.mobilenet_v3_small(pretrained=True)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=True)
    elif name == "wrn50":
        model = models.wide_resnet50_2(pretrained=True)
    elif name == "wrn101":
        model = models.wide_resnet101_2(pretrained=True)
    else:
        raise NotImplementedError("No such model!")



    return model.cuda()

if __name__ == "__main__":
    model = model_selection("vim-tiny")
    print(model)