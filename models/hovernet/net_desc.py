import math
from collections import OrderedDict
from termcolor import colored
import numpy as np
import torch
import torch.nn as nn
from .net_utils import (DenseBlock, Net, ResidualBlock, TFSamepaddingLayer,
                        UpSample2x)
from .utils import crop_op, crop_to_shape
import json
from .ulsam import ULSAM

class HoVerNet(Net):
    """Initialise HoVer-Net."""

    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        module_list = [
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == 'fast':  # prepend the padding for `fast` mode
            module_list = [("pad", TFSamepaddingLayer(ksize=7, stride=1))] + module_list

        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 2, stride=1)
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 3, stride=2)
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 4, stride=2)
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 2, stride=2)

        self.d0_ulsam = ULSAM(256, 256, 264, 264, 16)
        self.d1_ulsam = ULSAM(512, 512, 132, 132, 16)
        self.d2_ulsam = ULSAM(1024, 1024, 66, 66, 16)
        self.d3_ulsam = ULSAM(2048, 2048, 33, 33, 16)

        self.u3_ulsam = ULSAM(1024, 1024, 66, 66, 16)
        self.u2_ulsam = ULSAM(512, 512, 60, 60, 16)
        self.u1_ulsam = ULSAM(256, 256, 80, 80, 16)

        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            module_list = [
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0), ])
            )
            return decoder

        ksize = 5 if mode == 'original' else 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()

    def forward(self, imgs):

        imgs = imgs / 255.0  # to 0-1 range to match XY

        if self.training:
            d0 = self.conv0(imgs)  # 1,64,264,264
            d0 = self.d0(d0, self.freeze)  # 1,256,264,264
            d0 = self.d0_ulsam(d0)
            with torch.set_grad_enabled(not self.freeze):
                d1 = self.d1(d0)  # 1,512,132,132
                d1 = self.d1_ulsam(d1)
                d2 = self.d2(d1)  # 1,1024,66,66
                d2 = self.d2_ulsam(d2)
                d3 = self.d3(d2)  # 1,2048,33,,33
                d3 = self.d3_ulsam(d3)

            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]
        else:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0)
            d0 = self.d0_ulsam(d0)
            d1 = self.d1(d0)
            d1 = self.d1_ulsam(d1)
            d2 = self.d2(d1)
            d2 = self.d2_ulsam(d2)
            d3 = self.d3(d2)
            d3 = self.d3_ulsam(d3)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]

        # TODO: switch to `crop_to_shape` ?
        if self.mode == 'original':
            d[0] = crop_op(d[0], [184, 184])
            d[1] = crop_op(d[1], [72, 72])
        else:
            d[0] = crop_op(d[0], [92, 92])
            d[1] = crop_op(d[1], [36, 36])

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = self.u3_ulsam(u3)
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = self.u2_ulsam(u2)
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = self.u1_ulsam(u1)
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict


    def convert_pytorch_checkpoint(self,net_state_dict):
        variable_name_list = list(net_state_dict.keys())
        is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
        if is_in_parallel_mode:
            colored_word = colored("WARNING", color="red", attrs=["bold"])
            print(
                (
                        "%s: Detect checkpoint saved in data-parallel mode."
                        " Converting saved model to single GPU mode." % colored_word
                ).rjust(80)
            )
            net_state_dict = {
                ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
            }
        return net_state_dict


####
def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    return HoVerNet(mode=mode, **kwargs)


def get_last_chkpt_path(prev_phase_dir, net_name):
    stat_file_path = prev_phase_dir + "/stats.json"
    with open(stat_file_path) as stat_file:
        info = json.load(stat_file)
    epoch_list = [int(v) for v in info.keys()]
    last_chkpts_path = "%s/%s_epoch=%d.tar" % (
        prev_phase_dir,
        net_name,
        max(epoch_list),
    )
    return last_chkpts_path

if __name__ == '__main__':
    net = create_model(input_ch=3, nr_types=5,freeze=False, mode='original')
    net.train()
    print(net)
    # net_state_dict = torch.load('/data4/jyh/hover_pannuke/pretrained/ImageNet-ResNet50-Preact_pytorch.tar')
    # net.convert_pytorch_checkpoint(net_state_dict)

    net = net.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    x = torch.zeros((1, 3, 270, 270))

    y = net(x)

    print( )
    for k, a in y.items():
        print(k + ':', a.size(), torch.sum(a))

    exit()
    print(y)

