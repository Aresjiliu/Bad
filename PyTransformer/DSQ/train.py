import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn


from ..transformers.torchTransformer import TorchTransformer
from ..transformers.quantize import QConv2d, QuantConv2d
from .DSQConv import DSQConv


def transfer_quan(args, model):
    if args.quantize is not None:
        transformer = TorchTransformer()
        if args.quantize == "uniform":
            print("Using Uniform Quantization...")
            print(args.quantize_input)
            if args.quantize_input:
                print("Quaintization Input !!!")
                transformer.register(nn.Conv2d, QuantConv2d)
            else:
                print("No Quaintization Input !!!")
                transformer.register(nn.Conv2d, QConv2d)

        else:
            print("Using DSQConv...")
            transformer.register(nn.Conv2d, DSQConv)
        model = transformer.trans_layers(model)

        # set quan bit
        # current use num_bit
        print("Setting target quanBit to {} bit".format(args.quan_bit))
        model = set_quanbit(model, args.quan_bit)
        print("Setting Quantization Input : {} ".format(args.quantize_input))
        model = set_quanInput(model, args.quantize_input)
        print("Setting target quanBit to {} bit".format(args.quan_bit))
        model = set_quanbit(model, args.quan_bit)
        print("Setting Quantization Input : {} ".format(args.quantize_input))
        model = set_quanInput(model, args.quantize_input)

    return model



def set_quanbit(model, quan_bit=8):
    for module_name in model._modules:
        if len(model._modules[module_name]._modules) > 0:
            set_quanbit(model._modules[module_name], quan_bit)
        else:
            if hasattr(model._modules[module_name], "num_bit"):
                setattr(model._modules[module_name], "num_bit", quan_bit)
    return model


def set_quanInput(model, quan_input=True):
    for module_name in model._modules:
        if len(model._modules[module_name]._modules) > 0:
            set_quanInput(model._modules[module_name], quan_input)
        else:
            # for DSQ
            if hasattr(model._modules[module_name], "quan_input"):
                setattr(model._modules[module_name], "quan_input", quan_input)
    return model

