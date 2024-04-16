from utils.tools import *
# from model.network import *

from torch.cuda.amp import autocast,GradScaler

# import os
import torch
import torch.optim as optim
import time
# import numpy as np
from loguru import logger
from model.HybridHash import HybridHash
# from torch.autograd import Variable
from ptflops import get_model_complexity_info
# from apex import amp
from utils.Hash_loss import HashNetLoss
torch.multiprocessing.set_sharing_strategy('file_system')


def get_config():
    config = {
        "alpha": 0.5,
        # "alpha": 1,


        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 1e-4, "weight_decay": 1e-5}, "lr_type": "step"},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 2.5e-5, "weight_decay": 1e-5}, "lr_type": "step"},
        "info": "[HybridHash]",
        "step_continuation": 20,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 4,
        # "net": AlexNet,
        # "net":ResNet,
        "dataset": "cifar10",
        # "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "imagenet",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        "epoch": 15,
        "test_map": 3,
        "save_path": "save/HybridHash",
        "device": torch.device("cuda:0"),
        "bit_list": [16, 32, 48, 64],
        "pretrained_dir":"checkpoint/jx_nest_base-8bc41011.pth",
        "img_size": 224,
        "patch_size": 4,
        "in_chans": 3,
        "num_work": 4,
    }
    config = config_dataset(config)
    return config


def train_val(config, bit):
    # Prepare model

    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["bit"] = bit
    # net = config["net"](bit).to(device)

    net = HybridHash(config, num_levels=3, embed_dims=(128, 256, 512), num_heads=(4, 8, 16), depths=(2, 2, 15))
    if config["pretrained_dir"] is not None:
        logger.info('Loading:', config["pretrained_dir"])
        state_dict = torch.load(config["pretrained_dir"])
        net.load_state_dict(state_dict, strict=False)
        logger.info('Pretrain weights loaded.')

    net.to(config["device"])

    # 计算模型计算力和参数量（Statistical model calculation and number of parameters）
    flops, num_params = get_model_complexity_info(net,(3,224,224), as_strings=True, print_per_layer_stat=False)
    # logger.info("{}".format(config))
    logger.info("Total Parameter: \t%s" % num_params)
    logger.info("Total Flops: \t%s" % flops)


    # net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    #原声自带apex训练
    scaler = GradScaler()

    criterion = HashNetLoss(config, bit)


    Best_mAP = 0.0


    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        logger.info("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:

            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            with autocast():
                u = net(image)
                loss = criterion(u, label.float(), ind, config)

            train_loss += loss.item()

            #原生自带apex
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss = train_loss / len(train_loader)

        logger.info("\b\b\b\b\b\b\b loss:%.4f" % (train_loss))
        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP, index_img = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch)


if __name__ == "__main__":
    # 原本自己的
    config = get_config()
    # 建立日志文件（Create log file）
    logger.add('logs/{time}' + config["info"] + '_' + config["dataset"] + ' alpha '+str(config["alpha"]) + '.log', rotation='50 MB', level='DEBUG')

    logger.info(config)
    for bit in config["bit_list"]:
        # config["pr_curve_path"] = f"log/alexnet/HashNet_{config['dataset']}_{bit}.json"
        train_val(config, bit)
