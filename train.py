'''
Author: supermantx
Date: 2024-09-06 16:45:39
LastEditTime: 2024-09-12 16:41:25
Description: 
'''
"""
train clip model
"""
import os
import datetime

import torch
from torch.optim import AdamW
from tqdm import tqdm

from model import build_model, ClipLoss
from dataset import get_data_loader
from utils.config import get_config, print_config
from utils.log_util import TbLogger
from utils.sundry import adjust_learning_rate


def train_one_epoch(model, criterion, loader, optimizer, tb_logger, epoch, config):
    t = tqdm(loader, desc=f"Epoch {epoch}")
    for idx, (imgs, texts) in enumerate(t):
        lr = adjust_learning_rate(optimizer, epoch + idx / len(loader), config)
        optimizer.zero_grad()
        imgs, texts = imgs.cuda(), texts.cuda()
        image_features, text_features = model(imgs, texts)
        loss = criterion(image_features, text_features)
        loss.backward()
        optimizer.step()

        log_scalars = {"lr": lr, "loss": loss.item()}
        tb_logger.log_scalar(log_scalars, epoch * len(loader) + idx)
        t.desc = f"Epoch {epoch} loss: {loss.item():.4f} lr: {lr:.4f}"

def train(args, config):
    # save log file
    config.log_path = f"./log/{datetime.datetime.now().strftime('%m_%d-%H%M')}_{config.experiment_name}"
    os.makedirs(config.log_path, exist_ok=True)
    tb_logger = TbLogger(config.log_path)

    # load dataset
    loader = get_data_loader(config.dataset)
    # load model
    clip = build_model(config, use_cuda=True)
    criterion = ClipLoss()
    if len(config.train.vision_model_pth) > 0:
        clip.visual_model.load_state_dict(torch.load(config.train.vision_model_pth), strict=False)
    # optimizer
    optimizer = AdamW(clip.parameters(), lr=config.train.max_lr)

    for epoch in range(1, config.train.epochs + 1):
        train_one_epoch(clip, criterion, loader, optimizer, tb_logger, epoch, config.train)
        # save model
        if epoch % config.train.save_interval == 0:
            torch.save(clip.state_dict(), f"{config.log_path}/{config.experiment_name}_{epoch}.pth")


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="face_clip.yaml")
    args = parser.parse_args()
    config = get_config(args.config)
    print_config(config)
    train(args, config)


if __name__ == "__main__":
    main()
