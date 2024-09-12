'''
Author: supermantx
Date: 2024-09-06 11:04:32
LastEditTime: 2024-09-12 15:49:23
Description: 
'''
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset.celebA import CelebA


def get_transform(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor()
        ])


def collate_fn(batch):
    imgs, texts = tuple(zip(*batch))
    max_text_len = max([len(text) for text in texts])
    ret_texts = []
    for text in texts:
        ret_texts.append(torch.cat([text, torch.full((max_text_len - len(text),), -1)], dim=0))
    return torch.stack(imgs, dim=0), torch.stack(ret_texts, dim=0)


def get_data_loader(cfg, is_train=True):
    celebA = CelebA(cfg.dataset_root, cfg.anno_file, transform=get_transform(is_train=is_train), is_preprocessed=cfg.pre_processed)
    return DataLoader(celebA, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn)


