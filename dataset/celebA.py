import os
import re

import cv2 as cv
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.dictionary import Dictionary


class CelebA(Dataset):
    def __init__(self, dataset_root, anno_file, is_preprocessed=False, transform=None):
        self.img_root = os.path.join(dataset_root, "CelebA-HQ-img")
        self.transform = transform
        self.anno_file = anno_file
        self.img_idx = []

        if not is_preprocessed:
            assert os.path.isfile(anno_file), f"File not found: {anno_file}"
            self.dictionary = Dictionary(self.anno_file)
            self.__preprocess()
        self.__complete_check()

        self.table = pd.read_table(self.anno_file, sep=' ', header=None)
        self.table.set_index(0, inplace=True)
        self.table = self.table[1]

    def __complete_check(self):
        """
        file completeness check, after check return img_idx
        """
        img_files = os.listdir(self.img_root)
        t = tqdm(total=len(img_files), desc="Completeness check.....")
        img_loss_num = 0
        with open(self.anno_file, "r") as f:
            line = f.readline()
            while line:
                if re.match("^\d+\.jpg", line):
                    t.update()
                    img_idx = line.split(" ")[0]
                    if img_idx not in img_files:
                        img_loss_num += 1
                        continue
                    self.img_idx.append(img_idx)
                line = f.readline()
        t.close()
        print(f"lack of {img_loss_num} images, {len(img_files) - len(self.img_idx)} images no annotation")
        print("Completeness check finished")

    def __preprocess(self):
        """
        preprocess annotation file
        use the build dictionary to convert the text to index and saves the index in local storage

        anno_file's content is like:
        img_name1.jpg description text1
        img_name2.jpg description text2
        ... ...
        """
        anno_file = open(self.anno_file, "r")
        processed_anno_file = open(self.anno_file.replace(".tx", "_processed.tx"), "w")
        line = anno_file.readline()
        descriptions_temp = []
        t = tqdm(total=(len(open(self.anno_file).readlines())), desc="Preprocessing.....")
        while line:
            if re.match("^\d+\.jpg", line):
                if len(descriptions_temp) > 0:
                    self.__write_cache(processed_anno_file, descriptions_temp)
                    descriptions_temp = []
                processed_anno_file.write(line.split(" ")[0] + " ")
                temp = line.split(" ")[1:]
            else:
                temp = line.split(" ")
            temp = list(map(lambda x: x.rstrip(".,!?\n").lower(), temp))
            descriptions_temp.extend(temp)
            t.update()
            line = anno_file.readline()
        if len(descriptions_temp) > 0:
            self.__write_cache(processed_anno_file, descriptions_temp)

        self.anno_file = self.anno_file.replace(".tx", "_processed.tx")
        anno_file.close()
        processed_anno_file.close()
        t.close()
        print("Preprocessing finished")

    def __write_cache(self, processed_anno_file, descriptions_temp):
        write_cache = [str(self.dictionary["<begin>"])]
        for word in descriptions_temp:
            if word != '' and self.dictionary.has_word(word):
                write_cache.append(str(self.dictionary[word]))
        write_cache.append(str(self.dictionary["<end>"]))
        processed_anno_file.write("[" + ",".join(write_cache) + "]\n")

    def __getitem__(self, idx):
        img_name = self.img_idx[idx]
        img = cv.imread(str(os.path.join(self.img_root, img_name)))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        texts = torch.tensor(eval(self.table.loc[img_name]), dtype=torch.long)
        return img, texts

    def __len__(self):
        return len(self.img_idx)



if __name__ == '__main__':
    CelebA("/home/jiangda/tx/data/CelebAMask-HQ_224x224/CelebA-HQ-img",
           "/home/jiangda/tx/data/CelebAMask-HQ_224x224/faces.tx", False)
