'''
Author: supermantx
Date: 2024-09-12 16:43:25
LastEditTime: 2024-09-12 17:18:19
Description: find the attribute of promote dataset , the number of vocabulary, the max length of text and so on 
'''
import pandas as pd 

# from dataset.dictionary import Dictionary

# dictionary = Dictionary("/home/jiangda/tx/data/CelebAMask-HQ_224x224/faces.tx")
# print(len(dictionary))
table = pd.read_table("/home/jiangda/tx/data/CelebAMask-HQ_224x224/faces_processed.tx", sep=' ', header=None)
table.set_index(0, inplace=True)
table = table[1:]
texts = table[1].to_list()
max_length = 0
for text in texts:
    max_length = max(max_length, len(text.split(",")))
print(max_length)