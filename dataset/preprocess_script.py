import os
import re

import pandas as pd
from matplotlib import pyplot as plt


class Vocabulary:
    """
    input texts and return a dict contain words frequency
    del common preposition and punctuation
    """

    def __init__(self, file_path, del_propositions=True):
        self.texts = self.open_desc_file(file_path)
        self.vocabulary = {}
        self.propositions = ["a", "an", 'and', "is", "has", "the", "in", "on", "at", "to", "of", "for", "with", "by",
                             "about", "as", "into", "onto", "upon", "out", "off", "up", "down", "or", "have", "has",
                             "over", "under", "through", "from", "between", "among", "around", "after", "but", "so",
                             "before", "during", "since", "while", "till", "until", "against", "without",
                             "within", "along", "across", "behind", "beside", "beyond", "inside", "outside",
                             "underneath", "underneath", "beneath", "throughout", "underneath", "above", "below"]
        self.build_vocabulary()
        # self.data_distribution_analysis("before delete propositions")
        if del_propositions:
            self.delete_propositions()
        # self.data_distribution_analysis("after delete propositions")

    @staticmethod
    def open_desc_file(file_path):
        res = ""
        with open(file_path, "r") as f:
            line = f.readline()
            while line:
                if re.match("^\d+\.jpg", line):
                    res += " ".join(line.split(" ")[1:]).rstrip("\n")
                else:
                    res += line.rstrip("\n")
                line = f.readline()
        return res

    def build_vocabulary(self):
        for word in self.texts.split():
            word = word.rstrip(".,!?").lower()

            if word in self.vocabulary:
                self.vocabulary[word] += 1
            else:
                self.vocabulary[word] = 1

    def delete_propositions(self):
        for pro in self.propositions:
            if pro in self.vocabulary:
                self.vocabulary.pop(pro)

    def get_vocabulary(self):
        return self.vocabulary

    def get_sorted_vocabulary(self):
        return dict(sorted(self.vocabulary.items(), key=lambda item: item[1], reverse=True))


def data_distribution_analysis(plt, vocabulary):
    freqs = [freq for word, freq in vocabulary.get_sorted_vocabulary().items()]
    plt.plot(freqs)
    plt.set_xlabel("word index")
    plt.set_ylabel("word frequency")
    plt.set_xscale("log")
    plt.set_yscale("log")


if __name__ == "__main__":
    vocabulary = Vocabulary("/home/jiangda/tx/data/microsoft/faces.tx", False)
    plt.figure(figsize=(10, 5))
    subplot = plt.subplot(1, 2, 1)
    data_distribution_analysis(subplot, vocabulary)
    vocabulary = Vocabulary("/home/jiangda/tx/data/microsoft/faces.tx")
    subplot = plt.subplot(1, 2, 2)
    data_distribution_analysis(subplot, vocabulary)
    plt.show()
