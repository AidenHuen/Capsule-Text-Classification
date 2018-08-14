import pandas
import config
import pickle
import numpy as np

def to_categorical(labels):
    y = []
    for label in labels:
        y_line = [0 for i in range(config.num_classes)]
        y_line[int(label) - 1] = 1
        y.append(y_line)
    y = np.array(y)
    return y

def get_Y(path):
    data = pandas.read_csv(path,sep="\t")
    y = to_categorical(data.label.values)
    return y

def build_XY(path,vocab,output_path):
    """
    读取path路径下的CSV数据，利用vocab词典构建numpy数据集，并保存至output_path
    :param path:
    :param vocab:
    :param output_path:
    :return:
    """
    data = pandas.read_csv(path,sep="\t")

    y = to_categorical(data.label.values)

    contents = data.content.values
    contents = [content.split(" ") for content in contents]
    x_index = [[vocab[word] for word in content] for content in contents]
    x = np.zeros((x_index.__len__(), config.padding_length), dtype="int")
    for i,index_list in enumerate(x_index):
        for j,index in enumerate(index_list):
            if j < config.padding_length:
                x[i][j] = index
    print(x)
    print(y)
    pickle.dump((x, y), open(output_path,"wb"))



def load_dataset():
    x_train, y_train = pickle.load(open(config.train_pk, "rb"))
    x_dev, y_dev = pickle.load(open(config.dev_pk,"rb"))
    embedding = pickle.load(open(config.word_embed_pk,"rb"))
    return x_train, y_train, x_dev, y_dev, embedding

vocab = pickle.load(open(config.voc_pk,"rb"))
build_XY(config.train_path, vocab, config.train_pk)