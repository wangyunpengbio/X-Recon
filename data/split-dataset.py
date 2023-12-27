from glob import glob
from os.path import join, isdir
import random


def write_txt(file_list, filename):
    with open(filename, 'w') as f:
        for item in file_list:
            f.write(item)
            f.write("\n")


if __name__ == '__main__':
    random.seed(2024)
    root_path = './data/9-ptx-whole'
    train_file = join(root_path, 'train.txt')
    test_file = join(root_path, 'test.txt')
    file_list = glob(join(root_path, "ptx-HDF5-256", "*"))
    files_list = [item for item in file_list if isdir(item)]
    file_list = [item.split("/")[-1] for item in file_list]
    files_list = sorted(files_list)
    random.shuffle(file_list)
    seperate_index = int(len(file_list) * 0.8)
    train_list = file_list[:seperate_index]
    test_list = file_list[seperate_index:]
    write_txt(train_list, train_file)
    write_txt(test_list, test_file)
