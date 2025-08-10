# utils.py
import glob
import os
import unicodedata
import string
import torch
import random

# 所有允许的字符（这里是a-z + 大写 + 一些标点）
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

# 将 Unicode 字符转为 ASCII（去掉重音符号等）
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS
    )

# 读取一个文件，并转换为名字列表
def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

# 加载数据：返回 {类别: 名字列表} 和 类别列表
def load_data():
    category_lines = {}
    all_categories = []
    # 假设数据在 ./data/names/ 目录下
    for filename in glob.glob(os.path.join('data', 'names', '*.txt')):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines
    return category_lines, all_categories

# 将单个字母转成 one-hot Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][ALL_LETTERS.find(letter)] = 1
    return tensor

# 将一个名字转成 (seq_len, 1, N_LETTERS) Tensor
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for li, letter in enumerate(line):
        tensor[li][0][ALL_LETTERS.find(letter)] = 1
    return tensor

# 随机获取一个 (类别, 名字, 类别Tensor, 名字Tensor)
def random_training_example(category_lines, all_categories):
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor
