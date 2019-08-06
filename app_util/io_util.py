#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2017/9/13 15:20
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import os
import csv
import sys
import pickle
import os.path


# 读取pickle
def load_pickle(file, file_dir=None):
    try:
        if file_dir:
            file = file_dir + '/' + file
        with open(file, 'rb') as pkl_file:
            return pickle.load(pkl_file)
    except IOError as ioe:
        print(ioe)
        print("nothing get")


# 将文件存储为pickle
def save_as_pickle(file, save_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = output_dir+"/"+save_name
    try:
        with open(save_path, 'wb') as output:
            pickle.dump(file, output)
    except IOError as ioe:
        print("nothing get")
        print(ioe)


# pickle 存读大于4GB文件
def dump_big_file(file, save_name, output_dir):
    data = bytearray(file)
    bytes_out = pickle.dumps(data)
    n_bytes = 2 ** 31
    max_bytes = 2 ** 31 - 1
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = output_dir+"/"+save_name
    with open(save_path, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx: idx+max_bytes])


def load_big_file(file_path):
    bytes_in = bytearray(0)
    max_bytes = 2 ** 31 - 1
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)


# 输出指定文件夹下的指定文件
def get_files(file_dir, extension=""):
    import glob
    if os.path.exists(file_dir):
        return glob.glob(file_dir+"/*"+extension)
    else:
        print("no this dir!")


def write2csv(data, save_name, save_path, header=False, plus=None):
    import numpy as np
    filename = save_path + '/' + save_name + '.csv'
    csv_file = open(filename, 'w+')
    writer = csv.writer(csv_file)
    if plus:
        writer.writerow([plus])
    if header:
        writer.writerow(header)
    if isinstance(data, list) or isinstance(data, np.ndarray):
        for row in data:
            writer.writerow(row)
    if isinstance(data, dict):
        for key, value in data.items():
            value = [str(int(i)) for i in value]
            writer.writerow([key, ",".join(value)])
    csv_file.close()


# 读取本地文件
def read_csv(file_dir):
    assert os.path.exists(file_dir)
    import csv
    data = []
    max_int = sys.maxsize
    decrement = True
    while decrement:
        decrement = False
        try:
            csv.field_size_limit(max_int)
        except OverflowError:
            max_int = int(max_int / 10)
            decrement = True
    with open(file_dir) as fin:
        reader = csv.reader(fin)
        for row in reader:
            data.append(row)
    return data


def file_extension(path):
    return os.path.splitext(path)[1]


def write2xls(data, filename, sheet_name, path, header=None, plus=None):
    from xlwt import Workbook
    wb = Workbook(encoding='UTF-8')
    ws = wb.add_sheet(sheet_name, cell_overwrite_ok=True)
    i_row = 0
    if plus:
        ws.write(i_row, 0, plus)
        i_row += 1
    if header:
        for i in range(len(header)):
            ws.write(i_row, i, header[i])
        i_row += 1
    for msg in data:
        for j in range(len(msg)):
            if len(msg[j]) > 32767:
                continue
            ws.write(i_row, j, msg[j])
        i_row += 1
    wb.save(path+'/'+filename)
