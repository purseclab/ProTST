import os
import re
import pickle
import binascii
import pandas as pd
from functools import reduce
from datasets import Dataset
from collections import defaultdict
from multiprocessing import cpu_count
from elftools.elf.elffile import ELFFile
from sklearn.model_selection import train_test_split

RE_PATTERN = (
    "(.*)_"
    + "(gcc-[.0-9]+|clang-[.0-9]+|"
    + "clang-obfus-[-a-z2]+|"
    + "gcc|clang)_"
    + "((?:x86|arm|mips|mipseb|ppc)_(?:32|64))_"
    + "(O0|O1|O2|O3|Os|Ofast)_"
    + "(.*)"
)

RESTR = re.compile(RE_PATTERN)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_fname(bin_path):
    base_name = os.path.basename(bin_path)
    matches = RESTR.search(base_name).groups()
    return matches

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_file(input):
    pkl_paths = input['file'][0]
    new_examples = {'func_text':[], 'func_arch':[], 'func_name':[], 'package_name':[], 'opti': []}
    if len(pkl_paths) != 5:
        return new_examples
    function_list = []
    pkl_dic = {}
    pkl_dic_funcname_idx = defaultdict(dict)
    for pkl_path in pkl_paths:
        pkl = pickle.load(open(pkl_path, 'rb'))
        funcs = []
        for idx, func_info in enumerate(pkl):
            pkl_dic_funcname_idx[pkl_path][func_info['name']] = idx
            funcs.append(func_info['name'])
        pkl_dic[pkl_path] = pkl
        function_list.append(funcs)
    function_set = [i for i in reduce(lambda x,y: set(x) & set(y), function_list) if type(i) == str]
    for func_name in function_set:
        func_pairs = []
        func_pairs_optis = []
        for fname, pkl in pkl_dic.items():
            hex_byte = pkl[pkl_dic_funcname_idx[fname][func_name]]['data'].hex()
            func_pairs.append(hex_byte)
            #package, opti, _ = parse_fname(fname)
            package, compiler, arch, opti, bin_name = parse_fname(fname) 
            func_pairs_optis.append(opti)
        new_examples['func_text'].append(func_pairs)
        new_examples['func_arch'].append(['x86_64'])
        new_examples['func_name'].append([func_name])
        new_examples['package_name'].append([package])
        new_examples['opti'].append(func_pairs_optis)
    return new_examples

def split_dataset(dataset, saveloc):
    train_save_loc = os.path.join(saveloc, 'train')
    test_save_loc = os.path.join(saveloc, 'test')
    
    create_directory(train_save_loc)
    create_directory(test_save_loc)

    train_test_split = dataset.shuffle().train_test_split(test_size=TEST_RATIO)

    train_test_split['train'].save_to_disk(train_save_loc)
    train_test_split['test'].save_to_disk(test_save_loc)


if __name__ == "__main__":
    TEST_RATIO = 0.8
    data_dir = 'data/binkit2/obfus'
    saveloc = 'workdir/4_prepare_finetune_dataset/obf/funcsim/all/1_99'

    if '95_5' in saveloc:
        assert saveloc.split('/')[-1] != '95_5', 'need to specify data size'

    files_grouped_by_bin_name = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for fpth in filenames:
            if 'x86_64' in fpth and 'clang-obfus-all' in fpth:
                full_path = os.path.join(dirpath, fpth)
                package, compiler, arch, opti, bin_name = parse_fname(full_path) 
                key = package + bin_name
                files_grouped_by_bin_name[key].append(full_path)

    files = []
    for key, file_list in files_grouped_by_bin_name.items():
        files.append(file_list)

    dataset = Dataset.from_dict({'file':files})
    dataset = dataset.map(process_file, batched=True, batch_size=1, num_proc=cpu_count(), remove_columns=dataset.column_names) 
    split_dataset(dataset, saveloc)


