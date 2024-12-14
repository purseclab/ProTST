import os
import re
import pickle
from functools import reduce
from datasets import Dataset
from datasets import concatenate_datasets
from multiprocessing import cpu_count

RE_PATTERN = (
    "(.*)-"
    + "(O0|O1|O2|O3|Os)-"
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

def read_byte_and_tokenize(samples):
    subdir = samples['subdir'][0]
    pkl_paths = [os.path.join(subdir,file) for file in os.listdir(subdir) if file != 'saved_index.pkl']
    new_examples = {'func_text':[], 'func_arch':[], 'func_name':[], 'package_name':[], 'opti': []}
    if len(pkl_paths) != 5:
        return new_examples
    function_list = []
    pkl_dic = {}
    for pkl_path in pkl_paths:
        pkl = pickle.load(open(pkl_path, 'rb'))
        function_list.append(list(pkl.keys()))
        pkl_dic[pkl_path] = pkl
    function_set = [i for i in reduce(lambda x,y: set(x) & set(y), function_list) if type(i) == str]
    for func_name in function_set:
        func_pairs = []
        func_pairs_optis = []
        for fname, pkl in pkl_dic.items():
            hex_byte = pkl[func_name][2]
            func_pairs.append(hex_byte)
            package, opti, _ = parse_fname(fname)
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

    #train_test_split = dataset.shuffle().select(range(100000)).train_test_split(test_size=TEST_RATIO)
    train_test_split = dataset.shuffle().train_test_split(test_size=TEST_RATIO)

    train_test_split['train'].save_to_disk(train_save_loc)
    train_test_split['test'].save_to_disk(test_save_loc)

if __name__ == "__main__":
    TEST_RATIO = 0.05
    data_dir = 'data/funcsim/{}'
    train_dir = 'small_train'
    test_dir = 'small_test'
    saveloc = 'workdir/4_prepare_finetune_dataset/x86/funcsim/95_5/100per'

    train_subdir_list = [os.path.join(data_dir.format(train_dir), subdir) for subdir in os.listdir(data_dir.format(train_dir))]
    test_subdir_list = [os.path.join(data_dir.format(test_dir), subdir) for subdir in os.listdir(data_dir.format(test_dir))]
    train_dataset = Dataset.from_dict({'subdir': train_subdir_list})
    test_dataset = Dataset.from_dict({'subdir': test_subdir_list})
    train_dataset = train_dataset.map(read_byte_and_tokenize, batched=True, batch_size=1, num_proc=cpu_count(), remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(read_byte_and_tokenize, batched=True, batch_size=1, num_proc=cpu_count(), remove_columns=test_dataset.column_names)
    dataset = concatenate_datasets([train_dataset, test_dataset])
    split_dataset(dataset, saveloc)