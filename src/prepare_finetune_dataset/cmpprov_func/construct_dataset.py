import os
import re
import pickle
import binascii
import pandas as pd
from datasets import Dataset
from multiprocessing import cpu_count
from elftools.elf.elffile import ELFFile
from sklearn.model_selection import train_test_split

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_file(input):
    file_path = input['file'][0]
    new_examples = {'package':[], 'compiler':[], 'arch':[], 'opti':[], 'bin_name':[], 'text':[]}

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        for i in range(len(data)):
            sample = data[i]
            new_examples['package'].append(sample['package'])
            new_examples['compiler'].append(sample['compiler'])
            new_examples['arch'].append(sample['arch'])
            new_examples['opti'].append(sample['opti'])
            new_examples['bin_name'].append(sample['bin_name'])
            new_examples['text'].append(sample['data'].hex())

    return new_examples

def split_dataset(dataset, saveloc):
    df = dataset.to_pandas()
    grouped = df.groupby(['arch', 'opti', 'compiler'])

    train_data = []
    test_data = []

    for _, group in grouped:
        group = group[:int(len(group) * 0.03)]
        train, test = train_test_split(group, test_size=TEST_RATIO)
        train_data.append(train)
        test_data.append(test)

    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_save_loc = os.path.join(saveloc, 'train')
    test_save_loc = os.path.join(saveloc, 'test')
    
    create_directory(train_save_loc)
    create_directory(test_save_loc)

    train_dataset.save_to_disk(train_save_loc)
    test_dataset.save_to_disk(test_save_loc)

if __name__ == "__main__":
    TEST_RATIO = 0.99
    data_dir = 'data/pretrain/binkit2_pickle'
    saveloc = 'workdir/4_prepare_finetune_dataset/x86/cmpprov_func/1_99'

    if '95_5' in saveloc:
        assert saveloc.split('/')[-1] != '95_5', 'need to specify data size'

    files = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        files.extend([os.path.join(dirpath, fpth) for fpth in filenames if ('x86_32' in fpth) and ('clang-obfus' not in fpth)])

    dataset = Dataset.from_dict({'file':files})
    dataset = dataset.map(process_file, batched=True, batch_size=1, num_proc=cpu_count(), remove_columns=dataset.column_names) 
    split_dataset(dataset, saveloc)



            
            

    

