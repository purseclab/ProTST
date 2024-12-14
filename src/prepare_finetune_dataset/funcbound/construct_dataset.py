import os
import re
import pandas as pd
from datasets import Dataset
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split

RE_PATTERN = (
    "(CPU2006|CPU2017|BAP)_"
    + "(Linux|Windows)_"
    + "(GCC4.7.2|GCC5.1.1|GCC9.2.0|ICC14.0.1|VS201x|VS2008|VS2019)_(x86|x64)_"
    + "(O0|O1|O2|O3|Od|Ox|Os)_"
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

def read_file(example):
    with open(example['file'], "r") as f:
        raw_bytes = []
        raw_labels = []
        for i, line in enumerate(f):
            parts = line.strip().split()
            raw_byte = parts[0].lower()
            if len(parts) > 1:
                if parts[1] == 'E':
                    ground_truth = FUNC_END
                elif parts[1] == 'S':
                    ground_truth = FUNC_START
                else:
                    raise Exception('Unknown gt')
            else:
                ground_truth = LABEL_NONE

            raw_bytes.append(raw_byte)
            raw_labels.append(ground_truth)

    return {'raw_bytes': raw_bytes, 'raw_labels': raw_labels}

def chunk(example):
    tokens = example['raw_bytes']
    labels = example['raw_labels']
    token_chunks = [tokens[i:i+len_limit-2] for i in range(0, len(tokens), len_limit-2)]
    label_chunks = [labels[i:i+len_limit-2] for i in range(0, len(labels), len_limit-2)]
    
    return {'bytes': token_chunks, 'labels': label_chunks}

def rebuild_dataset(examples):
    tokens = examples['bytes']
    labels = examples['labels']
    fname = examples['file'][0]

    new_examples = {'chunk':[], 'chunk_label':[], 'chunk_idx':[], 'package':[], 'platform':[], 'compiler':[], 'arch':[], 'opti':[], 'bin_name':[]}
    for tokens_list,labels_list in zip(tokens, labels):
        for idx, (tchunk, lchunk) in enumerate(zip(tokens_list, labels_list)):
            lchunk_padded = [-100] * len_limit
            lchunk_padded[1:min(len(lchunk) + 1, len_limit-1)] = lchunk
            package, platform, compiler, arch, opti, bin_name = parse_fname(fname)

            if arch == 'x86':
                arch = 'x86_32'
            elif arch == 'x64':
                arch = 'x86_64'
            else:
                raise Exception('unknown architecture')

            new_examples['chunk'].append(tchunk)
            new_examples['chunk_label'].append(lchunk_padded)
            new_examples['chunk_idx'].append(idx)
            new_examples['package'].append(package)
            new_examples['platform'].append(platform)
            new_examples['compiler'].append(compiler)
            new_examples['arch'].append(arch)    
            new_examples['opti'].append(opti)    
            new_examples['bin_name'].append(bin_name)      

    return new_examples

def split_dataset(dataset, saveloc):
    df = dataset.to_pandas()
    grouped = df.groupby(['platform', 'package', 'arch', 'opti'])

    train_data = []
    test_data = []

    for _, group in grouped:
        group = group[:int(len(group) * 0.5)]
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
    TEST_RATIO = 0.12
    LABEL_NONE = 0
    FUNC_START = 1
    FUNC_END = 2
    len_limit = int(512)
    data_dir = f'data/funcbound'
    saveloc = f'workdir/4_prepare_finetune_dataset/x86/funcbound/{len_limit}/95_5/33per'

    if '95_5' in saveloc:
        assert saveloc.split('/')[-1] != '95_5', 'need to specify data size'
    
    file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if 'x64' in file]
    dataset = Dataset.from_dict({'file': file_list})
    dataset = dataset.map(read_file, num_proc=cpu_count())
    dataset = dataset.map(chunk, batched=False, num_proc=cpu_count())
    dataset = dataset.map(rebuild_dataset, batched=True, batch_size=1, num_proc=cpu_count(), remove_columns=dataset.column_names)
    split_dataset(dataset, saveloc)
    