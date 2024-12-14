import os
import re
import random
import pandas as pd
from datasets import Dataset
from multiprocessing import cpu_count
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

def read_file(example):
    with open(example['file'], "r") as f:
        raw_bytes = []
        raw_labels = []
        for i, line in enumerate(f):
            parts = line.strip().split()
            raw_byte = '0'+parts[0].lower() if len(parts[0].lower()) == 1 else parts[0].lower()
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

    new_examples = {'chunk':[], 'chunk_label':[], 'platform':[], 'chunk_idx':[], 'package':[], 'compiler':[], 'arch':[], 'opti':[], 'bin_name':[]}
    for tokens_list,labels_list in zip(tokens, labels):
        for idx, (tchunk, lchunk) in enumerate(zip(tokens_list, labels_list)):
            lchunk_padded = [-100] * len_limit
            lchunk_padded[1:min(len(lchunk) + 1, len_limit-1)] = lchunk
            package, compiler, arch, opti, bin_name = parse_fname(fname)
            
            new_examples['chunk'].append(tchunk)
            new_examples['chunk_label'].append(lchunk_padded)
            new_examples['chunk_idx'].append(idx)
            new_examples['package'].append(package)
            new_examples['compiler'].append(compiler)
            new_examples['platform'].append('Linux')
            new_examples['arch'].append(arch)    
            new_examples['opti'].append(opti)    
            new_examples['bin_name'].append(bin_name)      

    return new_examples

def split_dataset(dataset, saveloc):
    df = dataset.to_pandas()
    grouped = df.groupby(['package', 'compiler', 'opti'])

    train_data = []
    test_data = []

    for _, group in grouped:
        if len(group) < 100:
            train_data.append(group.iloc[:1])
            test_data.append(group.iloc[1:])
        else:
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
    TEST_RATIO = 0.9
    LABEL_NONE = 0
    FUNC_START = 1
    FUNC_END = 2
    len_limit = int(512)
    data_dir = 'data/binkit2/x32_instfunc_bound/funcbound'
    saveloc = f'workdir/4_prepare_finetune_dataset/x86/funcbound/{len_limit}/1_9'

    if '95_5' in saveloc:
        assert saveloc.split('/')[-1] != '95_5', 'need to specify data size'

    categories = ['gcc']
    file_lists = {category: [] for category in categories}

    # Categorize files
    for file in os.listdir(data_dir):
        filepath = os.path.join(data_dir, file)
        for category in categories:
            if category in file:
                file_lists[category].append(filepath)

    # Shuffle and select the first 5% from each category
    selected_files = []
    for category, files in file_lists.items():
        random.shuffle(files)
        selected_files.extend(files[:int(len(files)*0.06)])

    dataset = Dataset.from_dict({'file': selected_files})
    dataset = dataset.map(read_file, num_proc=cpu_count())
    dataset = dataset.map(chunk, batched=False, num_proc=cpu_count())
    dataset = dataset.map(rebuild_dataset, batched=True, batch_size=1, num_proc=cpu_count(), remove_columns=dataset.column_names)
    split_dataset(dataset, saveloc)
