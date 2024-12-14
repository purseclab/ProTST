import os
import re
import binascii
import pandas as pd
from datasets import Dataset
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

# matches => package, compiler, arch, opti, bin_name
def parse_fname(bin_path):
    base_name = os.path.basename(bin_path)[:-4] # strip filtered.pickle
    matches = RESTR.search(base_name).groups()
    return matches

def read_file(example):
    with open(example['file'], 'rb') as elf_file:
        elf = ELFFile(elf_file)
        text_section = elf.get_section_by_name('.text')
        text_data = text_section.data().hex()

        return {'raw_bytes' : text_data}

def chunk(example):
    tokens = example['raw_bytes']
    token_chunks = [tokens[i:i+2*(len_limit-2)] for i in range(0, len(tokens), 2*(len_limit-2)) if len(tokens[i:i + 2 * (len_limit - 2)]) == 2 * (len_limit - 2)]
    
    return {'bytes': token_chunks}

def rebuild_dataset(examples):
    tokens = examples['bytes'][0]
    fname = examples['file'][0]

    package, compiler, arch, opti, bin_name = parse_fname(fname)

    new_examples = {'chunk':[], 'chunk_idx':[], 'compiler':[], 'arch':[], 'opti':[], 'bin_name':[]}
    for idx, chunk in enumerate(tokens):
        new_examples['chunk'].append(' '.join(chunk[i:i+2] for i in range(0, len(chunk), 2)))
        new_examples['chunk_idx'].append(idx)
        new_examples['compiler'].append(compiler)
        new_examples['arch'].append(arch)
        new_examples['opti'].append(opti)
        new_examples['bin_name'].append(bin_name)

    return new_examples

def split_dataset(dataset, saveloc):
    df = dataset.to_pandas()
    grouped = df.groupby(['compiler', 'arch', 'opti'])

    train_data = []
    test_data = []

    for _, group in grouped:
        group = group[:int(len(group)*0.011)]
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
    TEST_RATIO = 0.33
    len_limit = int(512)
    data_dir = 'data/mlm/binutils'
    saveloc = f'workdir/4_prepare_finetune_dataset/x86/mlm/{len_limit}/95_5/5k'

    if '95_5' in saveloc:
        assert saveloc.split('/')[-1] != '95_5', 'need to specify data size'

    file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if 'x86_32' in file]
    dataset = Dataset.from_dict({'file': file_list})
    dataset = dataset.map(read_file, num_proc=cpu_count())
    dataset = dataset.map(chunk, batched=False, num_proc=cpu_count())
    dataset = dataset.map(rebuild_dataset, batched=True, batch_size=1, num_proc=cpu_count(), remove_columns=dataset.column_names)
    split_dataset(dataset, saveloc)


            
            

    

