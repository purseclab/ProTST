import os
import re
import pickle
import pandas as pd
from datasets import Dataset
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split

'''
RE_PATTERN = (
    "(gcc|clang)-"
    + "(64|32)-"
    + "(O0|O1|O2|O3)-"
    + "(.*)"
)
'''
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

def process_file(input):
    file_path = input['file'][0]
    new_examples = {'inst_bytes':[], 'num_args':[], 'ret_type':[], 'funcname':[], 'opti':[], 'arch':[], 'compiler':[], 'package':[], 'bin_name':[]}

    package, compiler, arch, opti, bin_name = parse_fname(file_path)

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        for i in range(len(data)):
            sample = data[i]
            new_examples['inst_bytes'].append(sample['data'].hex())
            new_examples['num_args'].append(len(sample['args']))
            #new_examples['args_type'].append([arg[2] for arg in sample['args']])
            new_examples['ret_type'].append(sample['ret_type'])
            new_examples['funcname'].append(sample['name'])

            new_examples['arch'].append(arch)
            new_examples['opti'].append(opti)
            new_examples['compiler'].append(compiler)
            new_examples['package'].append(package)
            new_examples['bin_name'].append(bin_name.rstrip('.pkl'))

    return new_examples


def reclassify_ret_type(ret_type):
    target_classes = ['int', 'char', 'void', 'double', 'bool']
    if ret_type is None:
        return 'others'
    for type_cls in target_classes:
        if type_cls in ret_type:
            return type_cls
    return 'others'


def split_dataset(dataset, saveloc):
    df = dataset.to_pandas()
    df['ret_type'] = df['ret_type'].apply(reclassify_ret_type)

    #grouped = df.groupby(['arch', 'opti', 'ret_type'])
    grouped = df.groupby('ret_type')

    balanced_data = []
    for class_name, group in grouped:
        max_samples_per_class = min(len(group), 100000 // len(grouped)) 
        balanced_data.append(group.sample(max_samples_per_class, random_state=42))

    balanced_df = pd.concat(balanced_data)

    train_data = []
    test_data = []

    for _, group in balanced_df.groupby(['arch', 'opti', 'ret_type']):
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
    TEST_RATIO = 0.99
    data_dir = 'data/pretrain/binkit2_pickle'
    saveloc = 'workdir/4_prepare_finetune_dataset/x86/funcsig/1_99'

    if '95_5' in saveloc:
        assert saveloc.split('/')[-1] != '95_5', 'need to specify data size'

    file_list = []
    for root, dirs, files in os.walk(data_dir):
        file_list.extend([os.path.join(root, fpth) for fpth in files if ('x86_32' in fpth) and ('clang-obfus' not in fpth)])

    # for 1_9
    file_list = file_list[:int(len(file_list))]
    
    dataset = Dataset.from_dict({'file': file_list})
    dataset = dataset.map(process_file, batched=True, batch_size=1, num_proc=cpu_count(), remove_columns=dataset.column_names)
    split_dataset(dataset, saveloc)