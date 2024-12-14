import os
import re
import pickle
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

def add_spaces_around_operators(op):
    """
    Add spaces around operators and brackets within the operand.
    """
    # Add spaces around brackets, plus, minus, and other operators
    op = re.sub(r'([\[\]\+\-\*/])', r' \1 ', op)  # Add spaces around specific operators
    op = re.sub(r'\s+', ' ', op)  # Remove extra spaces for cleanup
    return op.strip()

def parse_instruction(ins):
    ins = re.sub('\s+', ', ', ins, 1)

    parts = ins.split(', ')
    operand = []
    token_lst = []

    if len(parts) > 1:
        operand = parts[1:]
    token_lst.append(parts[0])

    for op in operand:
        # Detect and process hex addresses
        hex_match = re.match(r'0x[0-9A-Fa-f]+', op)
        if hex_match and 6 < len(hex_match.group(0)) < 15:
            token_lst.append("address")
        else:
            # Add spaces around operators
            token_lst.append(add_spaces_around_operators(op))

    return ' '.join(token_lst)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_fname(bin_path):
    base_name = os.path.basename(bin_path)
    matches = RESTR.search(base_name).groups()
    return matches


def process_file(input):
    file_path = input['file'][0]
    new_examples = {'disassembly':[], 'num_args':[], 'ret_type':[], 'funcname':[], 'opti':[], 'arch':[], 'compiler':[], 'package':[], 'bin_name':[]}

    package, compiler, arch, opti, bin_name = parse_fname(file_path)

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        for i in range(len(data)):
            sample = data[i]
            new_examples['disassembly'].append([parse_instruction(ins) for ins in sample['disassembly']])
            new_examples['num_args'].append(len(sample['args']))
            new_examples['ret_type'].append(sample['ret_type'])
            new_examples['funcname'].append(sample['name'])

            new_examples['arch'].append(arch)
            new_examples['opti'].append(opti)
            new_examples['compiler'].append(compiler)
            new_examples['package'].append(package)
            new_examples['bin_name'].append(bin_name.rstrip('filtered.pkl'))

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

    # Define the desired distribution
    target_distribution = {
        'int': 0.1,
        'char': 0.05,
        'void': 0.6,
        'double': 0.01,
        'bool': 0.06,
        'others': 0.18
    }

    total_samples = 100000
    train_data = []
    test_data = []

    for ret_type, proportion in target_distribution.items():
        target_samples = int(total_samples * proportion)
        class_samples = df[df['ret_type'] == ret_type]

        if len(class_samples) > target_samples:
            sampled_class = class_samples.sample(target_samples, random_state=42)
        else:
            sampled_class = class_samples

        train, test = train_test_split(sampled_class, test_size=TEST_RATIO, random_state=42)
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
    data_dir = 'data/raw-data/binkit2_pickle_diassembly'
    saveloc = 'disassembly'

    file_list = []
    for root, dirs, files in os.walk(data_dir):
        file_list.extend([os.path.join(root, fpth) for fpth in files if ('x86_64' in fpth) and ('clang-obfus' not in fpth)])

    # for 1_9
    file_list = file_list[:int(len(file_list))]
    
    dataset = Dataset.from_dict({'file': file_list})
    dataset = dataset.map(process_file, batched=True, batch_size=1, num_proc=cpu_count(), remove_columns=dataset.column_names)
    split_dataset(dataset, saveloc)