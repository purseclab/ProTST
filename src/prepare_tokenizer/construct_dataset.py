import os
import re
import sys
import pickle
from datasets import Dataset
from functools import reduce
from collections import defaultdict
from multiprocessing import cpu_count
sys.path.insert(0,"src")

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

def parse_fname(bin_path):
    base_name = os.path.basename(bin_path)[:-15] 
    matches = RESTR.search(base_name).groups()
    return matches

def read_byte_and_tokenize(samples):
    pkl_paths = samples['pickles'][0]
    new_examples = defaultdict(lambda: defaultdict(list))

    pkl_dic = {}
    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            pkl = pickle.load(f)
            pkl_dic[pkl_path] = pkl
    
    for pkl_path, pkl in pkl_dic.items():
        for func in pkl:
            func_name = func['name']
            hex_byte = func['data']
            func_arch = func['arch']
            file_path = pkl_path.split('/')[-1]
            if 'binkit2_pickle' in pkl_path:
                package, compiler, arch, opti, bin_name = parse_fname(pkl_path)
                func_id = os.path.join(package, bin_name.rstrip('filtered.pickle'), func_name)
            elif 'cornucopia_pickle' in pkl_path:
                func_id = os.path.join(pkl_path.split('_')[-1].rstrip('filtered.pickle'), func_name)
            else:
                raise Exception()

            # Append the function data to the corresponding lists
            new_examples[func_name]['func_bytes'].append(hex_byte)
            new_examples[func_name]['func_arch'].append(func_arch)
            new_examples[func_name]['func_id'].append(func_id)
            new_examples[func_name]['file_pth'].append(file_path)

    # Convert defaultdict to regular dict for the final structure
    examples = {func_name: dict(func_data) for func_name, func_data in new_examples.items()}
    return_examples = {'func_bytes':[], 'func_arch':[], 'func_id':[], 'file_pth':[]}

    for func_name, func_data in examples.items():
        assert all(ele == func_data['func_id'][0] for ele in func_data['func_id'])
        return_examples['func_id'].append(func_data['func_id'][0])
        return_examples['func_arch'].append(func_data['func_arch'])
        return_examples['func_bytes'].append(func_data['func_bytes'])
        return_examples['file_pth'].append(func_data['file_pth'])

    return return_examples

if __name__ == '__main__':
    tardir1 = 'data/pretrain/binkit2_pickle'
    tardir2 = 'data/pretrain/cornucopia_pickle'

    # binkit2
    pack_bins = {}
    for root, dirs, files in os.walk(tardir1):
        files = [file for file in files if file.endswith('filtered.pickle')]
        for file in files:
            bin_path = os.path.join(root, file)
            package, compiler, arch, opti, bin_name = parse_fname(bin_path)
            if package not in pack_bins:
                pack_bins[package] = {}
            if bin_name not in pack_bins[package]:
                pack_bins[package][bin_name] = []
            pack_bins[package][bin_name].append(bin_path)

    binkit_list = []
    for pack in pack_bins.keys():
        for bname, binpaths in pack_bins[pack].items():
            binkit_list.append(binpaths)

    # cornucopia
    cornucopia_list = []
    for subdir in os.listdir(tardir2):
        subdirpth = os.path.join(tardir2, subdir)
        if os.path.isdir(subdirpth):
            cornucopia_list.append([os.path.join(subdirpth, pth) for pth in os.listdir(subdirpth)])

    # obfuscation augmentation
    obfus_aug_list = []
    for bin_list in binkit_list:
        obfus_aug_paths = [path for path in bin_list if ('clang-12.0_x86_32_O0' in path) or ('clang-obfus-all_x86_32_O0'in path) or ('clang-obfus-bcf_x86_32_O0'in path) or ('clang-obfus-fla_x86_32_O0'in path) or ('clang-obfus-sub_x86_32_O0'in path)]
        if len(obfus_aug_paths) != 0:
            obfus_aug_list.append(obfus_aug_paths)

    dataset = Dataset.from_dict({'pickles': obfus_aug_list})
    dataset = dataset.map(read_byte_and_tokenize, batched=True, batch_size=1, num_proc=cpu_count(), remove_columns=dataset.column_names)
    dataset.save_to_disk('workdir/1_prepare_pretrain_dataset/obfuscation_augmentation')
    print(dataset)
    
