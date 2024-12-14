import os
import re
import nltk
import random
import pickle
import pandas as pd
from datasets import Dataset
import sentencepiece as spm
from nltk.corpus import wordnet
from collections import Counter
from tokenizers import Tokenizer
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
from nltk.stem.wordnet import WordNetLemmatizer
from elftools.elf.elffile import ELFFile
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

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

def contain_single_character(s):
    contain = False
    for w in s.split():
        if len(w) == 1:
            contain = True
            break
    return contain

def is_whitespace_less_than_20_percentage(s):
    whitespace_count = s.count(' ')
    return whitespace_count / len(s) < 0.2

def func_name_segmentation(word):
    """
    Segment concatenated words into individual words
    """
    res = sp.encode_as_pieces(word)
    res[0] = res[0][1:]
    return res

def get_pos(treebank_tag):
    """
    get the pos of a treebank tag
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None # for easy if-statement

def func_name_preprocessing(func_name):
    """
    Preprocess function name by:
        - tokenize whole name into words
        - remove digits
        - segment concatenated words
        - lemmatize words
    """

    # split whole name into words and remove digits
    func_name = func_name.replace('_', ' ')
    tmp = ''
    for c in func_name:
        if not c.isalpha(): # filter out numbers and other special characters, e.g. '_' and digits
            tmp = tmp + ' '
        elif c.isupper():
            tmp = tmp + ' ' + c
        else:
            tmp = tmp + c
    tmp = tmp.strip()
    tmp = tmp.split(' ')

    res = []
    i = 0
    while i < len(tmp):
        cap = ''
        t = tmp[i]

        # handle series of capital letters: e.g., SHA, MD
        while i < len(tmp) and len(tmp[i]) == 1:
            cap = cap + tmp[i]
            i += 1
        if len(cap) == 0:
            res.append(t)
            i += 1
        else:
            res.append(cap)

    #segment concatenated words
    words = []
    for word in res:
        if not isinstance(word, str) or word == '':
            continue
        splited = func_name_segmentation(word)
        for w in splited:
            if not isinstance(w, str) or w == '':
                continue
            words.append(w)

    # lemmatize words
    final_words = []
    tokens = nltk.pos_tag(words)

    for word, tag in tokens:
        wntag = get_pos(tag)
        if wntag is None:  # not supply tag in case of None
            word = lem.lemmatize(word)
        else:
            word = lem.lemmatize(word, pos=wntag)
        final_words.append(word)
    
    if len(final_words) == 0:
        return None

    resulting_name = ' '.join(final_words)
    return resulting_name.lower()

def process_file(input):
    file_path = input['file'][0]
    new_examples = {'funcbyte':[], 'funcname':[], 'opti':[], 'arch':[], 'compiler':[], 'package':[], 'binname':[]}

    package, compiler, arch, opti, bin_name = parse_fname(file_path)

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        for i in range(len(data)):
            sample = data[i]

            func_name = sample['name']

            if not func_name:
                continue

            if func_name.startswith("_"):
                continue

            processed_funcname = func_name_preprocessing(func_name)

            if not is_whitespace_less_than_20_percentage(processed_funcname): 
                continue

            if contain_single_character(processed_funcname):
                continue

            new_examples['funcbyte'].append(sample['data'])
            new_examples['funcname'].append(processed_funcname)

            new_examples['arch'].append(arch)
            new_examples['opti'].append(opti)
            new_examples['compiler'].append(compiler.split('-')[-1])
            new_examples['package'].append(package)
            new_examples['binname'].append(bin_name.rstrip('.pkl'))

    return new_examples

def save_frequency_and_tokenizer(dataset, saveloc, frequency_saved_location, tokenizer_saved_location, funcnamelst_pth):
    train_dataset = dataset.load_from_disk(os.path.join(saveloc, 'train'))

    with open(funcnamelst_pth, 'w') as f:
        for i in range(len(train_dataset)):
            f.write(train_dataset[i]['funcname'] + '\n')

    # build vocabulary
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["<s>", "<pad>", "</s>", "<mask>", "<unk>"])
    tokenizer.train([funcnamelst_pth], trainer)
    tokenizer.save(tokenizer_saved_location)

    # count frequency
    token_counts = Counter()
    with open(funcnamelst_pth, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            token_counts.update(tokens)

    sorted_token_counts =  token_counts.most_common()

    with open(frequency_saved_location, "w", encoding="utf-8") as f:
        for token, count in sorted_token_counts:
            f.write(f"{token} {count}\n")

def construct_word_frequency(dataset):
    token_counts = Counter()
    for i in range(len(dataset)):
        tokens = dataset[i]['funcname'].strip().split()
        token_counts.update(tokens)

    return token_counts

def filter_with_frequency(batch):
    empty_example = {'funcbyte':[], 'funcname':[], 'arch':[], 'opti':[], 'package':[], 'binname':[]}
    for word in batch['funcname'][0].split():
        if word_frequency[word] < threshold_freq:
            return empty_example
        elif word_frequency[word] > threshold_freq_upper:
            return empty_example

    return batch

def split_dataset(dataset, saveloc):
    df = dataset.to_pandas()
    grouped = df.groupby(['arch', 'compiler'])

    train_data = []
    test_data = []

    for _, group in grouped:
        #group = group[:int(len(group) * 0.65)]
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
    TEST_RATIO = 0.8
    threshold_freq = 10 
    threshold_freq_upper = 10000 
    data_dir = 'data/binkit2/obfus'
    saveloc = 'workdir/4_prepare_finetune_dataset/obf/funcname/1_9'
    lem = WordNetLemmatizer()
    sp = spm.SentencePieceProcessor()
    sp.load('src/4_prepare_finetune_dataset/funcname/segmentation.model')
    funcnamelst_pth = os.path.join(saveloc, 'funcname_list')
    frequency_saved_location = os.path.join(saveloc, "counts.txt")
    tokenizer_saved_location = os.path.join(saveloc, "tokenizer.json")

    if '95_5' in saveloc:
        assert saveloc.split('/')[-1] != '95_5', 'need to specify data size'

    file_list = []
    for root, dirs, files in os.walk(data_dir):
        file_list.extend([os.path.join(root, fpth) for fpth in files if ('x86_64' in fpth) and ('clang-obfus' in fpth)])

    file_list = file_list[:int(len(file_list))]
    
    dataset = Dataset.from_dict({'file': file_list})
    dataset = dataset.map(process_file, batched=True, batch_size=1, num_proc=cpu_count(), remove_columns=dataset.column_names)
    word_frequency = construct_word_frequency(dataset)
    dataset = dataset.map(filter_with_frequency, batched=True, batch_size=1, num_proc=cpu_count())
    split_dataset(dataset, saveloc)
    save_frequency_and_tokenizer(dataset, saveloc, frequency_saved_location, tokenizer_saved_location, funcnamelst_pth)