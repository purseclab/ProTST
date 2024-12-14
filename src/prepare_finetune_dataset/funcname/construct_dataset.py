import os
import re
import nltk
import random
import pandas as pd
import sentencepiece as spm
from datasets import Dataset
from nltk.corpus import wordnet
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from multiprocessing import cpu_count
from elftools.elf.elffile import ELFFile
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

def read_file_and_tokenize(batch):
    new_examples = {'funcbyte':[], 'funcname':[], 'arch':[], 'opti':[], 'package':[], 'binname':[]}
    file = batch['file'][0] 
    
    with open(file, 'rb') as elff:
        try:
            elffile = ELFFile(elff)
            symtab = elffile.get_section_by_name('.symtab')
            text_section = elffile.get_section_by_name('.text')
        except:
            return new_examples

        text_section_index = None
        for i, section in enumerate(elffile.iter_sections()):
            if section.name == '.text':
                text_section_index = i
                break

        text_offset = text_section['sh_offset']
        text_addr = text_section['sh_addr']

        for symbol in symtab.iter_symbols():
            if (symbol['st_info']['type'] == 'STT_FUNC') and (symbol['st_shndx'] == text_section_index):
                func_name = symbol.name
                func_addr = symbol['st_value']
                func_size = symbol['st_size']

                if not func_name or func_size == 0:
                    continue

                if func_name.startswith("_"):
                    continue

                processed_funcname = func_name_preprocessing(func_name)

                if not is_whitespace_less_than_20_percentage(processed_funcname): 
                    continue

                if contain_single_character(processed_funcname):
                    continue

                offset = text_offset + (func_addr - text_addr)
                elff.seek(offset)
                func_bytes = elff.read(func_size)

                new_examples['funcbyte'].append(func_bytes)
                new_examples['funcname'].append(processed_funcname)

                arch = file.split('/')[2]
                if arch == 'x86':
                    arch = 'x86_32'
                elif arch == 'x64':
                    arch = 'x86_64'
                elif arch == 'arm':
                    arch = 'arm_32'
                elif arch == 'mips':
                    arch = 'mipseb_32'
                else:
                    raise Exception('unkown arch')

                new_examples['arch'].append(arch)
                new_examples['opti'].append(file.split('/')[3])
                new_examples['package'].append(file.split('/')[4])
                new_examples['binname'].append(file.split('/')[5])

    return new_examples

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

def split_dataset(dataset, saveloc):
    df = dataset.to_pandas()
    grouped = df.groupby(['arch', 'opti'])

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
    TEST_RATIO = 0.9
    threshold_freq = 100 #100
    threshold_freq_upper = 10000 #5000
    data_dir = 'data/funcname'
    filelist = []
    lem = WordNetLemmatizer()
    sp = spm.SentencePieceProcessor()
    sp.load('src/4_prepare_finetune_dataset/funcname/segmentation.model')
    saveloc = 'workdir/4_prepare_finetune_dataset/x86/funcname/1_9'
    funcnamelst_pth = os.path.join(saveloc, 'funcname_list')
    frequency_saved_location = os.path.join(saveloc, "counts.txt")
    tokenizer_saved_location = os.path.join(saveloc, "tokenizer.json")
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if 'x86' in root: 
                filelist.append(os.path.join(root, file))

    random.shuffle(filelist)

    dataset = Dataset.from_dict({'file': filelist})
    dataset = dataset.map(read_file_and_tokenize, batched=True, batch_size=1, num_proc=cpu_count(), remove_columns=dataset.column_names)
    word_frequency = construct_word_frequency(dataset)
    dataset = dataset.map(filter_with_frequency, batched=True, batch_size=1, num_proc=cpu_count())
    split_dataset(dataset, saveloc)
    save_frequency_and_tokenizer(dataset, saveloc, frequency_saved_location, tokenizer_saved_location, funcnamelst_pth)