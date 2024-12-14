import os
import pandas as pd
from datasets import Dataset
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_file_and_tokenize(batch):
    new_examples = {'text':[], 'arch':[], 'label':[]}

    for example in batch['file']:
        label = os.path.basename(example).split('.')[-2]
        new_examples['label'].append(int(label))
        with open(example, 'r') as file:
            text = file.read(len_limit * 3)
        new_examples['text'].append(text)
        new_examples['arch'].append('x86_32')

    return new_examples

def split_dataset(dataset, saveloc):
    df = dataset.to_pandas()
    grouped = df.groupby(['label'])

    train_data = []
    test_data = []

    for _, group in grouped:
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
    len_limit = 5000 
    data_dir = 'data/mlwcls'
    saveloc = 'workdir/4_prepare_finetune_dataset/x86/mlwcls/1_9'

    mlw_dataset = Dataset.from_dict({'file': [os.path.join(data_dir, file) for file in os.listdir(data_dir)]})
    mlw_dataset = mlw_dataset.map(read_file_and_tokenize, batched=True, batch_size=1000, num_proc=cpu_count())
    split_dataset(mlw_dataset, saveloc)
    
    