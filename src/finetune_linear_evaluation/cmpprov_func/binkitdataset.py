import os
import torch
from itertools import product
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import (
    LongformerTokenizerFast, 
    RobertaTokenizerFast,
)

class binkitdataset(Dataset):
    """binkit dataset"""
    def __init__(self, args, split):
        self.tokenized_dataset = load_from_disk(os.path.join(args.dataset_pt,split))
        self.dataset_arch = args.dataset_arch
        if 'obf' not in self.dataset_arch:
            self.label_list = [''.join(tup) for tup in product(['gcc', 'clang'], set(self.tokenized_dataset['opti']))]
        else:
            self.label_list = list(set(self.tokenized_dataset['opti']))
        args.label_list = self.label_list

        if args.backbone == 'RoBerta':
            self.input_len = args.input_length
            self.arch_tokenizer = RobertaTokenizerFast(tokenizer_file=args.arch_tokenizer_pt,
                                                            bos_token="[CLS]",
                                                            eos_token="[SEP]",
                                                            cls_token="[CLS]",
                                                            sep_token="[SEP]",
                                                            unk_token="[UNK]",
                                                            pad_token="[PAD]",
                                                            mask_token="[MASK]")
        
            self.tokenizer = RobertaTokenizerFast(tokenizer_file=args.tokenizer_pt,
                                                        bos_token="[CLS]",
                                                        eos_token="[SEP]",
                                                        cls_token="[CLS]",
                                                        sep_token="[SEP]",
                                                        unk_token="[UNK]",
                                                        pad_token="[PAD]",
                                                        mask_token="[MASK]")
        elif args.backbone == 'Longformer':
            self.input_len = int(4 * 1024)
            self.arch_tokenizer = LongformerTokenizerFast(tokenizer_file=args.arch_tokenizer_pt,
                                                            bos_token="[CLS]",
                                                            eos_token="[SEP]",
                                                            cls_token="[CLS]",
                                                            sep_token="[SEP]",
                                                            unk_token="[UNK]",
                                                            pad_token="[PAD]",
                                                            mask_token="[MASK]")

            self.tokenizer = LongformerTokenizerFast(tokenizer_file=args.tokenizer_pt,
                                                            bos_token="[CLS]",
                                                            eos_token="[SEP]",
                                                            cls_token="[CLS]",
                                                            sep_token="[SEP]",
                                                            unk_token="[UNK]",
                                                            pad_token="[PAD]",
                                                            mask_token="[MASK]")

        elif args.backbone == 'MalConv2':
            self.input_len = args.input_length
            self.arch_tokenizer = LongformerTokenizerFast(tokenizer_file=args.arch_tokenizer_pt,
                                                            bos_token="[CLS]",
                                                            eos_token="[SEP]",
                                                            cls_token="[CLS]",
                                                            sep_token="[SEP]",
                                                            unk_token="[UNK]",
                                                            pad_token="[PAD]",
                                                            mask_token="[MASK]")

            self.tokenizer = LongformerTokenizerFast(tokenizer_file=args.tokenizer_pt,
                                                            bos_token="[CLS]",
                                                            eos_token="[SEP]",
                                                            cls_token="[CLS]",
                                                            sep_token="[SEP]",
                                                            unk_token="[UNK]",
                                                            pad_token="[PAD]",
                                                            mask_token="[MASK]")
            
        else:
            raise Exception('unknown backbone')
            

        args.arch_tokenizer = self.arch_tokenizer

    def __len__(self):
        return len(self.tokenized_dataset)

    def tensor_conversion(self, input):

        input['text'] = ' '.join(input['text'][i:i+2] for i in range(0, len(input['text']), 2))
        sample = self.tokenizer(input['text'], truncation=True, padding='max_length', max_length=self.input_len, return_tensors='pt')
        sample['input_ids'] = sample['input_ids'].squeeze(0)
        sample['attention_mask'] = sample['attention_mask'].squeeze(0)
        sample['arch'] = torch.tensor([self.arch_tokenizer(input['arch'])['input_ids'][0] for _ in range(self.input_len)]).squeeze()
        sample['opti'] = input['opti']
        sample['compiler'] = input['compiler']

        if 'obf' not in self.dataset_arch:
            if 'gcc' in input['compiler']:
                label = self.label_list.index('gcc'+input['opti'])
            elif 'clang' in input['compiler']:
                label = self.label_list.index('clang'+input['opti'])
            else:
                raise Exception
        else:
            label = self.label_list.index(input['opti'])

        label = torch.tensor(label)

        assert self.tokenizer.unk_token_id not in sample['input_ids'], 'Unknown token in input_ids'
        assert self.arch_tokenizer.unk_token_id not in sample['arch'], 'Unknown token in arch'
        
        return sample, label      
    
    def __getitem__(self, idx):
        input = self.tokenized_dataset[idx]
                                            
        return self.tensor_conversion(input) 