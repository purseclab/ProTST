import os
import torch
import random
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import (
    LongformerTokenizerFast, 
    RobertaTokenizerFast,
)

class binarycorpdataset(Dataset):
    """func similarity dataset"""
    def __init__(self, args, split):
        self.tokenized_dataset = load_from_disk(os.path.join(args.dataset_pt, split))
        self.split = split

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
        return int(len(self.tokenized_dataset))

    def create_fake(self):
        hex_byte = ' '.join('00' for _ in range(self.input_len))
        sample = self.tokenizer(hex_byte, truncation=True, padding='max_length', max_length=self.input_len, return_tensors='pt')
        sample['input_ids'] = sample['input_ids'].squeeze(0)
        sample['attention_mask'] = sample['attention_mask'].squeeze(0)

        sample['arch'] = torch.tensor([self.arch_tokenizer('x86_64')['input_ids'][0] for _ in range(self.input_len)])
        sample['opti'] = 'Fake'
                       
        assert self.tokenizer.unk_token_id not in sample['input_ids'], 'Unknown token in input_ids'
        assert self.arch_tokenizer.unk_token_id not in sample['arch'], 'Unknown token in arch'

        return sample                                   

    def tensor_conversion(self, input, pos):
        hex_byte = input['func_text'][pos].hex()
        #hex_byte = input['func_text'][pos]
        hex_byte = ' '.join(hex_byte[idx:idx+2] for idx in range(0, len(hex_byte), 2))
        sample = self.tokenizer(hex_byte, truncation=True, padding='max_length', max_length=self.input_len, return_tensors='pt')
        sample['input_ids'] = sample['input_ids'].squeeze(0)
        sample['attention_mask'] = sample['attention_mask'].squeeze(0)

        sample['arch'] = torch.tensor([self.arch_tokenizer(input['func_arch'])['input_ids'][0] for _ in range(self.input_len)]).squeeze()
        if self.split == 'test':
            sample['opti'] = input['opti'][pos]

        assert self.tokenizer.unk_token_id not in sample['input_ids'], 'Unknown token in input_ids'
        assert self.arch_tokenizer.unk_token_id not in sample['arch'], 'Unknown token in arch'

        return sample

    def __getitem__(self, idx):
        if self.split == 'train':
            input = self.tokenized_dataset[idx]
            num_pairs = len(input['func_text'])
            pos=random.randint(0,num_pairs-1)
            pos2=random.randint(0,num_pairs-1)
            while pos2==pos:
                pos2=random.randint(0,num_pairs-1)
            f1=self.tensor_conversion(input, pos)
            f2=self.tensor_conversion(input, pos2)

            return f1, f2

        elif self.split == 'test':
            input = self.tokenized_dataset[idx]
            fs = ()
            for i in range(len(input['func_text'])):
                f = self.tensor_conversion(input, i)
                fs += (f,)

            f1, f2, f3, f4, f5 = self.create_fake(), self.create_fake(), self.create_fake(), self.create_fake(), self.create_fake()
            for f in fs:
                if f['opti'] == 'O0':
                    f1 = f
                elif f['opti'] == 'O1':
                    f2 = f
                elif f['opti'] == 'O2':
                    f3 = f
                elif f['opti'] == 'O3':
                    f4 = f
                elif f['opti'] == 'Os':
                    f5 = f

            return f1, f2, f3, f4, f5
        
        else:
            raise Exception('Unknown split')