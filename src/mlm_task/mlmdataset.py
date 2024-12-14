import os
import torch
import random
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import (
    LongformerTokenizerFast, 
    RobertaTokenizerFast,
)

class MLMDataset(Dataset):
    """inst boundary dataset"""
    def __init__(self, args, split):
        self.tokenized_dataset = load_from_disk(os.path.join(args.dataset_pt, split))
        self.label_list = [f'{i:02x}' for i in range(256)]
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


    def mask_raw_bytes(self, raw_bytes_list):
        total_bytes = len(raw_bytes_list)
        num_to_mask = int(total_bytes * 0.2)  
        num_to_mask_with_token = int(num_to_mask * 0.5) 
        num_to_mask_with_random = num_to_mask - num_to_mask_with_token 

        masked_bytes_list = raw_bytes_list.copy()

        indices_to_mask_with_token = random.sample(range(total_bytes), num_to_mask_with_token)
        for index in indices_to_mask_with_token:
            masked_bytes_list[index] = '[MASK]'

        indices_to_mask_with_random = random.sample(set(range(total_bytes)) - set(indices_to_mask_with_token), num_to_mask_with_random)
        for index in indices_to_mask_with_random:
            random_byte = random.choice(self.label_list)
            masked_bytes_list[index] = random_byte

        indices_to_mask = indices_to_mask_with_token + indices_to_mask_with_random

        return masked_bytes_list, indices_to_mask

    def tensor_conversion(self, input):
        masked_chunk, indices_to_mask = self.mask_raw_bytes(input['chunk'].split(' '))
        
        sample = self.tokenizer(' '.join(masked_chunk), truncation=True, padding='max_length', max_length=self.input_len, return_tensors='pt')
        sample['input_ids'] = sample['input_ids'].squeeze(0)
        sample['attention_mask'] = sample['attention_mask'].squeeze(0)
        sample['arch'] = torch.tensor([self.arch_tokenizer(input['arch'])['input_ids'][0] for _ in range(self.input_len)]).squeeze()

        label = self.tokenizer(input['chunk'], truncation=True, padding='max_length', max_length=self.input_len, return_tensors='pt')['input_ids'][:,1:-1].squeeze(0)
        label = label - 5

        assert self.tokenizer.unk_token_id not in sample['input_ids'], f"Unknown token in input_ids, {sample['input_ids']}, {masked_chunk}"
        assert self.arch_tokenizer.unk_token_id not in sample['arch'], f"Unknown token in arch, {sample['arch']}"
        assert (label >= 0).all(), f"wrong target, {label}"
        assert (label < 256).all(), f"wrong target, {label}"

        return sample, label, torch.tensor(indices_to_mask)    

    def __getitem__(self, idx):
        input = self.tokenized_dataset[idx]
                                            
        return self.tensor_conversion(input)   