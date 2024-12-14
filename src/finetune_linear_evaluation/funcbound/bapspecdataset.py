import os
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import (
    LongformerTokenizerFast, 
    RobertaTokenizerFast,
)

class FuncboundDataset(Dataset):
    """func boundary dataset"""
    def __init__(self, args, split):
        self.tokenized_dataset = load_from_disk(os.path.join(args.dataset_pt, split))
        args.label_list = ['None', 'Start', 'End']

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

    def tensor_conversion(self, input):
        sample = self.tokenizer(' '.join(input['chunk']), truncation=True, padding='max_length', max_length=self.input_len, return_tensors='pt')
        sample['input_ids'] = sample['input_ids'].squeeze(0)
        sample['attention_mask'] = sample['attention_mask'].squeeze(0)
        sample['arch'] = torch.tensor([self.arch_tokenizer(input['arch'])['input_ids'][0] for _ in range(self.input_len)]).squeeze()
        sample['opti'] = input['opti']
        sample['package'] = input['package']
        #sample['compiler'] = input['compiler'].split('-')[-1]
        sample['platform'] = input['platform']

        label = torch.tensor(input['chunk_label'])

        assert self.tokenizer.unk_token_id not in sample['input_ids'], 'Unknown token in input_ids'
        assert self.arch_tokenizer.unk_token_id not in sample['arch'], 'Unknown token in arch'

        return sample, label       

    def __getitem__(self, idx):
        input = self.tokenized_dataset[idx]
                                            
        return self.tensor_conversion(input)   