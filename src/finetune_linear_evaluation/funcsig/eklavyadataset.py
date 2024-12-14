import os
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import (
    LongformerTokenizerFast, 
    RobertaTokenizerFast,
)

class EklavyaDataset(Dataset):
    """func name dataset"""
    def __init__(self, args, split):
        self.tokenized_dataset = load_from_disk(os.path.join(args.dataset_pt, split))

        # return type
        self.label_list = ['int', 'char', 'void', 'double', 'bool', 'others']
        self.count_list = ['0','1','2','3','4','5','others']
        self.num_args = 1
        args.num_args = self.num_args
        args.label_list = self.label_list
        args.count_list = self.count_list

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
        #hex_byte = ' '.join(self.tokenizer.convert_ids_to_tokens(input['inst_bytes']))
        input['inst_bytes'] = ' '.join(input['inst_bytes'][i:i+2] for i in range(0, len(input['inst_bytes']), 2))
        sample = self.tokenizer(input['inst_bytes'], truncation=True, padding='max_length', max_length=self.input_len, return_tensors='pt')
        sample['input_ids'] = sample['input_ids'].squeeze(0)
        sample['attention_mask'] = sample['attention_mask'].squeeze(0)
        sample['arch'] = torch.tensor([self.arch_tokenizer(input['arch'])['input_ids'][0] for _ in range(self.input_len)]).squeeze()
        sample['opti'] = input['opti'] 
        sample['compiler'] = input['compiler']

        label = [] 

        ret_type = input['ret_type'].lower()
        if ('int' in ret_type) and ('inte' not in ret_type) and ('point' not in ret_type):
            label.append(self.label_list.index('int'))
        elif 'char' in ret_type:
            label.append(self.label_list.index('char'))
        elif 'void' in ret_type:
            label.append(self.label_list.index('void'))
        elif 'double' in ret_type:
            label.append(self.label_list.index('double'))
        elif 'bool' in ret_type:
            label.append(self.label_list.index('bool'))
        else:
            label.append(self.label_list.index('others'))

        label = torch.tensor(label)

        if str(input['num_args']) in self.count_list:
            count_label = torch.tensor(self.count_list.index(str(input['num_args'])))
        else:
            count_label = torch.tensor(self.count_list.index('others'))

        assert self.tokenizer.unk_token_id not in sample['input_ids'], 'Unknown token in input_ids'
        assert self.arch_tokenizer.unk_token_id not in sample['arch'], 'Unknown token in arch'

        return sample, (label, count_label)

    def __getitem__(self, idx):
        input = self.tokenized_dataset[idx]
                                            
        return self.tensor_conversion(input) 