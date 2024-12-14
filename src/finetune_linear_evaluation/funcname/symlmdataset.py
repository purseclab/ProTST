import os
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import (
    LongformerTokenizerFast, 
    RobertaTokenizerFast,
)

class SymlmDataset(Dataset):
    """func name dataset"""
    def __init__(self, args, split):
        self.tokenized_dataset = load_from_disk(os.path.join(args.dataset_pt, split))

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

            self.label_tokenizer = RobertaTokenizerFast(tokenizer_file=args.label_tokenizer_pt,
                                                    bos_token="<s>",
                                                    eos_token="</s>",
                                                    cls_token="<s>",
                                                    sep_token="</s>",
                                                    unk_token="<unk>",
                                                    pad_token="<pad>",
                                                    mask_token="<mask>")
            
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

            self.label_tokenizer = LongformerTokenizerFast(tokenizer_file=args.label_tokenizer_pt,
                                                    bos_token="<s>",
                                                    eos_token="</s>",
                                                    cls_token="<s>",
                                                    sep_token="</s>",
                                                    unk_token="<unk>",
                                                    pad_token="<pad>",
                                                    mask_token="<mask>")

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

            self.label_tokenizer = LongformerTokenizerFast(tokenizer_file=args.label_tokenizer_pt,
                                                    bos_token="<s>",
                                                    eos_token="</s>",
                                                    cls_token="<s>",
                                                    sep_token="</s>",
                                                    unk_token="<unk>",
                                                    pad_token="<pad>",
                                                    mask_token="<mask>")

        else:
            raise Exception('unknown backbone')
        
        self.num_labels = self.label_tokenizer.vocab_size - 4
        args.num_labels = self.num_labels
        args.label_tokenizer = self.label_tokenizer
        args.arch_tokenizer = self.arch_tokenizer
        
    def __len__(self):
        return int(len(self.tokenized_dataset))  # use part of data

    def tensor_conversion(self, input):
        hex_byte = input['funcbyte'].hex()
        hex_byte = ' '.join(hex_byte[idx:idx+2] for idx in range(0, len(hex_byte), 2))
        sample = self.tokenizer(hex_byte, truncation=True, padding='max_length', max_length=self.input_len, return_tensors='pt')
        sample['input_ids'] = sample['input_ids'].squeeze(0)
        sample['attention_mask'] = sample['attention_mask'].squeeze(0)
        sample['arch'] = torch.tensor([self.arch_tokenizer(input['arch'])['input_ids'][0] for _ in range(self.input_len)]).squeeze()
        sample['opti'] = input['opti'] 
        #sample['compiler'] = input['compiler']
        label = input['funcname']
        tokenized_label = [id - 4 for id in self.label_tokenizer.convert_tokens_to_ids(label.split())]

        assert all(label >= 0 for label in tokenized_label), 'special ids appear'

        one_hot_label = F.one_hot(torch.tensor(tokenized_label, dtype=torch.long), num_classes=self.num_labels)
        one_hot_label = one_hot_label.sum(dim=0).to(torch.float)

        assert self.tokenizer.unk_token_id not in sample['input_ids'], 'Unknown token in input_ids'
        assert self.arch_tokenizer.unk_token_id not in sample['arch'], 'Unknown token in arch'

        return sample, one_hot_label

    def __getitem__(self, idx):
        input = self.tokenized_dataset[idx]
                                            
        return self.tensor_conversion(input)   