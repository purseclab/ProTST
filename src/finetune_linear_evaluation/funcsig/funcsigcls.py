import sys
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    RobertaConfig,
    RobertaForSequenceClassification
)

class FuncSigCls(nn.Module):
    def __init__(self, args):
        super(FuncSigCls, self).__init__()
        num_labels = len(args.label_list)
        self.backbone = args.backbone.lower()
        id2label = {0:'int', 1:'char', 2:'void', 3:'double', 4:'bool', 5:'others'}
        label2id = {'int':0, 'char':1, 'void':2, 'double':3, 'bool':4, 'others':5}

        self.argument_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.projection_dim, args.projection_dim),  
                nn.Dropout(p=0.1, inplace=False),            
                nn.Linear(args.projection_dim, num_labels)    
            )
            for _ in range(args.num_args)
        ])

        self.argument_counter = nn.Sequential(
                nn.Linear(args.projection_dim, args.projection_dim),  
                nn.Dropout(p=0.1, inplace=False),            
                nn.Linear(args.projection_dim, len(args.count_list))    
            )

        if self.backbone == 'longformer' or self.backbone == 'roberta':
            config = RobertaConfig(
                vocab_size = args.len_tokenizer,
                hidden_size = args.projection_dim,
                num_hidden_layers = args.layers,
                max_position_embeddings = args.input_length+2,
                id2label=id2label,
                label2id=label2id
            )

            self.model = RobertaForSequenceClassification(config)
            self.model.classifier = torch.nn.Identity() # remove pooler

            if args.finetune_freeze:
                print('Enocder model freezed, parameters would not updated during finetuning')
                for param in self.model.roberta.parameters():
                    param.requires_grad = False

        elif self.backbone == 'malconv2':
            self.model = MalConv(window_size=7, num_layers=args.layers, emb_size=args.projection_dim, id2label=id2label, label2id=label2id)

        else:
            raise Exception('unknown backbone') 

    def forward(self, x):

        if self.backbone == 'longformer':
            input_embeddings = self.model.longformer.embeddings(x['input_ids'])

            outputs = self.model.longformer(inputs_embeds = input_embeddings, attention_mask = x['attention_mask'])
            last_hidden_state = outputs.last_hidden_state[:,0,:]
            

        elif self.backbone == 'roberta':
            input_embeddings = self.model.roberta.embeddings(x['input_ids'])

            outputs = self.model.roberta(inputs_embeds = input_embeddings, attention_mask = x['attention_mask'])
            last_hidden_state = outputs.last_hidden_state[:,0,:]

        elif self.backbone == 'malconv2':
            last_hidden_state = self.model(x['input_ids'])

        else:
            raise Exception('unknown backbone')

        arg_types_pred = [classifier(last_hidden_state) for classifier in self.argument_classifiers]
        arg_cnt_pred = self.argument_counter(last_hidden_state)

        return arg_types_pred, arg_cnt_pred