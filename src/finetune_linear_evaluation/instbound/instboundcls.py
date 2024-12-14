import sys
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    RobertaConfig,
    RobertaForTokenClassification
)
sys.path.insert(0,"src/MalConv")
from MalConvML import MalConvTokenLevel

class InstBoundcls(nn.Module):
    def __init__(self, args):
        super(InstBoundcls, self).__init__()

        self.backbone = args.backbone.lower()
        id2label = {0:'Start', 1: 'Byte'} 
        label2id = {'Start':0, 'Byte':1}

        if self.backbone == 'longformer' or self.backbone == 'roberta':
            config = RobertaConfig(
                num_labels=2,
                vocab_size = args.len_tokenizer,
                hidden_size = args.projection_dim,
                num_hidden_layers = args.layers,
                max_position_embeddings = args.input_length+2,
                id2label=id2label,
                label2id=label2id,
            )

            self.model = RobertaForTokenClassification(config)

            if args.finetune_freeze:
                print('Enocder model freezed, parameters would not updated during finetuning')
                for param in self.model.roberta.parameters():
                    param.requires_grad = False
        
        elif self.backbone == 'malconv2': 
            self.model = MalConvTokenLevel(window_size=7, emb_size=args.projection_dim, num_layers=args.layers, id2label=id2label, label2id=label2id)
            self.droppout = nn.Dropout(p=0.1)
            self.classifier = nn.Linear(in_features=args.projection_dim, out_features=2, bias=True)

        else:
            raise Exception('unknown backbone')        
    
    def forward(self, x):

        if self.backbone == 'longformer':
            input_embeddings = self.model.longformer.embeddings(x['input_ids'])

            outputs = self.model.longformer(inputs_embeds = input_embeddings, attention_mask = x['attention_mask'])
            sequence_output = outputs.last_hidden_state
            sequence_output = self.model.dropout(sequence_output)
            logits = self.model.classifier(sequence_output)
        
        elif self.backbone == 'roberta':
            input_embeddings = self.model.roberta.embeddings(x['input_ids'])

            outputs = self.model.roberta(inputs_embeds = input_embeddings, attention_mask = x['attention_mask'])
            sequence_output = outputs.last_hidden_state
            sequence_output = self.model.dropout(sequence_output)
            logits = self.model.classifier(sequence_output)

        elif self.backbone == 'malconv2':
            output = self.model(x['input_ids'])
            output = self.droppout(output)
            logits = self.classifier(output)

        else:
            raise Exception('unknown backbone')

        return logits 