import sys
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    RobertaConfig,
    RobertaForSequenceClassification
)

class CompilerProvenceClassifier(nn.Module):
    def __init__(self, args):
        super(CompilerProvenceClassifier, self).__init__()
        self.backbone = args.backbone.lower()

        label_list = args.label_list
        ids = [i for i in range(len(label_list))]
        id2label =  {key : value for key, value in zip(ids, label_list)}
        label2id = {key : value for key, value in zip(label_list, ids)}
        num_labels = len(label_list)

        if self.backbone == 'longformer' or self.backbone == 'roberta':
            config = RobertaConfig(
                num_labels=num_labels,
                vocab_size = args.len_tokenizer,
                hidden_size = args.projection_dim,
                num_hidden_layers = args.layers,
                max_position_embeddings = args.input_length+2,
                id2label=id2label,
                label2id=label2id,
            )

            self.model = RobertaForSequenceClassification(config)

            if args.finetune_freeze:
                print('Enocder model freezed, parameters would not updated during finetuning')
                for param in self.model.roberta.parameters():
                    param.requires_grad = False

        elif self.backbone == 'malconv2':
            self.model = MalConv(window_size=7, emb_size=args.projection_dim, num_layers=args.layers, id2label=id2label, label2id=label2id)
            self.dense = nn.Linear(in_features=args.projection_dim, out_features=args.projection_dim, bias=True)
            self.dropout = nn.Dropout(p=0.1, inplace=False)
            self.out_proj = nn.Linear(in_features=args.projection_dim, out_features=num_labels, bias=True)

        else:
            raise Exception('unknown backbone')  

    def forward(self, x):
        
        if self.backbone == 'longformer':
            input_embeddings = self.model.longformer.embeddings(x['input_ids'])

            outputs = self.model.longformer(inputs_embeds = input_embeddings, attention_mask = x['attention_mask'])
            sequence_output = outputs.last_hidden_state
            logits = self.model.classifier(sequence_output)
        
        elif self.backbone == 'roberta':
            input_embeddings = self.model.roberta.embeddings(x['input_ids'])

            outputs = self.model.roberta(inputs_embeds = input_embeddings, attention_mask = x['attention_mask'])
            sequence_output = outputs.last_hidden_state
            logits = self.model.classifier(sequence_output) 

        elif self.backbone == 'malconv2':
            output = self.model(x['input_ids'])
            output = self.dense(output)
            output = self.dropout(output)
            logits = self.out_proj(output)

        else:
            raise Exception('unknown backbone')    
        
        return logits
