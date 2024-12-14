import sys
import torch.nn as nn
import pandas as pd
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    RobertaConfig,
    RobertaForSequenceClassification
)

class MalwareClassifier(nn.Module):
    def __init__(self, args):
        super(MalwareClassifier, self).__init__()
        self.backbone = args.backbone.lower()

        dataset_meta = pd.read_csv(args.dataset_meta)
        num_labels = dataset_meta.shape[0]
        id2label = {
            dataset_meta_row["index"]: dataset_meta_row["name"]
            for dataset_meta_row in dataset_meta.to_dict(orient="records")
        }
        label2id = {label: i for i, label in id2label.items()}


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
            self.model = MalConv(window_size=7, num_layers=args.layers, emb_size=args.projection_dim, id2label=id2label, label2id=label2id)
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
        



