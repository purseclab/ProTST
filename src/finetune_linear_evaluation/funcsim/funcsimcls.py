import sys
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    RobertaConfig,
    RobertaForSequenceClassification
)
import torch.nn.functional as F

class funcsimcls(nn.Module):
    def __init__(self, args):
        super(funcsimcls, self).__init__()
        self.backbone = args.backbone.lower()

        if self.backbone == 'longformer' or self.backbone == 'roberta':
            config = RobertaConfig(
                vocab_size = args.len_tokenizer,
                hidden_size = args.projection_dim,
                num_hidden_layers = args.layers,
                max_position_embeddings = args.input_length+2,
            )

            self.model = RobertaForSequenceClassification(config)
            self.model.classifier = torch.nn.Identity() # remove pooler

            self.decoder = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
                nn.Tanh(),
                nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
            )
            
            if args.finetune_freeze:
                print('Enocder model freezed, parameters would not updated during finetuning')
                for param in self.model.roberta.parameters():
                    param.requires_grad = False
        
        elif self.backbone == 'malconv2':
            self.model = MalConv(window_size=7, num_layers=args.layers, emb_size=args.projection_dim)
            self.decoder = nn.Sequential(
                nn.Linear(args.projection_dim, args.projection_dim),
                nn.Tanh(),
                nn.Linear(args.projection_dim, args.projection_dim)
            )

        else:
            raise Exception('unknown backbone')  

    def forward(self, x):

        if self.backbone == 'longformer' or self.backbone == 'roberta':
            input_embeddings = self.model.roberta.embeddings(x['input_ids'])

            outputs = self.model.roberta(inputs_embeds = input_embeddings, attention_mask = x['attention_mask'])
            last_hidden_state = outputs.last_hidden_state

            non_padded_embedding = [last_hidden_state[i][x['attention_mask'][i].bool()] for i in range(last_hidden_state.shape[0])]
            mean_embedding = [torch.mean(non_padded_embedding[i], dim=0) for i in range(len(non_padded_embedding))]
            mean_embedding = torch.stack(mean_embedding)
            func_embedding = self.decoder(mean_embedding)
            normalized_func_embedding = F.normalize(func_embedding, p=2, dim=1)

        elif self.backbone == 'malconv2':
            output = self.model(x['input_ids'])
            func_embedding = self.decoder(output)
            normalized_func_embedding = F.normalize(func_embedding, p=2, dim=1)
            
        else:
            raise Exception('unknown backbone')          
        
        return normalized_func_embedding
        