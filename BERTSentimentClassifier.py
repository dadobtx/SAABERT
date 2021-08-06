from transformers import BertModel
from torch import nn


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

class BERTSentimentClassifier(nn.Module):
        print('Ingreso a clase BERT')
        def __init__(self, n_classes):
            super(BERTSentimentClassifier, self).__init__()
            self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,return_dict=False)
            # Reduce and improve overfitting. Train data and test data generate a similar accuracy
            self.drop = nn.Dropout(p=0.3)
            #hidden size 768 BERT model
            self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

  
        def forward(self, input_ids, attention_mask):
            _, cls_output = self.bert( #cls_output = clasificate token
            input_ids = input_ids,
            attention_mask = attention_mask
            )
            drop_output = self.drop(cls_output)
            output = self.linear(drop_output)
            return output  