import streamlit as st
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from textwrap import wrap
import pickle
from BERTSentimentClassifier import BERTSentimentClassifier

RANDOM_SEED = 42 #
MAX_LEN = 200 # Text review size
BATCH_SIZE = 16
NCLASSES = 2 # because there are two type of label: positive sentiment and negative sentiment. BERT learns to identify pos and neg comments

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

def classifySentiment(review_text,model):
    encoding_review = tokenizer.encode_plus(
        review_text,
        max_length = MAX_LEN,
        truncation = True,
        add_special_tokens = True,
        return_token_type_ids = False,
        pad_to_max_length = True,
        return_attention_mask = True,
        return_tensors = 'pt'
        )
  
    input_ids = encoding_review['input_ids'].to(device)
    attention_mask = encoding_review['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    print("\n".join(wrap(review_text)))
    if prediction:
        print('Sentimiento predicho: * * * * *')
        
        st.write('Sentiment Analysis: Positive : :+1:')
    else:
        print('Sentimiento predicho: *')
       
        st.write('Sentiment Analysis: Negative: :-1:')

def main():

    RANDOM_SEED = 42 #
    MAX_LEN = 200 # Text review size
    BATCH_SIZE = 16
    NCLASSES = 2 # because there are two type of label: positive sentiment and negative sentiment. BERT learns to identify pos and neg comments

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
            
    model = BERTSentimentClassifier(NCLASSES)
      
    model = pickle.load(open('TrainedModelCPU.sav','rb'))
    
    st.title('SENTIMENT ANALYSIS')
    st.header('REVIEWS AMAZON')
    placeholder = st.empty()
    sentence = placeholder.text_input('Input your review here:')
    submit = st.button('Sentiment Analyze')
    #clear = st.button('Clear')

    if sentence:
        
        if submit:
            classifySentiment(sentence,model)
    
    #if clear:
    #    sentence=placeholder.text_input('Input your review here:',value=' ')

if __name__ == '__main__':
    main()
