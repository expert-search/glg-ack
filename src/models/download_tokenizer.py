import pickle
import json
from pathlib import Path

import tensorflow as tf
from transformers import BertTokenizerFast, DistilBertTokenizerFast
from transformers.models.bert.modeling_tf_bert import TFBertMainLayer

def main():
    model_dir = Path(__file__).resolve().parents[2]/'models'

    model_name = 'bert-base-uncased'
    with open(model_dir/'bert_tokenizer.pkl', 'wb') as f:
        tkzr = BertTokenizerFast.from_pretrained(model_name)
        pickle.dump(tkzr, f)

    model_name = 'distilbert-base-uncased'
    with open(model_dir/'distilbert_tokenizer.pkl', 'wb') as f:
        tkzr = DistilBertTokenizerFast.from_pretrained(model_name)
        pickle.dump(tkzr, f)

if __name__ == '__main__':
    main()