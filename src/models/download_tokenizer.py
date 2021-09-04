import pickle
import json
from pathlib import Path

import tensorflow as tf
from transformers import BertTokenizerFast
from transformers.models.bert.modeling_tf_bert import TFBertMainLayer

def main():
    model_dir = Path(__file__).resolve().parents[2]/'models'

    # Tokenizer
    model_name = 'bert-base-uncased'
    with open(model_dir/'bert_tokenizer.pkl', 'wb') as f:
        tkzr = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=model_name)
        pickle.dump(tkzr, f)

    # Hacky Model Conversion
    model = tf.keras.models.load_model(model_dir/'bert_model.hdf5')
    with open(model_dir/'bert_model.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights(model_dir/'bert_weights.hdf5')

if __name__ == '__main__':
    main()