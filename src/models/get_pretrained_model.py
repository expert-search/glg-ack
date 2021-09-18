from pathlib import Path

from transformers import TFDistilBertModel, DistilBertConfig
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

def main():
    ## Classifier
    model_name = 'distilbert-base-uncased'
    num_labels = 3

    # Max length of tokens
    max_length = 100

    # Load transformers config
    config = DistilBertConfig.from_pretrained(model_name)
    config.output_hidden_states = True
    config.num_labels = num_labels

    # Load the Transformers DistilBERT model
    transformer_model = TFDistilBertModel.from_pretrained(model_name, config=config)
    bert = transformer_model.layers[0]
    
    # Build your model input
    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    inputs = {'input_ids': input_ids}

    # Load the Transformers BERT model as a layer in a Keras model
    bert_model = bert(inputs)[0]
    dropout = Dropout(config.dropout, name='pooled_output')
    pooled_output = dropout(bert_model, training=False)
    cls_token = pooled_output[:, 0, :]

    # Then build your model output
    label = Dense(
        units=num_labels,
        kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
        name='label'
    )(cls_token)
    outputs = {'label': label}

    # And combine it all in a model object
    model = Model(inputs=inputs, outputs=outputs, name='DistilBERT_MultiClass')
    optimizer = Adam(
        learning_rate=5e-05,
        epsilon=1e-08,
        decay=0.01,
        clipnorm=1.0
    )

    loss = {'label': CategoricalCrossentropy(from_logits = True)}
    metric = {'label': CategoricalAccuracy('accuracy')}

    model.compile(
        optimizer=optimizer,
        loss=loss, 
        metrics=metric
    )

    model_dir = Path(__file__).resolve().parents[2]/'models'
    with open(model_dir/'distilbert_model.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights(model_dir/'distilbert_weights.hdf5')

if __name__ == '__main__':
    main()
