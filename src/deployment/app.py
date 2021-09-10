import pickle
from pathlib import Path
from sys import getfilesystemencodeerrors

import spacy
import numpy as np
import tensorflow as tf
import streamlit as st
import pandas as pd
import altair as alt
from transformers.models.distilbert.modeling_tf_distilbert import TFDistilBertMainLayer  # required


## Page Setup
SIMPLE_PREDICTION = 'Simple Prediction'
SHOWCASE = 'Showcase'

with st.sidebar:
    """For today's demo, select from the following choices:"""
    page = st.radio('', (SIMPLE_PREDICTION, SHOWCASE))

## Session State
model_dir = Path('__file__').resolve().parents[2]/'models'
def get_object(fname, fmethod='rb'):
    with open(model_dir/fname, fmethod) as f:
        return pickle.load(f) if '.pkl' in fname else f.read()

def get_bert_clf():
    model = tf.keras.models.model_from_json(get_object('distilbert_model.json', 'r'))
    model.load_weights(model_dir/'distilbert_weights.hdf5')
    return model

session_vars = {'bert_clf', 'bert_tkzr'}
if not session_vars.intersection(set(st.session_state)):
    st.session_state.bert_tkzr = get_object('distilbert_tokenizer.pkl')
    st.session_state.bert_clf = get_bert_clf()
    st.session_state.ner = spacy.load('en_core_web_md')  # spacy.load(model_dir/'kaggleTrainedNER10000')

def ss(key):
    return st.session_state[key]

bert_tkzr = ss('bert_tkzr')
bert_clf = ss('bert_clf')
ner = ss('ner')

## Helpers

def predict(query, clf_type, probas=False):
    clf_types = ['bert']
    if clf_type not in clf_types:
        raise ValueError(f'`clf_type` must be one of {clf_types}')
    if clf_type == 'bert':
        x = bert_tkzr(query, padding='max_length', max_length=100, truncation=True, return_tensors='tf')
        pred = bert_clf(x['input_ids'])
        pred = tf.math.softmax(pred['label']).numpy()  # Convert to probabilities
        pred = pred[:, [0, 2, 1]]

    return pred if probas else np.argmax(pred)

def apply_ner(query, ner):
    pred = ner(query)
    ignore = ['CARDINAL', 'DATE', 'MONEY', 'ORDINAL', 'PERCENT', 'QUANTITY', 'TIME']
    return [ent.text for ent in pred.ents if ent.label_ not in ignore]

def show_df(labels):
    st.write(df[df['Label']].reset_index(drop=True))

## Apps
if page == SIMPLE_PREDICTION:
    """
    # GLG-ACK

    Our model automatically classifies client queries as being either healthcare-related or technology-related. You can test it out below.
    """
    query = st.text_area('Enter your query:')
    if len(query) == 0:
        st.write('Prediction:')
        st.write('Entities:')
    else:
        pred = predict(query, 'bert', probas=True)
        pred = pd.DataFrame(pred, index=['probability'], columns=['Healthcare', 'Technology', 'Other'])
        st.write(pred.style.format(precision=4))
        st.write(f'Prediction: {pred.columns[np.argmax(pred)]}')
        st.write(f'Entities: {apply_ner(query, ner)}')

elif page == SHOWCASE:
    # MORE FEATURES:
    # - Provide DF with ['date_received', 'client', 'label', 'hover_for_full_query', 'NER hashtags'] -- clients can be random from Fortune-500
    # - Hashtags show a DF with connected experts (random names)
    # - Show graph of entity tags over time

    # Demo Setup
    from demo import add_fake_metadata, get_unique_tags, get_filterable_df, filter_df, df_done
    if 'demo_df' not in st.session_state:
        n = 10
        label_map = {0: 'Healthcare', 1: 'Technology', 2: 'Other'}
        with open('demo/all_sentences_list.pkl', 'rb') as f:
            queries = pickle.load(f)[:n]
        preds, tags = [], []
        pbar = st.progress(0)
        for i, item in enumerate(queries):
            pbar.progress((i+1)/n)
            preds.append(label_map[predict(item, 'bert')])
            tags.append(apply_ner(item, ss('ner')))
        df = pd.DataFrame({'Label': preds, 'Hashtags': tags, 'Query': queries})
        st.session_state.demo_df = add_fake_metadata(df)
    df = st.session_state.demo_df
    """
    # Welcome, GLG Agent
    ---
    """
    df_chart = pd.concat([df, df_done])
    df_chart = df_chart[df_chart['Label'] != 'Other']
    c = alt.Chart(df_chart, title='Queries Over the Past Week').mark_line().encode(x=alt.X('monthdate(Date)', title='Day'), y=alt.Y('count(Client)', title='Num Queries'), color='Label').configure_legend(title=None, orient='bottom-left')
    st.altair_chart(c, use_container_width=True)
    df_tech = df[df['Label'] == 'Technology']
    df_heal = df[df['Label'] == 'Healthcare']
    df_else = df[df['Label'] == 'Other']
    f"## There are {len(df_tech)} technology queries available today:"
    with st.expander(''):
        tags_tech = st.multiselect('', get_unique_tags(df_tech), help='In the box below, you can type or select to filter queries by hashtag')
        df_tech = get_filterable_df(df_tech)
        st.write(filter_df(df_tech, tags_tech))
    f"## There are {len(df_heal)} healthcare queries available today:"
    with st.expander(''):
        tags_heal = st.multiselect('', get_unique_tags(df_heal), help='In the box below, you can type or select to filter queries by hashtag')
        df_heal = get_filterable_df(df_heal)
        st.write(filter_df(df_heal, tags_heal))
    f"## There are {len(df_else)} queries needing manual review today:"
    with st.expander(''):
        tags_else = st.multiselect('', get_unique_tags(df_else), help='In the box below, you can type or select to filter queries by hashtag')
        df_else = get_filterable_df(df_else)
        st.write(filter_df(df_else, tags_else))