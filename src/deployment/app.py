import pickle
from pathlib import Path

import tensorflow as tf
import streamlit as st
import pandas as pd
import altair as alt
from transformers.models.bert.modeling_tf_bert import TFBertMainLayer


## Page Setup
SIMPLE_PREDICTION = 'Simple Prediction'
SHOWCASE = 'Showcase'

with st.sidebar:
    """For today's demo, select from the following choices:"""
    page = st.radio('', (SIMPLE_PREDICTION, SHOWCASE))

## Session State
model_dir = Path('__file__').resolve().parents[2]/'models'
def get_object(fname, fmethod='rb'):
    if '.hdf5' in fname:
        return model_dir/fname
    with open(model_dir/fname, fmethod) as f:
        return pickle.load(f) if '.pkl' in fname else f.read()

def get_bert_clf():
    model = tf.keras.models.model_from_json(get_object('bert_model.json', 'r'))
    model.load_weights(get_object('bert_weights.hdf5'))
    return model

session_vars = {'tfidf', 'clf', 'bert_clf', 'bert_tkzr'}
if not session_vars.intersection(set(st.session_state)):
    st.session_state.tfidf = get_object('grail_qa_tfidf.pkl')
    st.session_state.clf = get_object('grail_qa_lr.pkl')
    st.session_state.bert_tkzr = get_object('bert_tokenizer.pkl')
    st.session_state.bert_clf = get_bert_clf()

tfidf = st.session_state.tfidf
clf = st.session_state.clf
bert_tkzr = st.session_state.bert_tkzr
bert_clf = st.session_state.bert_clf

def predict(query, clf_type):
    clf_types = ['bert', 'lr']
    if clf_type not in clf_types:
        raise ValueError(f'`clf_type` must be one of {clf_types}')
    if clf_type == 'bert':
        x = bert_tkzr(query, padding='max_length', max_length=100, truncation=True, return_tensors='tf')
        pred = bert_clf(x['input_ids'])
        pred = tf.math.softmax(pred['label']).numpy()  # Convert to probabilities
    if clf_type == 'lr':
        pred = clf.predict_proba(tfidf.transform([query]))
    
    return pred

## Apps
if page == SIMPLE_PREDICTION:
    """
    # GLG-ACK

    Our model automatically classifies client queries as being either healthcare-related or technology-related. You can test it out below.
    """
    query = st.text_input('Enter your query:')
    if len(query) == 0:
        st.write('_Prediction: _')
    else:
        pred = predict(query, 'bert')
        pred = pd.DataFrame(pred, index=['probability'], columns=['Healthcare', 'Technology'])
        st.write(pred.style.format(precision=4))
        healthcare_proba, technology_proba = pred.Healthcare[0], pred.Technology[0]
        if healthcare_proba < 0.7 and technology_proba < 0.7:
            pred = 'Other'
        else:
            pred = 'Healthcare' if healthcare_proba > technology_proba else 'Technology'
        st.write(f'_Prediction: {pred}_')

elif page == SHOWCASE:
    # MORE FEATURES:
    # - Provide DF with ['date_received', 'client', 'label', 'hover_for_full_query', 'NER hashtags'] -- clients can be random from Fortune-500
    # - Hashtags show a DF with connected experts (random names)
    # - Provide manual review
    # - Show graph of entity tags over time
    from demo import n_tech, n_heal, df
    """
    # Welcome, GLG Agent
    ---
    """
    c = alt.Chart(df, title='Last Week\'s Queries Per Day').mark_line().encode(x=alt.X('monthdate(Date)', title='Date'), y=alt.Y('count(Client)', title='Num Queries'), color='Label').configure_legend(title=None, orient='bottom-left')
    st.altair_chart(c, use_container_width=True)
    f"## There are {n_tech} technology queries available:"
    with st.expander('View'):
        st.write(df[df['Label'] == 'Technology'].reset_index(drop=True))
    f"## There are {n_heal} healthcare queries available:"
    with st.expander('View'):
        st.write(df[df['Label'] == 'Healthcare'].reset_index(drop=True))
