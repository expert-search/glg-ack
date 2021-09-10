## Hard-coded variables for demo day
import pickle
from datetime import date, timedelta

import numpy as np
import pandas as pd

today = date.today()
past_week = [today - timedelta(days=i) for i in range(1, 8)]

# Simulate client names
clients_tech = pd.read_csv('demo/fortune-tech.csv')['Client']
clients_heal = pd.read_csv('demo/fortune-heal.csv')['Client']
clients_else = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

def set_random_clients(df, label):
    client_map = {'Technology': clients_tech, 'Healthcare': clients_heal, 'Other': clients_else}
    df.loc[df['Label'] == label, 'Client'] = np.random.choice(client_map[label], size=len(df[df['Label'] == label]))
    return df

def add_fake_clients(df):
    df['Client'] = ''
    for label in ['Technology', 'Healthcare', 'Other']:
        df = set_random_clients(df, label)
    return df

def add_fake_date(df):
    df['Date'] = [today] * len(df)
    return df

def add_fake_metadata(df):
    df = add_fake_clients(df)
    df = add_fake_date(df)
    return df[['Client', 'Label', 'Date', 'Hashtags', 'Query']]

# Simulate queries over the last week
n_done = np.random.randint(1300, 1700)
df_done = pd.DataFrame({
    'Client': ['Completed'] * n_done,
    'Label': np.random.choice(['Technology', 'Healthcare'], size=n_done, p=[0.6, 0.4]),
    'Date': np.random.choice(past_week, size=n_done),
    'Hashtags': [[] for _ in range(n_done)],
    'Query': [''] * n_done
})

def get_unique_tags(df):
    return np.unique([t for htags in df['Hashtags'] for t in htags])

def get_filterable_df(df):
        df_tags = pd.DataFrame(df['Hashtags'].to_list(), index=df.index)
        df = df.merge(df_tags, left_index=True, right_index=True)
        return df

def filter_df(df, tags):
    cols_to_show = ['Client', 'Label', 'Date', 'Hashtags', 'Query']
    return df.loc[df.iloc[:, 5:].isin(tags).any(axis=1), cols_to_show].reset_index(drop=True)
