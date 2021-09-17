## Data and helper functions for demo day
import pickle
from datetime import date, timedelta

import numpy as np
import pandas as pd


with open('demo/queries.txt', 'rt') as f:
    queries = [s.strip() for s in f.readlines()]

today = date.today()
past_week = [today - timedelta(days=i) for i in range(1, 8)]

# Simulate queries over the last week
n_done = np.random.randint(1300, 1700)
df_done = pd.DataFrame({
    'Client': ['Completed'] * n_done,
    'Label': np.random.choice(['Technology', 'Healthcare'], size=n_done, p=[0.6, 0.4]),
    'Date': np.random.choice(past_week, size=n_done),
    'Hashtags': [[] for _ in range(n_done)],
    'Query': [''] * n_done
})

# Simulate client names
clients_tech = pd.read_csv('demo/fortune-tech.csv')['Client']
clients_heal = pd.read_csv('demo/fortune-heal.csv')['Client']
clients_else = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

def add_fake_clients(df):
    df['Client'] = ''
    client_map = {'Technology': clients_tech, 'Healthcare': clients_heal, 'Other': clients_else}
    for label in ['Technology', 'Healthcare', 'Other']:
        df.loc[df['Label'] == label, 'Client'] = np.random.choice(client_map[label], size=len(df[df['Label'] == label]))
    return df

def add_fake_date(df):
    df['Date'] = [today] * len(df)
    return df

def add_fake_metadata(df):
    df = add_fake_clients(df)
    df = add_fake_date(df)
    return df[['Client', 'Label', 'Date', 'Hashtags', 'Query']]

# Hashtag helpers
def get_unique_tags(df):
    return np.unique([t for htags in df['Hashtags'] for t in htags])

def get_filterable_df(df):
    df_tags = pd.DataFrame(df['Hashtags'].to_list(), index=df.index)
    df = df.merge(df_tags, left_index=True, right_index=True)
    return df

def filter_df(df, tags):
    is_expert = 'Expert' in df.columns
    cols_to_show = ['Client', 'Label', 'Date', 'Hashtags', 'Query'] if not is_expert else ['Expert', 'Hashtags']
    idx = 5 if not is_expert else 2
    if not tags:
        return df[cols_to_show].reset_index(drop=True)
    return df.loc[df.iloc[:, idx:].isin(tags).any(axis=1), cols_to_show].reset_index(drop=True)

# Simulate experts
with open('demo/experts.txt', 'rt') as f:
    df_experts = pd.DataFrame(
        {'Expert': [s.strip() for s in f.readlines()]}
    )

def add_random_expertise(df, df_experts):
    tags = get_unique_tags(df)
    df_experts['Hashtags'] = [np.random.choice(tags, size=np.random.randint(1, 3)) for _ in range(len(df_experts))]
    return df_experts
