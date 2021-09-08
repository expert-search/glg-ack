## Hard-coded variables for demo day
from datetime import date, timedelta

import numpy as np
import pandas as pd

today = date.today()
past_week = [today - timedelta(days=i) for i in range(1, 8)]

# Randomize sample size
n_tech = np.random.randint(150, 196)
n_heal = np.random.randint(70, 79)
n_done = np.random.randint(1300, 1700)
n_else = np.random.randint(12, 18)

df_tech = pd.read_csv('demo/fortune-tech.csv')
df_heal = pd.read_csv('demo/fortune-heal.csv')

df_tech = df_tech.sample(n_tech)
df_heal = df_heal.sample(n_heal)
df_tech['Label'] = 'Technology'
df_heal['Label'] = 'Healthcare'
df_tech['Date'] = [today] * n_tech
df_heal['Date'] = [today] * n_heal

# Simulate unconfident predictions
df_else = pd.DataFrame({
    'Client': np.random.choice(['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'], size=n_else),
    'Label': ['Other'] * n_else,
    'Date': [today] * n_else
})

# Simulate queries over the last week
df_done = pd.DataFrame({
    'Client': ['Completed'] * n_done,
    'Label': np.random.choice(['Technology', 'Healthcare'], size=n_done, p=[0.6, 0.4]),
    'Date': np.random.choice(past_week, size=n_done)
})

df_chart = pd.concat([df_tech, df_heal, df_done])
df = pd.concat([df_tech, df_heal, df_else])

with open('demo/entities.txt', 'rt') as f:
    entities = [x.strip() for x in f.readlines()]

df_entities = pd.DataFrame([np.random.choice(entities, size=np.random.randint(2, 5)) for _ in range(len(df))])
df = df.merge(df_entities, left_index=True, right_index=True)