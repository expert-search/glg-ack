## Hard-coded variables for demo day
from datetime import date, timedelta

import numpy as np
import pandas as pd

today = date.today()
past_week = [today - timedelta(days=i) for i in range(8)]

# Randomize sample size
n_tech = np.random.randint(150, 196)
n_heal = np.random.randint(70, 79)

df_tech = pd.read_csv('demo/fortune-tech.csv')
df_heal = pd.read_csv('demo/fortune-heal.csv')

df_tech = df_tech.sample(n_tech)
df_heal = df_heal.sample(n_heal)
df_tech['Label'] = 'Technology'
df_heal['Label'] = 'Healthcare'
df_tech['Date'] = [np.random.choice(past_week) for _ in range(n_tech)]
df_heal['Date'] = [np.random.choice(past_week) for _ in range(n_heal)]

df = pd.concat([df_tech, df_heal])