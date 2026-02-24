# %%
import pandas as pd
import numpy as np
import sklearn as sk

# %%
salary_data = pd.read_csv('2025_salaries.csv', header = 1)
stats = pd.read_csv('nba_2025.txt', sep = ',', encoding = 'latin-1')
# %%
merged_data = pd.merge(salary_data, stats, on = 'Player')

# %%
duplicates = merged_data[merged_data.duplicated(subset = 'Player', keep = False)]
# %%
