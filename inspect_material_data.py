"""
材料劣化データの構造を調査
"""
import pandas as pd
import numpy as np

# Load with no header to inspect structure
df_raw = pd.read_excel('data/Dataset of material degradation event within process industry.xlsx', 
                       header=None)

print('Raw shape:', df_raw.shape)
print('\n=== First 15 rows (first 5 columns) ===')
for i in range(min(15, len(df_raw))):
    print(f'Row {i}:', list(df_raw.iloc[i, :5]))

# Try different header rows
print('\n=== Trying header=7 ===')
try:
    df = pd.read_excel('data/Dataset of material degradation event within process industry.xlsx', 
                       header=7)
    print('Shape:', df.shape)
    print('Columns:', df.columns.tolist())
    print('\nFirst 3 rows:')
    print(df.head(3))
    if 'Cause' in df.columns:
        print('\nCause distribution:')
        print(df['Cause'].value_counts())
except Exception as e:
    print(f'Error: {e}')

# Try header=8
print('\n=== Trying header=8 ===')
try:
    df = pd.read_excel('data/Dataset of material degradation event within process industry.xlsx', 
                       header=8)
    print('Shape:', df.shape)
    print('Columns:', df.columns.tolist())
    print('\nFirst 3 rows:')
    print(df.head(3))
    if 'Cause' in df.columns:
        print('\nCause distribution:')
        print(df['Cause'].value_counts())
except Exception as e:
    print(f'Error: {e}')
