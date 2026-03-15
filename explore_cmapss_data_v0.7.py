"""
CMAPSSデータ（FD004）の探索スクリプト
航空エンジンの劣化データを調査
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# データの読み込み
def load_cmapss_data(dataset='FD004'):
    """CMAPSSデータの読み込み"""
    data_dir = Path('data/CMAPSSData')
    
    # カラム名
    columns = ['unit', 'cycle'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]
    
    # 訓練データ
    train_df = pd.read_csv(
        data_dir / f'train_{dataset}.txt',
        sep='\s+',
        header=None,
        names=columns
    )
    
    # テストデータ
    test_df = pd.read_csv(
        data_dir / f'test_{dataset}.txt',
        sep='\s+',
        header=None,
        names=columns
    )
    
    # RUL（Remaining Useful Life）
    rul_df = pd.read_csv(
        data_dir / f'RUL_{dataset}.txt',
        sep='\s+',
        header=None,
        names=['RUL']
    )
    
    return train_df, test_df, rul_df

print("=" * 80)
print("CMAPSS FD004 DATA EXPLORATION")
print("=" * 80)

# データ読み込み
print("\nLoading data...")
train_df, test_df, rul_df = load_cmapss_data('FD004')

print(f"\nTrain shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")
print(f"RUL shape:   {rul_df.shape}")

print("\n--- COLUMNS ---")
print(train_df.columns.tolist())

print("\n--- TRAIN DATA SAMPLE ---")
print(train_df.head(10))

print("\n--- DATA TYPES ---")
print(train_df.dtypes)

print("\n--- SUMMARY STATISTICS ---")
print(train_df.describe())

print("\n--- UNIT COUNT ---")
print(f"Number of engines in train: {train_df['unit'].nunique()}")
print(f"Number of engines in test:  {test_df['unit'].nunique()}")

print("\n--- CYCLE DISTRIBUTION ---")
print("Train cycles per engine:")
cycles_per_engine = train_df.groupby('unit')['cycle'].max()
print(cycles_per_engine.describe())
print(f"Min cycles: {cycles_per_engine.min()}")
print(f"Max cycles: {cycles_per_engine.max()}")
print(f"Mean cycles: {cycles_per_engine.mean():.1f}")

print("\nTest cycles per engine:")
test_cycles = test_df.groupby('unit')['cycle'].max()
print(test_cycles.describe())

print("\n--- OPERATIONAL SETTINGS ---")
for i in range(1, 4):
    col = f'op_setting_{i}'
    print(f"\n{col}:")
    print(f"  Unique values: {train_df[col].nunique()}")
    print(f"  Range: [{train_df[col].min():.4f}, {train_df[col].max():.4f}]")
    print(f"  Top 5 values: {train_df[col].value_counts().head()}")

print("\n--- SENSOR MEASUREMENTS ---")
sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
print("\nConstant sensors (should be removed):")
for sensor in sensor_cols:
    if train_df[sensor].std() < 0.001:
        print(f"  {sensor}: std={train_df[sensor].std():.6f}")

print("\nSensor correlation with cycle (degradation indicators):")
correlations = []
for sensor in sensor_cols:
    # 各エンジンごとに相関を計算し、平均を取る
    engine_corrs = []
    for unit in train_df['unit'].unique()[:50]:  # 最初の50エンジン
        unit_data = train_df[train_df['unit'] == unit]
        if len(unit_data) > 10:
            corr = unit_data['cycle'].corr(unit_data[sensor])
            if not np.isnan(corr):
                engine_corrs.append(abs(corr))
    
    if engine_corrs:
        avg_corr = np.mean(engine_corrs)
        correlations.append((sensor, avg_corr))

correlations.sort(key=lambda x: x[1], reverse=True)
print("\nTop 10 sensors correlated with degradation (cycle):")
for sensor, corr in correlations[:10]:
    print(f"  {sensor}: {corr:.4f}")

print("\n--- RUL DISTRIBUTION ---")
print(rul_df.describe())
print(f"\nRUL range: [{rul_df['RUL'].min()}, {rul_df['RUL'].max()}]")
print(f"RUL mean: {rul_df['RUL'].mean():.1f}")
print(f"RUL median: {rul_df['RUL'].median():.1f}")

print("\n--- TARGET VARIABLE CREATION STRATEGY ---")
print("""
For training data, we need to create RUL for each cycle:
RUL(t) = max_cycle - current_cycle

For regression, we typically use:
- Piecewise Linear RUL: cap at 125 cycles (common practice)
- This prevents early cycles from dominating the loss

For multi-mechanism degradation analysis:
- HPC Degradation (High Pressure Compressor)
- Fan Degradation
- Combined with 6 operational conditions
- 21 sensor measurements capturing different aspects

Key features for GAMs:
1. Operational settings (3) - operating conditions
2. Sensor measurements (21) - degradation indicators
3. Cycle number - time progression
4. Rolling statistics - trend features
""")

print("\n--- CHECKING FOR MISSING VALUES ---")
print(f"Train missing: {train_df.isnull().sum().sum()}")
print(f"Test missing:  {test_df.isnull().sum().sum()}")

print("\n=" * 80)
print("EXPLORATION COMPLETED")
print("=" * 80)
print("\nNext steps:")
print("1. Remove constant sensors")
print("2. Create RUL target variable (piecewise linear)")
print("3. Create rolling/lag features")
print("4. Normalize sensor values")
print("5. Train GAM models")
