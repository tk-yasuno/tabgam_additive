"""
材料劣化データの詳細調査 - v0.6
複合劣化メカニズムの分析のためのデータ探索
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_excel('data/Dataset of material degradation event within process industry.xlsx', 
                   header=8)

print("=" * 80)
print("MATERIAL DEGRADATION DATA - v0.6")
print("=" * 80)
print(f"\nShape: {df.shape[0]:,} rows, {df.shape[1]} columns")

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
missing_pct = 100 * missing / len(df)
missing_df = pd.DataFrame({
    'Missing': missing,
    'Percentage': missing_pct
}).sort_values('Missing', ascending=False)
print(missing_df[missing_df['Missing'] > 0])

print("\n--- KEY VARIABLES ---")

# Cause (degradation mechanism)
print("\n1. CAUSE (劣化メカニズム)")
print(df['Cause'].value_counts())
print(f"\nUnique causes: {df['Cause'].nunique()}")

# Identify multi-mechanism events
df['is_multi_mechanism'] = df['Cause'].str.contains('\+', na=False)
print(f"\nMulti-mechanism events: {df['is_multi_mechanism'].sum()} ({100*df['is_multi_mechanism'].mean():.2f}%)")
print("\nMulti-mechanism combinations:")
print(df[df['is_multi_mechanism']]['Cause'].value_counts())

# Outcome
print("\n2. OUTCOME (事故結果)")
print(df['Outcome'].value_counts())

# Final Scenario
print("\n3. FINAL SCENARIO (最終シナリオ)")
print(df['Final Scenario'].value_counts())

# Macro-Sector
print("\n4. MACRO-SECTOR (産業セクター)")
print(df['Macro-Sector'].value_counts())

# Injured
print("\n5. INJURED (負傷者)")
print(df['Injured'].value_counts())

# Fatality
print("\n6. FATALITY (死亡者)")
print(df['Fatality'].value_counts())

# Economical losses
print("\n7. ECONOMICAL LOSSES (経済損失)")
print(df['Economical losses'].value_counts())

# Environmental contamination
print("\n8. ENVIRONMENTAL CONTAMINATION (環境汚染)")
print(df['Environmental contamination'].value_counts())

# Age
print("\n9. AGE (設備年齢)")
print(df['Age'].value_counts())

# Equipment involved
print("\n10. EQUIPMENT INVOLVED (関係設備)")
print(df['Equipment involved'].value_counts())

# Substance hazard classification
print("\n11. SUBSTANCE HAZARD CLASSIFICATION")
print(df['Substance hazard classification'].value_counts())

# Date analysis
print("\n--- TEMPORAL ANALYSIS ---")
df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
print(f"Date range: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
print("\nEvents per decade:")
df['Decade'] = (df['Year'] // 10) * 10
print(df['Decade'].value_counts().sort_index())

# Geographic analysis
print("\n--- GEOGRAPHIC ANALYSIS ---")
print("\nContinent distribution:")
print(df['Continent'].value_counts())
print("\nTop 10 countries:")
print(df['Place'].value_counts().head(10))

print("\n--- TARGET VARIABLE CREATION STRATEGY ---")
print("""
事故重大度スコアの作成:
1. Outcome: Accident > LOC > Incident > Near Miss
2. Injured: MF > SF > NI
3. Fatality: MF > SF > NF
4. Economical losses: High > Medium > Low > Unknown
5. Environmental contamination: SED > MMD > ND

複合スコア = 
    outcome_score * 0.3 + 
    injured_score * 0.2 + 
    fatality_score * 0.3 + 
    economic_score * 0.1 + 
    environmental_score * 0.1
""")

# Create severity score
def create_severity_score(row):
    """事故重大度スコアの作成"""
    score = 0.0
    
    # Outcome
    outcome_map = {
        'Near Miss': 1,
        'Incident': 2, 
        'LOC': 3,
        'Accident': 4
    }
    score += outcome_map.get(row['Outcome'], 2) * 0.3
    
    # Injured
    injured_map = {
        'NI': 0,
        'SI': 2,
        'MI': 4
    }
    score += injured_map.get(row['Injured'], 0) * 0.2
    
    # Fatality
    fatality_map = {
        'NF': 0,
        'SF': 3,
        'MF': 5
    }
    score += fatality_map.get(row['Fatality'], 0) * 0.3
    
    # Economical losses
    econ_map = {
        'Unknown': 0,
        'Low': 1,
        'Medium': 2,
        'High': 3
    }
    score += econ_map.get(row['Economical losses'], 0) * 0.1
    
    # Environmental contamination
    env_map = {
        'ND': 0,
        'MMD': 1,
        'SED': 2
    }
    score += env_map.get(row['Environmental contamination'], 0) * 0.1
    
    return score

df['severity_score'] = df.apply(create_severity_score, axis=1)

print("\n--- SEVERITY SCORE STATISTICS ---")
print(df['severity_score'].describe())

print("\n--- SEVERITY BY CAUSE ---")
severity_by_cause = df.groupby('Cause')['severity_score'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
print(severity_by_cause.head(15))

print("\n--- MULTI-MECHANISM vs SINGLE-MECHANISM ---")
print(df.groupby('is_multi_mechanism')['severity_score'].describe())

print("\n=== DATA EXPLORATION COMPLETED ===")
print(f"Output saved: This information will be used for v0.6 preprocessing")
