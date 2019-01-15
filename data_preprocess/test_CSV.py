import pandas as pd
csv_path = '/Users/yangyucheng/Desktop/海上风场SCADA/dataset/001/201807.csv'
df = pd.read_csv(csv_path)
print(df.head())