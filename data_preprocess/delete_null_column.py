import pandas as pd
import os
csv_input_path = '/Users/yangyucheng/Desktop/SCADA/dataset'
csv_output_path = '/Users/yangyucheng/Desktop/SCADA/train'
csv_path_list = []
file_name = '201807.csv'
for i in range(33):
    if i < 9:
        path = os.path.join(csv_input_path, '00%d' % (i + 1), file_name)
    else:
        path = os.path.join(csv_input_path, '0%d' % (i+1), file_name)
    csv_path_list.append(path)
print(csv_path_list)
# df = pd.read_csv(csv_path)
# print(df.head())
num = 1
# print(os.path.join(csv_output_path, '201807_%d.csv' %num))
for paths in csv_path_list:
    df = pd.read_csv(paths)
    df = df.dropna()
    df.to_csv(os.path.join(csv_output_path, '201807_%d.csv' %num))
    num = num + 1

