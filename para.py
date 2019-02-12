# train_data = '/Users/jiayuzhai/Downloads/dataset/'
train_data = '/Users/yangyucheng/Desktop/SCADA/dataset/'
to_be_submit = train_data + 'template_submit_result.csv'
machine_num = 33

attrs = ['ts','wtid']
for i in range(68):
    attrs.append("var{0:03d}".format(i+1))

hidden_size = 150
batch_size = 8
sequence_length = 100
learning_rate = 0.0001

num_epoch = 2

