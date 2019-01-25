train_data = '/Users/yangyucheng/Desktop/SCADA/dataset/'
to_be_submit = train_data + 'template_submit_result.csv'
machine_num = 33

attrs = ['ts','wtid']
for i in range(68):
    attrs.append("var{0:03d}".format(i+1))