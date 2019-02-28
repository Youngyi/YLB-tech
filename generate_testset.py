import numpy as np
import para
import pandas as pd
import matplotlib.pyplot as plt 
import time

with  open('testset.csv','w') as f:
	f.write('ts,wtid,var001,var002,var003,var004,var005,var006,var007,var008,var009,var010,var011,var012,var013,var014,var015,var016,var017,var018,var019,var020,var021,var022,var023,var024,var025,var026,var027,var028,var029,var030,var031,var032,var033,var034,var035,var036,var037,var038,var039,var040,var041,var042,var043,var044,var045,var046,var047,var048,var049,var050,var051,var052,var053,var054,var055,var056,var057,var058,var059,var060,var061,var062,var063,var064,var065,var066,var067,var068'+'\n')
for i in range(1,33+1):
	start = time.clock()
	data = pd.read_csv(para.train_data+str(i).zfill(3)+'/201807.csv',parse_dates=[0])
	res = pd.read_csv(para.train_data + 'template_submit_result.csv',parse_dates=[0])[['ts','wtid']]

	res = res[res['wtid']==i]
	data = res.merge(data, on=['wtid','ts'],how = 'outer')
	data = data.sort_values(['wtid','ts']).reset_index(drop = True)
	for k in range(data.shape[0]//para.sequence_length):
		d = data[k*para.sequence_length:k*para.sequence_length+para.sequence_length]
		if d.isna().any().any(): #存在缺失
			seq = d.isna().any(axis=1)
			for a in range(len(seq.values)):
				if seq.values[a]:
					break
			for b in range(len(seq.values))[::-1]:
				if seq.values[b]:
					break
			miss_start = a
			miss_end = b
			if miss_start > 50 or miss_start == 0: # 缺失开始位置在段数据一半以前 或 段数据开始就缺失（认为是接续）#TODO:处理比较粗糙
				pass
			else:
				d = data[k*para.sequence_length-50:k*para.sequence_length+para.sequence_length]
				
			with open('testset.csv','a') as f:
				pd.DataFrame(d).to_csv(f,header = False,index=False,sep=',')
	
	elapsed = (time.clock() - start)
	print("Time used:{0}.".format(elapsed))