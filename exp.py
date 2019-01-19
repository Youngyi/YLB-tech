import numpy as np
from utilities import DataLoader, PreProc
import para

def main():
	dl = DataLoader()
	machine_num = 1
	raw_data,_ = dl(para.train_data,machine_num) #加载数据

	data = raw_data[:,2:] #移除前两列
	pp = PreProc(data,dl.t[2:]) #预处理
	inputs = np.concatenate((pp.proc_cont,pp.proc_disc),axis=1) #预处理输出
	print(inputs.shape)
	print(inputs[0])

if __name__ == '__main__':
	main()