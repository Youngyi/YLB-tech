import pickle
# from model import EncoderRNN, DecoderRNN
import torch
import pandas as pd
import sys
sys.path.append("..")
import para
import numpy as np
from tqdm import tqdm
epoch = 8

def check_continue(df):
    '''
    检查是否连续
    '''
    d = df.values # to numpy
    diff = d[1:,0] - d[:-1,0]
    diff = np.array([di.total_seconds() for di in diff])
    return all(diff<14)


def evalute(input_tensor, encoder, decoder):
    '''
    预测函数
    '''
    encoder_hidden = encoder.initHidden(1)
    target_length = input_tensor.size(0)
    #input_tensor: L x B x F
    #encoder_hidden: 1 x B x H
    #encoder_output: L x B x H
    encoder_output, encoder_hidden = encoder(
            input_tensor, encoder_hidden)
    decoder_input = torch.zeros_like(input_tensor[0]) # B x F
    decoder_hidden = encoder_hidden.view(-1, para.hidden_size) # B x H
    output = []
    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        decoder_input = decoder_output.detach()  # detach from history as input
        output.append(decoder_output)
    return torch.stack(output, dim=1)


def fill_pred(df,pred):
    '''
    为缺失值填上预测值
    '''
    mask = df.isna().values.astype('i2')[:,1:]
    df = df.fillna(0)
    return np.concatenate([df['ts'].values.reshape(-1,1),df.values[:,1:] + pred * mask],axis=1)
def get_data(i):
    """
    get_ori_data
    """
    data = pd.read_csv('/Users/yangyucheng/Desktop/SCADA/dataset/' + str(i).zfill(3) + '/201807.csv', parse_dates=[0])
    res = pd.read_csv('/Users/yangyucheng/Desktop/SCADA/template_submit_result.csv', parse_dates=[0])[['ts', 'wtid']]
    print(data.shape)
    res = res[res['wtid'] == i]
    # res['flag'] = 1
    data = res.merge(data, on=['wtid', 'ts'], how='outer')
    data = data.sort_values(['wtid', 'ts']).reset_index(drop=True)

    print(data.shape)
    return data

def main():
    # 1.预处理meta
    with open("meta.pkl",'rb') as file:
        pp = pickle.loads(file.read())
    print('加载预处理meta完成。',flush=True)
    # 2.加载模型
    encoder = torch.load('encoder10.pkl')
    decoder = torch.load('decoder10.pkl')
    print('加载模型完成。',flush=True)
    
    # 3.加载数据

    sl = para.sequence_length 
    psl = sl//10 # half sequence length
    forecast_data = pd.DataFrame()
    for i in tqdm(range(1, 34)):
        data = get_data(i)
        for i in tqdm(range(data.shape[0]//psl)):
            if 100+i*psl > data.shape[0]:
                part_data = data[data.shape[0]-10:data.shape[0]]
                df = data[data.shape[0]-100:data.shape[0]]
            else:
                part_data = data[90+i*psl:100+i*psl]
                df = data[i * psl:100 + i * psl]
            if part_data.isna().any().any():
                inputs = pp.transform(torch.tensor(df.values[:, 1:].astype('f4')))
                inputs[np.isnan(inputs)] = 0
                processed_pred = evalute(inputs.view(para.sequence_length, -1, 141).float(), encoder, decoder)
                processed_pred = processed_pred[0]
                raw_pred = pp.recover(processed_pred.detach())
                raw_pred = raw_pred[90:100]
                new_df = fill_pred(part_data,raw_pred)
                if 100 + i * psl > data.shape[0]:
                    data[data.shape[0] - 10:data.shape[0]] = new_dfforecast_data
                else:
                    data[90 + i * psl:100 + i * psl] = new_df
        forecast_data = pd.concat([forecast_data, data], axis=0)
    print('预测完成',flush=True)
    
    # 5.制作submit文件
    res = pd.read_csv(para.train_data + 'template_submit_result.csv',parse_dates=[0])[['ts','wtid']]
    DF = pd.merge(res,forecast_data, on=['wtid','ts'],how = 'inner')
    print(res.shape,data.shape,DF.shape)
    # print(DF.isna().any(axis=1))
    DF.to_csv('epoch'+str(epoch)+'.csv',index=False,float_format='%.2f')
    print('结果输出完成',flush=True)
    return data


if __name__ == '__main__':
    data = main()