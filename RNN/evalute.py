import pickle
# from model import EncoderRNN, DecoderRNN
import torch
import pandas as pd
import sys
sys.path.append("..")
import para
import numpy as np
from tqdm import tqdm
epoch = 0

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
    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        decoder_input = decoder_output.detach()  # detach from history as input

    return decoder_output

def fill_pred(df,pred):
    '''
    为缺失值填上预测值
    '''
    mask = df.isna().values.astype('i2')[:,1:]
    df = df.fillna(0)
    return np.concatenate([df['ts'].values.reshape(-1,1),df.values[:,1:] + pred * mask],axis=1)

def main():
    # 1.预处理meta
    with open("meta.pkl",'rb') as file:
        pp = pickle.loads(file.read())
    print('加载预处理meta完成。',flush=True)
    # 2.加载模型
    encoder = torch.load('encoder{0}.pkl'.format(epoch+1))
    decoder = torch.load('decoder{0}.pkl'.format(epoch+1))
    print('加载模型完成。',flush=True)
    
    # 3.加载数据
    data = pd.read_csv('testset.csv',parse_dates=[0])
    sl = para.sequence_length 
    hsl = sl//2 # half sequence length
    
    # 4.预测
    for i in tqdm(range(data.shape[0]//hsl-1)):
        df = data[i*hsl:i*hsl+sl]
        # if check_continue(df):
        inputs = pp.transform(torch.tensor(df.values[:,1:].astype('f4')))
        inputs[np.isnan(inputs)] = 0
        processed_pred = evalute(inputs.view(para.sequence_length,-1,141).float(),encoder,decoder)
        print(processed_pred.shape)
        raw_pred = pp.recover(processed_pred.detach())
        print(raw_pred.shape)
        new_df = fill_pred(df,raw_pred)
        print(new_df.shape)
        data[i*hsl:i*hsl+sl] = new_df
    print('预测完成',flush=True)
    
    # 5.制作submit文件
    res = pd.read_csv(para.train_data + 'template_submit_result.csv',parse_dates=[0])[['ts','wtid']]
    DF = pd.merge(res,data, on=['wtid','ts'],how = 'inner')
    print(res.shape,data.shape,DF.shape)
    # print(DF.isna().any(axis=1))
    DF.to_csv('epoch'+str(epoch)+'.csv',index=False,float_format='%.2f')
    print('结果输出完成',flush=True)
    return data


if __name__ == '__main__':
    data = main()