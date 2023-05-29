import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import sys

def str_to_class(name):
    return getattr(sys.modules[__name__], name)


CONTEXT_SIZE= 16

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width)
        out = self.gamma*out + x
        return out


class CRUFT_encoder(nn.Module):
    def __init__(self, length=3, sr=50,n_feature=9, activity_class=4, phone_class=3, ES_feature = 78):
        super(CRUFT_encoder,self).__init__()
        self.sr = sr
        self.length = length
        self.ES_MLP = nn.Sequential(
            nn.Linear(ES_feature,64),
            nn.LeakyReLU() ,
            nn.Dropout(p=0.1), 
            nn.Linear(64,64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1)
            )
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels = 1 , out_channels = 32 , kernel_size= (1,9), stride = (1,2), padding = (0,4)),
            nn.MaxPool2d(kernel_size=(1,2), stride = (1,2)),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels = 32 , out_channels = 64 , kernel_size= (1,3), stride = (1,1), padding = (0,1)),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1), 
            nn.Conv2d(in_channels = 64 , out_channels = 128 , kernel_size= (1,3), stride = (1,1), padding = (0,1)),
            nn.MaxPool2d(kernel_size=(1,2), stride = (1,2)),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels = 128 , out_channels = 128 , kernel_size= (n_feature,1), stride = (1,1), padding = (0,0)),
            nn.LeakyReLU(),
            )
        self.attention = Self_Attn(128,'leaky_relu')
        self.maxpool = nn.MaxPool1d(6)
        self.meanpool = nn.AvgPool1d(6)
        

    def forward(self,x,feature):

        x = x.reshape(x.shape[0],x.shape[1],self.length,self.sr) #[N,C,W] -> [N,C,W//50,50]
        x = x.permute(0,2,1,3) # [N,C,W//10,10]->[N,10,C,W//10]
        x = x.reshape(-1,x.shape[2],self.sr) #  [N,10,C,W//10]->[N*10,C,W//10]
        
        feature = feature.permute(0,2,1)
        feature = feature.reshape(-1,feature.shape[2])
        
        h1 = self.ES_MLP(feature)
        h1 = h1.reshape(-1,self.length,64)
        h2 = self.CNN(x[:,None,:,:,]) #(N,C,W) to (N,1,C,W), output (N,128,1,6)
        h2 = h2.flatten(1,2) # output (N,128,6)
        h2_attn = torch.sum(self.attention(h2),2)
        h2_max_pool = self.maxpool(h2).reshape(-1,128)
        h2_mean_pool = self.meanpool(h2).reshape(-1,128)
        h2 = torch.cat((h2_attn,h2_max_pool,h2_mean_pool),1)
        h2 = h2.reshape(-1,self.length,128*3)
        return h1,h2

class CRUFT_classifier(nn.Module):
    def __init__(self, length=3, sr=50,n_feature=9, activity_class=4, phone_class=3, ES_feature = 78):
        super(CRUFT_classifier,self).__init__()
        self.sr = sr
        self.length = length

        self.lstm = nn.LSTM(128*3, 64, num_layers = 2, dropout =0.1, bidirectional =True, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.linear_prediction_act = nn.Linear(128,activity_class)
        self.linear_prediction_phone = nn.Linear(128,phone_class)
        self.linear_uncertain_act = nn.Linear(128,activity_class)
        self.linear_uncertain_phone = nn.Linear(128,phone_class)

    def forward(self,h1,h2):

        h2,_ = self.lstm(h2)
        h2 = h2[:,:,:64] #select only forward output from LSTM 
        h = torch.cat((h1,h2),2)
        
        prediction_act = self.linear_prediction_act(h)
        prediction_phone = self.linear_prediction_phone(h)
        uncertain_act = self.linear_uncertain_act(h)
        uncertain_phone = self.linear_uncertain_phone(h)
        return F.softmax(prediction_act),F.softmax(prediction_phone),F.softplus(uncertain_act),F.softplus(uncertain_phone)


def get_model(model_name, length, hz):
    return str_to_class(model_name)(length)

if __name__ == '__main__':
    #CNN testing
    # inp = torch.zeros((4, 6, 300))
    # model = CNN_4_16()
    # out = model(inp)

    #CRUFT testing
    inp = torch.rand((2, 9, 50*3))
    features = torch.rand((2, 78, 3))
    model = CRUFT()
    out,_,_,_ = model(inp,features)
    print(out.shape)
