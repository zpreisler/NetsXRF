import torch
from torch import nn,optim

class Skip(nn.Module):

    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.conv = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=in_channels, padding=2, kernel_size=5,bias=False),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Conv1d(in_channels=in_channels, out_channels=in_channels, padding=2, kernel_size=5,bias=False)
                )

        self.pooling = nn.Sequential(
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, padding=0, kernel_size=1),
                #nn.MaxPool1d(kernel_size=2, stride=2)
                nn.Conv1d(out_channels,out_channels,1,2)
                )

    def forward(self,x):

        y = self.conv(x)
        x = x + y
        x = self.pooling(x)

        return x

class ResNet0(nn.Module):

    def __init__(self,channels = 16, kernel_size = 5):
        super().__init__()

        self.l0 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=channels, padding= int(kernel_size / 2), kernel_size=kernel_size)
        )

        self.reduce = nn.MaxPool1d(kernel_size=2, stride=2)

        self.skip_0 = Skip(channels,channels)
        self.skip_1 = Skip(channels,channels)
        self.skip_2 = Skip(channels,channels)
        self.skip_3 = Skip(channels,1)

        self.fc = nn.Sequential(
            nn.Linear(64,5),
        )

    def forward(self,x):
        
        #print(x.shape)
        x = self.l0(x)

        x = self.skip_0(x)
        #x = self.skip_1(x) + self.reduce(x)
        #x = self.skip_2(x) + self.reduce(x)
        x = self.skip_1(x) 
        x = self.skip_2(x)
        x = self.skip_3(x)

        #print('0:',x.shape)
        z = x.flatten(1)
        #print('1:',z.shape)
        z = self.fc(z)
        #print('2:',z.shape)

        return z

class ResNet1(nn.Module):

    def __init__(self,channels = 16, kernel_size = 5):
        super().__init__()

        self.l0 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=channels, padding= int(kernel_size / 2), kernel_size=kernel_size)
        )

        #self.reduce = nn.MaxPool1d(kernel_size=2, stride=2)
        self.reduce0 = nn.Conv1d(channels,channels,1,2)
        self.reduce1 = nn.Conv1d(channels,channels,1,2)
        self.reduce2 = nn.Conv1d(channels,channels,1,2)

        self.skip_0 = Skip(channels,channels)
        self.skip_1 = Skip(channels,channels)
        self.skip_2 = Skip(channels,channels)
        self.skip_3 = Skip(channels,1)

        self.fc = nn.Sequential(
            nn.Linear(64,5),
            nn.ReLU()
        )

    def forward(self,x):
        positive_x = x
        positive_x[x<=0] = 1
        #mean = torch.mean(x,2)
        #_mean = mean
        #mean = mean.unsqueeze(1)
        #std = torch.std(x,2)
        #_std = std
        #std = std.unsqueeze(1)
        #mean = mean.expand(-1,1,1024)
        #std = std.expand(-1,1,1024)
        #print(std.shape,mean.shape,x.shape)
        
        #x = torch.cat([x,torch.log(positive_x),mean,std],axis=1)
        x = torch.cat([x,torch.log(positive_x)],axis=1)
        #print(x.shape)
        x = self.l0(x)

        x = self.skip_0(x)
        x = self.skip_1(x) + self.reduce1(x)
        x = self.skip_2(x) + self.reduce2(x)
        #x = self.skip_1(x)
        #x = self.skip_2(x)
        x = self.skip_3(x)

        #print('0:',x.shape)
        z = x.flatten(1)
        #print('1:',z.shape)

        #print(_mean.shape)
        #w = torch.cat([z,_mean,_std],axis=1)
        #print('2:',w.shape)
        z = self.fc(z)

        return z

class CNN1(nn.Module):

    def __init__(self,channels = 16, kernel_size = 5):
        super().__init__()

        self.conv_0 = nn.Conv1d(in_channels=2, out_channels=channels,
                kernel_size=kernel_size, padding=int(kernel_size / 2),dtype=torch.float)

        self.conv_1 = nn.Conv1d(in_channels=self.conv_0.out_channels, out_channels=self.conv_0.out_channels * 2,
                kernel_size=kernel_size, padding=int(kernel_size / 2),dtype=torch.float)

        self.conv_2 = nn.Conv1d(in_channels=self.conv_1.out_channels, out_channels=self.conv_1.out_channels * 2,
                kernel_size=kernel_size, padding=int(kernel_size / 2),dtype=torch.float)

        self.conv_3 = nn.Conv1d(in_channels=self.conv_2.out_channels, out_channels=self.conv_2.out_channels * 2,
                kernel_size=kernel_size, padding=int(kernel_size / 2),dtype=torch.float)

        self.c0 = nn.Sequential(
                self.conv_0,
                nn.BatchNorm1d(self.conv_0.out_channels,dtype=torch.float),
                nn.ReLU(),
                nn.Conv1d(self.conv_0.out_channels,self.conv_0.out_channels,1,2,dtype=torch.float)
        )
        self.c1 = nn.Sequential(
                self.conv_1,
                nn.BatchNorm1d(self.conv_1.out_channels,dtype=torch.float),
                nn.ReLU(),
                nn.Conv1d(self.conv_1.out_channels,self.conv_1.out_channels,1,2,dtype=torch.float)
        )
        self.c2 = nn.Sequential(
                self.conv_2,
                nn.BatchNorm1d(self.conv_2.out_channels,dtype=torch.float),
                nn.ReLU(),
                nn.Conv1d(self.conv_2.out_channels,self.conv_2.out_channels,1,2,dtype=torch.float)
        )
        self.c3 = nn.Sequential(
                self.conv_3,
                nn.BatchNorm1d(self.conv_3.out_channels,dtype=torch.float),
                nn.ReLU(),
                nn.Conv1d(self.conv_3.out_channels,self.conv_3.out_channels,1,4,dtype=torch.float)
        )
        self.fc = nn.Sequential(
            nn.Linear(4096,5,dtype=torch.float),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096,2048,dtype=torch.float),
            nn.Linear(2048,5,dtype=torch.float),
            nn.ReLU()
        )

    def forward(self,x):

        positive_x = x
        positive_x[x<=0] = 1

        x = torch.cat([x,torch.log(positive_x)],axis=1)
        #print("Xshape",x.shape)

        x = self.c0(x)
        #print(x.shape)
        x = self.c1(x)
        #print(x.shape)
        x = self.c2(x)
        #print(x.shape)
        x = self.c3(x)
        #print(x.shape)

        z = x.flatten(1)

        z1 = self.fc(z)
        z2 = self.fc2(z)

        z = z1 + z2

        return z
