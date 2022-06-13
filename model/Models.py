
import torch
import torch.nn as nn
import torchvision.models as models

from Tools.Model import Model

class SVCNN(Model):
    def __init__(self, nclasses, cnn):
        super(SVCNN, self).__init__('SVCNN')

        self.nclasses = nclasses
        self.pretraining = True
        self.cnn = cnn

        # vgg
        if self.cnn == 'vgg':
            self.net_1 = models.vgg16(pretrained=self.pretraining).features
            self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
            self.net_2._modules['6'] = nn.Linear(4096, nclasses)

    def forward(self, x):
        # x size: N*V,C,H,W
        y = self.net_1(x)
        return self.net_2(y.view(y.shape[0],-1))

class MVCNN(Model):
    def __init__(self, s_model, nclasses, num_views, cnn):
        super(MVCNN, self).__init__('MVCNN')

        # self.s_model = s_model
        self.nclasses = nclasses
        self.num_views = num_views
        self.pretraining = True
        self.cnn = cnn

        self.net_1 = s_model.net_1
        self.net_2 = s_model.net_2


    def forward(self, x):
        # x size: N*V,C,H,W
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))
        return self.net_2(torch.max(y,1)[0].view(y.shape[0],-1))

class A_MVCNN(Model):
    def __init__(self, nclasses, num_views, cnn):
        super(A_MVCNN, self).__init__('A_MVCNN')

        # self.smodel = smodel
        self.nclasses = nclasses
        self.num_views = num_views
        self.pretraining = True
        self.cnn = cnn
        # self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        # self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.net_1 = models.vgg16(pretrained=self.pretraining).features
        self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
        self.net_2._modules['6'] = nn.Linear(4096, nclasses)

        # vgg
        # if self.cnn == 'vgg':
        #     self.net_1 = models.vgg16(pretrained=self.pretraining).features
        #     self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
        #     self.net_2._modules['6'] = nn.Linear(4096, nclasses)

    def forward(self, x):
        # x size: N*V,C,H,W
        y = self.net_1(x)
        y = y.view((int(x.shape[0] / self.num_views), self.num_views, y.shape[-3], y.shape[-2],
                    y.shape[-1]))  # (8,6,512,7,7)
        # print(y.view((int(x.shape[0]/self.num_views),self.num_views,-1)))# = [batch, 512, 7, 7] ==> [batch, 512*7*7]
        return self.net_2(torch.max(y, 1)[0].view(y.shape[0], -1))
