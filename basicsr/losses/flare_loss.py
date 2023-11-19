import cv2
import numpy as np
import torch
from torch import abs_, nn
from torch import optim
from PIL import Image
from typing import Mapping,Sequence,Tuple,Union
from torchvision.models import vgg19
import torchvision.models.vgg as vgg
from basicsr.utils.registry import LOSS_REGISTRY

class L_Abs_sideout(nn.Module):
    def __init__(self):
        super(L_Abs_sideout, self).__init__()
        self.resolution_weight=[1.,1.,1.,1.]

    def forward(self,x,flare_gt):
        #[256,256],[128,128],[64,64],[32,32]
        Abs_loss=0
        for i in range(4):
            flare_loss=torch.abs(x[i]-flare_gt[i])
            Abs_loss+=torch.mean(flare_loss)*self.resolution_weight[i]
        return Abs_loss
    

class L_Abs(nn.Module):
    def __init__(self):
        super(L_Abs, self).__init__()

    def forward(self,x,flare_gt,base_gt,mask_gt,merge_gt):
        base_predicted=base_gt*mask_gt+(1-mask_gt)*x
        flare_predicted=merge_gt-(1-mask_gt)*x
        base_loss=torch.abs(base_predicted-base_gt)
        flare_loss=torch.abs(flare_predicted-flare_gt)
        Abs_loss=torch.mean(base_loss+flare_loss)
        return Abs_loss

@LOSS_REGISTRY.register()
class L_Abs_pure(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(L_Abs_pure, self).__init__()
        self.loss_weight=loss_weight

    def forward(self,x,flare_gt):
        flare_loss=torch.abs(x-flare_gt)
        Abs_loss=torch.mean(flare_loss)
        return self.loss_weight*Abs_loss

@LOSS_REGISTRY.register()
class L_Abs_weighted(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(L_Abs_weighted, self).__init__()
        self.loss_weight=loss_weight

    def forward(self,x,flare_gt,weight):
        flare_loss=torch.abs(x-flare_gt)
        Abs_loss=torch.mean(flare_loss*weight)
        '''
        mask_area=torch.mean(torch.abs(weight))
        if mask_area>0:
            return self.loss_weight*Abs_loss/mask_area
        else:
        '''
        return self.loss_weight*Abs_loss

@LOSS_REGISTRY.register()
class L_percepture(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(L_percepture, self).__init__()
        self.loss_weight=loss_weight
        vgg = vgg19(pretrained=True)
        model = nn.Sequential(*list(vgg.features)[:31])
        model=model.cuda()
        model = model.eval()
        # Freeze VGG19 #
        for param in model.parameters():
            param.requires_grad = False

        self.vgg = model
        self.mae_loss = nn.L1Loss()
        self.selected_feature_index=[2,7,12,21,30]
        self.layer_weight=[1/2.6,1/4.8,1/3.7,1/5.6,10/1.5]
    
    def extract_feature(self,x):
        selected_features = []
        for i,model in enumerate(self.vgg):
            x = model(x)
            if i in self.selected_feature_index:
                selected_features.append(x.clone())
        return selected_features

    def forward(self, source, target):
        source_feature = self.extract_feature(source)
        target_feature = self.extract_feature(target)
        len_feature=len(source_feature)
        perceptual_loss=0
        for i in range(len_feature):
            perceptual_loss+=self.mae_loss(source_feature[i],target_feature[i])*self.layer_weight[i]
        return self.loss_weight*perceptual_loss

@LOSS_REGISTRY.register()
class CorssEntropy(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(CorssEntropy, self).__init__()
        self.loss_weight=loss_weight
        self.loss = nn.BCELoss()

    def forward(self, source, target):
        cross_entropy_loss = self.loss(source, target)
        return self.loss_weight*cross_entropy_loss

@LOSS_REGISTRY.register()
class WeightedBCE(nn.Module):
    def __init__(self, loss_weight=1.0,class_weight=[1.0,1.0]):
        super(WeightedBCE, self).__init__()
        self.loss_weight=loss_weight
        self.class_weight = class_weight

    def forward(self, input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - self.class_weight[1] * target * torch.log(input) - (1 - target) * self.class_weight[0] * torch.log(1 - input)
        return torch.mean(bce)

#matrix
@LOSS_REGISTRY.register()
class Orth_dist(nn.Module):
    def __init__(self, stride, loss_weight=1.0):
        super(Orth_dist, self).__init__()
        self.stride=stride
        self.loss_weight=loss_weight

    def forward(self, mat):
        mat = mat.reshape( (mat.shape[0], -1) )
        if mat.shape[0] < mat.shape[1]:
            mat = mat.permute(1,0)
        return self.loss_weight * torch.norm( torch.t(mat)@mat - torch.eye(mat.shape[1]).cuda())

#conv
@LOSS_REGISTRY.register()
class Orth_conv_dist(nn.Module):
    def __init__(self, stride, loss_weight=1.0):
        super(Orth_conv_dist, self).__init__()
        self.stride = stride
        self.loss_weight=loss_weight

    def forward(self, kernel):
       [o_c, i_c, w, h] = kernel.shape
        assert (w == h),"Do not support rectangular kernel"
        #half = np.floor(w/2)
        assert self.stride < w,"Please use matrix orthgonality instead"
        new_s = self.stride*(w-1) + w#np.int(2*(half+np.floor(half/stride))+1)
        temp = torch.eye(new_s*new_s*i_c).reshape((new_s*new_s*i_c, i_c, new_s,new_s)).cuda()
        out = (F.conv2d(temp, kernel, stride = self.stride)).reshape((new_s*new_s*i_c, -1))
        Vmat = out[np.floor(new_s**2/2).astype(int)::new_s**2, :]
        temp= np.zeros((i_c, i_c*new_s**2))
        for i in range(temp.shape[0]):temp[i,np.floor(new_s**2/2).astype(int)+new_s**2*i]=1
        return self.loss_weight * torch.norm( Vmat@torch.t(out) - torch.from_numpy(temp).float().cuda() )

#deconv
@LOSS_REGISTRY.register()
class Orth_deconv_dist(nn.Module):
    def __init__(self, stride, padding, loss_weight=1.0):
        super(Orth_deconv_dist, self).__init__()
        self.stride  = stride
        self.padding = padding
        self.loss_weight=loss_weight



    def forward(self, kernel):
        [o_c, i_c, w, h] = kernel.shape
        output = torch.conv2d(kernel, kernel, stride=self.stride, padding=self.padding)
        target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
        ct = int(np.floor(output.shape[-1]/2))
        target[:,:,ct,ct] = torch.eye(o_c).cuda()
        return self.loss_weight * torch.norm( output - target )