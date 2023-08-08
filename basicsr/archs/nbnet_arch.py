from torch import nn
from torch.nn import init
import torch
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY



def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, downsample, relu_slope):
        super(UNetConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
            # nn.GELU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))
        # print(f'insize {in_size} out_size = {out_size}')
        # self.block = nn.Sequential(AOTBlock(in_size, opt.rates) )   
        self.downsample = downsample
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)
        # self.shortcut = nn.Sequential(*[AOTBlock(in_size,out_size , opt.rates) for _ in range(opt.num_res)])
    def forward(self, x):

        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc

        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope, subnet_repeat_num, subspace_dim=16 ):
        super(UNetUpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)
        self.num_subspace = subspace_dim
        # print(self.num_subspace, subnet_repeat_num)
        
        self.subnet = Subspace(in_size, self.num_subspace)
        self.skip_m = skip_blocks(out_size, out_size, subnet_repeat_num)

    def forward(self, x, bridge):
        up = self.up(x)
        bridge = self.skip_m(bridge)
        out = torch.cat([up, bridge], 1)

        if self.subnet:
            b_, c_, h_, w_ = bridge.shape
            sub = self.subnet(out)
            V_t = sub.reshape(b_, self.num_subspace, h_*w_)
            V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdim=True))
            V = V_t.transpose(1, 2)
            mat = torch.matmul(V_t, V)
            mat_inv = torch.inverse(mat)
            project_mat = torch.matmul(mat_inv, V_t)
            bridge_ = bridge.reshape(b_, c_, h_*w_)
            project_feature = torch.matmul(project_mat, bridge_.transpose(1, 2))
            bridge = torch.matmul(V, project_feature).transpose(1, 2).reshape(b_, c_, h_, w_)
            out = torch.cat([up, bridge], 1)
        
        out = self.conv_block(out)
        return out

class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = nn.ModuleList()

        self.blocks.append(UNetConvBlock(in_size, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x + sc


class skip_blocks(nn.Module):

    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = nn.ModuleList()

        self.re_num = repeat_num
        mid_c = 128
        self.blocks.append(UNetConvBlock(in_size, mid_c, False, 0.2))

        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2))

        self.blocks.append(UNetConvBlock(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)

        for m in self.blocks:
            x = m(x)
        return x + sc

class AOTBlock(nn.Module):
    def __init__(self, dim_in, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        
        
        for i, rate in enumerate(rates):
            
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    nn.ReflectionPad2d(int(rate)),
                    nn.Conv2d(dim_in, dim_in//len(self.rates), 3, padding=0, dilation=int(rate)),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim_in,dim_in ,3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim_in, dim_in,3, padding=0, dilation=1))

    def forward(self, x):
        
        out = []
        for i  in range(len(self.rates)):
            block_name = f'block{str(i).zfill(2)}'
            
            
            block = self.__getattr__(block_name)
            # print(f"Block {i} output shape: {x.shape}")
            input_channels = x.shape[1]
            input_channels_block = block[1].in_channels
            output_channels = block[1].out_channels
            # print(f"Block {i} - Input Channels: {input_channels}, in: {input_channels_block} Output Channels: {output_channels}")
            
            block_output = block(x)
            out.append(block_output)
            # print(f"Block {i} output shape: {block_output.shape}")

        # out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        # print(f"Concatenated output shape: {out[0].shape, out[1].shape}")
        out = torch.cat(out, 1)
        # print(f"Concatenated output shape: {out.shape}")
        
        out = self.fuse(out)
        # print(f"Fused output shape: {out.shape}")
        
        mask = my_layer_norm(self.gate(x))
        # print(f"Mask shape: {mask.shape}")
        
        
        mask = torch.sigmoid(mask)
        # print(f"Sigmoid Mask shape: {mask.shape}")
        
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

    
@ARCH_REGISTRY.register()
class NBNet(nn.Module):
    def __init__(self,in_chn,output_ch , wf=32, depth=5, relu_slope=0.2, subspace_dim = 16):
        super(NBNet,self).__init__()
        
        self.depth = depth
        self.down_path = nn.ModuleList()
        # self.down_path = []

        prev_channels = self.get_input_chn(in_chn)
        # print(f'prev_channels{prev_channels}')
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wf, downsample, relu_slope))
            prev_channels = (2**i) * wf
            # print(f'prev_channels {prev_channels}')

        # self.ema = EMAU(prev_channels, prev_channels//8)
        # self.up_path = []
        # if opt.aot:
        #     self.middle = nn.Sequential(*[AOTBlock(prev_channels, opt.rates) for _ in range(opt.num_aot)])
        
        self.up_path = nn.ModuleList()

        subnet_repeat_num = 1
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope, subnet_repeat_num, subspace_dim))
            prev_channels = (2**i)*wf
            subnet_repeat_num += 1

        self.last = conv3x3(prev_channels, output_ch, bias=True)
        # self.refine = EBlock(out_channel=3, num_res=4)
        #self._initialize()
        # self.apply(self._init_weights)


    def _init_weights(self,m):
        init_type='normal'
        # init_type='orthogonal'
        gain=0.02
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)


    def forward(self, x, jetmap):
    # def forward(self, x):
        x1 = torch.cat((x, jetmap), dim=1)
        blocks = []
        for i, down in enumerate(self.down_path):
            # print(x1.shape)
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                blocks.append(x1_up)
            else:
                x1 = down(x1)
        # # print(x1.shape)
        # x1 = self.ema(x1)
        # if opt.aot:
        #     x1 = self.middle(x1)
        for i, up in enumerate(self.up_path):
            # # print(x1.shape, blocks[-i-1].shape)
            x1 = up(x1, blocks[-i-1])

        pred = self.last(x1)
        
        # pred_flare = x - self.refine(pred_flare)
        
        
        return pred
    
    def get_input_chn(self, in_chn):
        return in_chn

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class ResBlock(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, active=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, active=False)
        )
        
    def forward(self, x):
        return self.main(x) + x
    
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, active=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if active:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)