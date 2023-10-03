from collections import OrderedDict
from models.backbone import common1 
# from models.backbone import ic_conv2d
# from models.backbone.ic_resnet import ic_resnet50
import torch.nn.functional as F
import torchvision
from torch import nn
from models.backbone.resnet_utils import resnet50
import torchvision.models as models
#import timm
from models.gcn_lib import Grapher
from timm.models.layers import DropPath

from timm.models.layers import DropPath

class Conv_FFN(nn.Module):
    def __init__(self, in_features, mlp_ratio=4, out_features=None, act='GeLU', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = in_features * mlp_ratio
        self.conv1 =nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.drop_path(x) + shortcut
        return x#.reshape(B, C, N, 1)

class GNN_Transformer(nn.Module):
    def __init__(
        self,
        dim=256,
        blocks = 1, 
        epsilon = 0.1,
        kernels = [3],
        reduction_ration = 1,
        dilation_rate = [1]
    ):        
        super().__init__()
        self.backbone = nn.ModuleList([])
        for j in range(blocks):
            self.backbone += [
                nn.Sequential(Grapher(dim, kernel_size=kernels[j], dilation=dilation_rate[j], conv='edge', act='gelu', norm='batch',
                                bias=True, stochastic=False, epsilon=epsilon , r=reduction_ration, n=196, drop_path=0.0,
                                relative_pos=True),
                      Conv_FFN(dim)
                     )]
       

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
        embedding_final = x + residual
        return embedding_final


pattern_path ='/l/users/20020129/PersonID/OIMNetPlus/models/pattern_zoo/detection/ic_resnet50_k9.json'
#load_path = 'ckpt/detection/r50_imagenet_retrain/ckpt.pth.tar'

class Backbone(nn.Sequential):
    def __init__(self, resnet):
        super(Backbone, self).__init__()
        layers = []
        self.res2=nn.Sequential(resnet.conv1,
                    resnet.bn1,
                     resnet.relu,
                     resnet.maxpool,
                    resnet.layer1)
          
             

        self.res3=resnet.layer2 # res3
        self.res4=resnet.layer3 # res4
        self.patch_maker=common1.PatchMaker(patchsize=3, stride=1)
        self.rc=nn.Conv2d(1536, 1024, 1, bias=False)
            
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        res2_feat = self.res2(x)
        #print(res2_feat.shape)
        res3_feat=self.res3(res2_feat)
        res4_feat=self.res4(res3_feat)
        features=[res3_feat,res4_feat]
        #print("features",features[0].shape,features[1].shape)
        #B,C,H,W=features[1].shape
        #out_dim=[H,W]

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[1]
        #print("ref_num_patches",ref_num_patches)

        for i in range(0, len(features)-1):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(*perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
            #print("_features",_features.shape)
            #_features = _features.permute(0, 1,4,5,2,3)
            #B,C=_features.shape[:2]
            #_features = _features.contiguous().view(B, C, -1, 3*3)
            #_features = _features.permute(0, 1, 3, 2) 
            #_features = _features.contiguous().view(B, C*3*3, -1)

            #_features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            #print("_features",_features.shape)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
            #print(features[0].shape,features[1].shape)
            preprocessing=common1.Preprocessing([512,1024],output_dim=ref_num_patches)
            x=preprocessing(features)
            x=self.rc(x)
            #print(x.shape)

        return  OrderedDict([["feat_res4", x]])


class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__()  # res5
        self.layer4 = nn.Sequential(resnet.layer4)  # res5
        self.out_channels = [1024, 2048]
        hidden_dim = 256
        in_dim = 1024
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, 1, stride=1, padding=0)
        self.conv2= nn.Conv2d(hidden_dim, in_dim, 1, stride=1, padding=0)
        self.graph_transformer = GNN_Transformer(dim=hidden_dim, blocks=1)
        #---
        #self.conv01 = nn.Conv2d(2048, 256, 1, stride=1, padding=0)
        #self.conv02= nn.Conv2d(256, 2048, 1, stride=1, padding=0)
        #self.graph_transformer01 = GNN_Transformer(dim=256, blocks=1,nd=49)
        #----


    def forward(self, x):
        shortcut = x
        conv1 = self.conv1(x)
        x_gnn_feat=self.graph_transformer(conv1)
        
        conv2 = self.conv2(x_gnn_feat)
       
        layer5_feat = self.layer4(conv2)
        #---------------------------------------
        #conv01 = self.conv01(layer5_feat)
        #x_gnn_feat01=self.graph_transformer01(conv01)
        #conv02 = self.conv02(x_gnn_feat01)
        #----------------------------------
 
    
        x_feat = F.adaptive_max_pool2d(conv2, 1)

        feat = F.adaptive_max_pool2d(layer5_feat, 1)
        #feat = F.adaptive_max_pool2d(conv02, 1)
        
        return OrderedDict([["feat_res4", x_feat], ["feat_res5", feat]])



def build_resnet(name="resnet50", pretrained=True):
    # resnet = resnet50(pretrained=True)
    #resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)
    wideresnet = torchvision.models.wide_resnet50_2(pretrained=True)
    #icresnet=ic_resnet50(pattern_path=pattern_path)
    #icresnet = timm.create_model('inception_resnet_v2.tf_in1k', pretrained=True)

    #state = torch.load(load_path, 'cpu')
    #icresnet.load_state_dict(state, strict=False)
    #state_keys = set(state.keys())
    #model_keys = set(net.state_dict().keys())
    #missing_keys = model_keys - state_keys
    #print(missing_keys)
    

    # freeze layers
    wideresnet.conv1.weight.requires_grad_(False)
    wideresnet.bn1.weight.requires_grad_(False)
    wideresnet.bn1.bias.requires_grad_(False)

    return Backbone(wideresnet), Res5Head(wideresnet)


    