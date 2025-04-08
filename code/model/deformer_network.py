import torch
from model.embedder import *
import numpy as np
import torch.nn as nn

class ForwardDeformer(nn.Module):
    def __init__(self,
                FLAMEServer,
                d_in,
                dims,
                multires,
                num_exp=50,
                deform_c=False,
                weight_norm=True,
                ghostbone=False,
                ):
        super().__init__()
        self.FLAMEServer = FLAMEServer
        # pose correctives, expression blendshapes and linear blend skinning weights
        d_out = 36 * 3 + num_exp * 3
        if deform_c:
            d_out = d_out + 3
        self.num_exp = num_exp
        self.deform_c = deform_c
        dims = [d_in] + dims + [d_out]
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 2):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            torch.nn.init.constant_(lin.bias, 0.0)
            torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.blendshapes = nn.Linear(dims[self.num_layers - 2], d_out)
        self.skinning_linear = nn.Linear(dims[self.num_layers - 2], dims[self.num_layers - 2])
        self.skinning = nn.Linear(dims[self.num_layers - 2], 6 if ghostbone else 5)
        torch.nn.init.constant_(self.skinning_linear.bias, 0.0)
        torch.nn.init.normal_(self.skinning_linear.weight, 0.0, np.sqrt(2) / np.sqrt(dims[self.num_layers - 2]))
        if weight_norm:
            self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
        # initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)
        torch.nn.init.constant_(self.blendshapes.bias, 0.0)
        torch.nn.init.constant_(self.blendshapes.weight, 0.0)
        torch.nn.init.constant_(self.skinning.bias, 0.0)
        torch.nn.init.constant_(self.skinning.weight, 0.0)

        self.ghostbone = ghostbone
        
        self.mlp1 = nn.Sequential(
            nn.Linear(50,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(36,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            )
        self.flamemlp = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            )

        self.spatialexp = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,50),
            nn.ReLU(),
            )
        self.spatialpose = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,36),
            nn.ReLU(),
            )
        self.concatFusion=ConcatFusion(214,128)
        # self.film=FiLM(50,128,128)
        # self.film=FiLM(214,128)
        # self.fc= nn.Linear(86, 2 * 128)
        
    def query_weights(self, pnts_c, betas,pose_feature,mask=None):
        # emotion_label = self.nncode.expand(pnts_c.shape[0], -1)
        # tmp=self.emotionlin(emotion_label)
        """
        """
        tmp1=self.mlp1(betas)
        tmp2=self.mlp2(pose_feature)
        
        if mask is not None:
            pnts_c = pnts_c[mask]
        if self.embed_fn is not None:
            x = self.embed_fn(pnts_c)
        else:
            x = pnts_c
        #修改
        # x= torch.cat((x, emotion_label), dim=1) 
        for l in range(0, self.num_layers - 2):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            # x = self.softplus(x)
            # x = self.softplus(x)+tmp1+tmp2
            # tmp=torch.cat((betas,pose_feature),1)
            # film = tmp
            # to_be_film = x
            # gamma, beta = torch.split(self.fc(film), 128, 1)
            # output = gamma * to_be_film + beta
            # x = self.fc_out(output)
            
            x = self.concatFusion(self.softplus(x),betas,pose_feature)
        # x = x.unsqueeze(0)
        # feature_vectors=feature_vectors.unsqueeze(0)
        # x,_=self.transformer1(x,feature_vectors)
       
        exp=torch.cat((x, tmp1), dim=1)
        pose=torch.cat((x, tmp2), dim=1)
        new_betas=self.spatialexp(exp)
        new_pose=self.spatialpose(pose)
        """
        new_betas=betas
        new_pose=pose_feature
        """

        blendshapes = self.blendshapes(x)

        posedirs = blendshapes[:, :36 * 3]
        shapedirs = blendshapes[:, 36 * 3: 36 * 3 + self.num_exp * 3]
        # posedirs=self.posedirs(x)
        # shapedirs=self.shapedirs(x)
        # offset=self.offset(x)

        lbs_weights = self.skinning(self.softplus(self.skinning_linear(x)))
        # softmax implementation
        lbs_weights_exp = torch.exp(20 * lbs_weights)
        lbs_weights = lbs_weights_exp / torch.sum(lbs_weights_exp, dim=-1, keepdim=True)
        if self.deform_c:
            pnts_c_flame = pnts_c + blendshapes[:, -3:]
            # pnts_c_flame = pnts_c + offset
        else:
            pnts_c_flame = pnts_c
        return shapedirs.reshape(-1, 3, self.num_exp), posedirs.reshape(-1, 4*9, 3), lbs_weights.reshape(-1, 6 if self.ghostbone else 5), pnts_c_flame,new_betas,new_pose

        
    # def query_weights(self, pnts_c, betas,pose_feature,mask=None):
    #     # emotion_label = self.nncode.expand(pnts_c.shape[0], -1)
    #     # tmp=self.emotionlin(emotion_label)
    #     """
    #     """
    #     tmp1=self.mlp1(betas)
    #     tmp2=self.mlp2(pose_feature)
        
    #     if mask is not None:
    #         pnts_c = pnts_c[mask]
    #     if self.embed_fn is not None:
    #         x = self.embed_fn(pnts_c)
    #     else:
    #         x = pnts_c
    #     x=torch.cat((x, betas,pose_feature), dim=1)
    #     #修改
    #     # x= torch.cat((x, emotion_label), dim=1) 
    #     for l in range(0, self.num_layers - 2):
    #         lin = getattr(self, "lin" + str(l))
    #         x = lin(x)
    #         # x = self.softplus(x)
    #         # x = self.softplus(x)+tmp1+tmp2
    #         # x = self.concatFusion(self.softplus(x),betas,pose_feature)
    #     # x = x.unsqueeze(0)
    #     # feature_vectors=feature_vectors.unsqueeze(0)
    #     # x,_=self.transformer1(x,feature_vectors)
        
    #     exp=torch.cat((x, betas), dim=1)
    #     pose=torch.cat((x, pose_feature), dim=1)
    #     new_betas=tmp1+self.spatialexp(exp)
    #     new_pose=tmp2+self.spatialpose(pose)
    #     """
    #     new_betas=betas
    #     new_pose=pose_feature
    #     """

    #     blendshapes = self.blendshapes(x)

    #     posedirs = blendshapes[:, :36 * 3]
    #     shapedirs = blendshapes[:, 36 * 3: 36 * 3 + self.num_exp * 3]
    #     # posedirs=self.posedirs(x)
    #     # shapedirs=self.shapedirs(x)
    #     # offset=self.offset(x)

    #     lbs_weights = self.skinning(self.softplus(self.skinning_linear(x)))
    #     # softmax implementation
    #     lbs_weights_exp = torch.exp(20 * lbs_weights)
    #     lbs_weights = lbs_weights_exp / torch.sum(lbs_weights_exp, dim=-1, keepdim=True)
    #     if self.deform_c:
    #         pnts_c_flame = pnts_c + blendshapes[:, -3:]
    #         # pnts_c_flame = pnts_c + offset
    #     else:
    #         pnts_c_flame = pnts_c
    #     return shapedirs.reshape(-1, 3, self.num_exp), posedirs.reshape(-1, 4*9, 3), lbs_weights.reshape(-1, 6 if self.ghostbone else 5), pnts_c_flame,new_betas,new_pose

    # def query_weights(self, pnts_c,betas,pose_feature, mask=None):
    #     if mask is not None:
    #         pnts_c = pnts_c[mask]
    #     if self.embed_fn is not None:
    #         x = self.embed_fn(pnts_c)
    #     else:
    #         x = pnts_c

    #     for l in range(0, self.num_layers - 2):
    #         lin = getattr(self, "lin" + str(l))
    #         x = lin(x)
    #         x = self.softplus(x)
    #         x = self.concatFusion(self.softplus(x),betas,pose_feature)
        
    #     exp=torch.cat((x, tmp1), dim=1)
    #     pose=torch.cat((x, tmp2), dim=1)
    #     new_betas=betas+self.spatialexp(exp)
    #     new_pose=pose_feature+self.spatialpose(pose)

    #     blendshapes = self.blendshapes(x)
    #     posedirs = blendshapes[:, :36 * 3]
    #     shapedirs = blendshapes[:, 36 * 3: 36 * 3 + self.num_exp * 3]
    #     lbs_weights = self.skinning(self.softplus(self.skinning_linear(x)))
    #     # softmax implementation
    #     lbs_weights_exp = torch.exp(20 * lbs_weights)
    #     lbs_weights = lbs_weights_exp / torch.sum(lbs_weights_exp, dim=-1, keepdim=True)
    #     if self.deform_c:
    #         pnts_c_flame = pnts_c + blendshapes[:, -3:]
    #     else:
    #         pnts_c_flame = pnts_c
    #     return shapedirs.reshape(-1, 3, self.num_exp), posedirs.reshape(-1, 4*9, 3), lbs_weights.reshape(-1, 6 if self.ghostbone else 5), pnts_c_flame

    def forward_lbs(self, pnts_c, pose_feature, betas, transformations, mask=None):
        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.query_weights(pnts_c, mask)
        pts_p = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=torch.float32)
        return pts_p, pnts_c_flame
    
class ConcatFusion(nn.Module):
    def __init__(self, input_dim=256, output_dim=128):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
 
    def forward(self, x, y,z):
        output = torch.cat((x, y,z), dim=1)
        output = self.fc_out(output)
        return output
    
 
#------------------------------------------#
# FiLM融合方法的定义,只定义一个全连接层
#------------------------------------------#
class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """
    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=False):
        super(FiLM, self).__init__()
        self.dim    = input_dim
        self.fc     = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)
        self.x_film = x_film
 
    def forward(self, x, y):
        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x
 
        gamma, beta = torch.split(self.fc(film), self.dim, 1)
 
        output = gamma * to_be_film + beta
        output = self.fc_out(output)
 
        return output
 