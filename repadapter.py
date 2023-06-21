# coding=utf-8
# Copyright 2022 Gen Luo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch import nn
import timm

def forward_vit_block_adapter(self, x):
    # NOTE 这是更改了原来的forward了
    #import ipdb; ipdb.set_trace()
    x = x + self.drop_path1(self.attn(self.adapter_attn(self.norm1(x)))) # NOTE adapter_attn -> attn, 串联; x.shape=[1, 197, 768] -> x.shape=[1, 197, 768]
    x = x + self.drop_path2(self.mlp(self.adapter_mlp(self.norm2(x)))) # NOTE adapter_mlp -> mlp, 串联 -> x.shape=[1, 197, 768]
    return x

def forward_vit_attn_adapter(self, x):
    import ipdb; ipdb.set_trace()
    x = x + self.drop_path1(self.attn(self.adapter_attn(self.norm1(x)))) # NOTE only this part changed
    x = x + self.drop_path2(self.mlp(self.norm2(x))) # keep the same with original model
    return x

def forward_swin_block_adapter(self, x):
    import ipdb; ipdb.set_trace()
    H, W = self.input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    shortcut = x
    x = self.adapter_attn(self.norm1(x)) # NOTE
    x = x.view(B, H, W, C)

    # cyclic shift
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x

    # partition windows
    x_windows = timm.models.swin_transformer.window_partition(shifted_x,
                                                              self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

    # W-MSA/SW-MSA
    attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = timm.models.swin_transformer.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)

    # FFN
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.adapter_mlp(self.norm2(x)))) # NOTE
    return x


def forward_swin_attn_adapter(self, x):
    import ipdb; ipdb.set_trace()
    H, W = self.input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    shortcut = x
    x = self.adapter_attn(self.norm1(x)) # NOTE
    x = x.view(B, H, W, C)

    # cyclic shift
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x

    # partition windows
    x_windows = timm.models.swin_transformer.window_partition(shifted_x,
                                                              self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

    # W-MSA/SW-MSA
    attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = timm.models.swin_transformer.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)

    # FFN
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x


def forward_convnext_attn_adapter(self, x):
    import ipdb; ipdb.set_trace()
    shortcut = x
    x = self.conv_dw(self.adapter_attn(x))
    if self.use_conv_mlp:
        x = self.norm(x)
        x = self.mlp(x)
    else:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
    if self.gamma is not None:
        x = x.mul(self.gamma.reshape(1, -1, 1, 1))

    x = self.drop_path(x) + shortcut
    return x

def forward_convnext_block_adapter(self, x):
    import ipdb; ipdb.set_trace()
    shortcut = x
    x = self.conv_dw(self.adapter_attn(x))
    if self.use_conv_mlp:
        x = self.norm(x)
        x = self.mlp(self.adapter_mlp(x))
    else:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(self.adapter_mlp(x))
        x = x.permute(0, 3, 1, 2)
    if self.gamma is not None:
        x = x.mul(self.gamma.reshape(1, -1, 1, 1))

    x = self.drop_path(x) + shortcut
    return x


class RepAdapter(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
            scale=1 # 0.1
    ):
        super().__init__()
        self.conv_A=nn.Conv1d(in_features,hidden_dim,1,groups=1,bias=True) # Conv1d(768, 8, kernel_size=(1,), stride=(1,))
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True) # Conv1d(8, 768, kernel_size=(1,), stride=(1,), groups=2) NOTE
        self.dropout=nn.Dropout(0.1) # Dropout(p=0.1, inplace=False)
        self.groups=groups # 2
        self.scale=scale # 0.1

        nn.init.xavier_uniform_( self.conv_A.weight) # torch.Size([8, 768, 1]), 认真初始化了
        nn.init.zeros_(self.conv_A.bias) # torch.Size([8]); 全0
        nn.init.zeros_(self.conv_B.weight) # torch.Size([768, 4, 1]); 全0
        nn.init.zeros_(self.conv_B.bias) # torch.Size([768]); 全0

    def forward(self, x):
        x=x.transpose(1,2)
        x=self.conv_B(self.dropout(self.conv_A(x)))*self.scale+x
        x=x.transpose(1,2).contiguous()
        return x


class RepAdapter2D(nn.Module):
    """ Pytorch Implemention of RepAdapter for 2d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
            scale=1
    ):
        super().__init__()
        self.conv_A = nn.Conv2d(in_features, hidden_dim, 1, groups=1, bias=True)
        self.conv_B = nn.Conv2d(hidden_dim, in_features, 1, groups=groups, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.groups = groups
        self.scale = scale

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x):
        x = self.conv_B(self.dropout(self.conv_A(x))) * self.scale + x
        return x

def reparameterize(Wa,Wb,Ba,Bb,scale=1,do_residual=False):
    import ipdb; ipdb.set_trace()
    bias = 0
    id_tensor=0
    if Ba is not None:
        bias=Ba@Wb
    if Bb is not None:
        bias=bias+Bb
    if do_residual:
        id_tensor=torch.eye(Wa.shape[0],Wb.shape[1]).to(Wa.device)
    weight = Wa @ Wb*scale + id_tensor
    return weight.T,bias*scale if isinstance(bias,torch.Tensor) else None

def sparse2dense(weight,groups):
    import ipdb; ipdb.set_trace()
    d,cg=weight.shape
    dg=d//groups
    weight=weight.view(groups,dg,cg).transpose(1,2)
    new_weight=torch.zeros(cg*groups,d,device=weight.device,dtype=weight.dtype)
    for i in range(groups):
        new_weight[i*cg:(i+1)*cg,i*dg:(i+1)*dg]=weight[i]
    return new_weight.T


def set_RepAdapter(model, method, dim=8, s=1, args=None,set_forward=True):
    #import ipdb; ipdb.set_trace(); dim=rank in LoRA; s=scale for scale-based residual adding
    if method == 'repblock':
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                #import ipdb; ipdb.set_trace()
                _.adapter_attn = RepAdapter(hidden_dim=dim,scale=s)
                _.adapter_mlp = RepAdapter(hidden_dim=dim,scale=s)
                _.s = s # scale=0.1
                bound_method = forward_vit_block_adapter.__get__(_, _.__class__) # object实例, class类 NOTE
                if set_forward: # True
                    setattr(_, 'forward', bound_method) # <class 'timm.models.vision_transformer.Block'>的对象
            elif type(_) == timm.models.swin_transformer.SwinTransformerBlock:
                _.adapter_attn = RepAdapter(in_features=_.dim,hidden_dim=dim,scale=s)
                _.adapter_mlp = RepAdapter(in_features=_.dim,hidden_dim=dim,scale=s)
                _.s = s
                bound_method = forward_swin_block_adapter.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif type(_)== timm.models.convnext.ConvNeXtBlock:
                _.adapter_attn = RepAdapter2D(in_features=_.norm.weight.shape[0], hidden_dim=dim, scale=s)
                _.adapter_mlp = RepAdapter2D(in_features=_.norm.weight.shape[0], hidden_dim=dim, scale=s)
                _.s = s
                bound_method = forward_convnext_block_adapter.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_RepAdapter(_, method, dim, s,args=args,set_forward=set_forward)

    else: # for 'repattn'
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                _.adapter_attn = RepAdapter(hidden_dim=dim,scale=s)
                _.s = s
                bound_method = forward_vit_attn_adapter.__get__(_, _.__class__)
                if set_forward:
                    setattr(_, 'forward', bound_method)
            elif type(_) == timm.models.swin_transformer.SwinTransformerBlock:
                _.adapter_attn =  RepAdapter(in_features=_.dim,hidden_dim=dim,scale=s)
                _.s = s
                bound_method = forward_swin_attn_adapter.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif type(_)== timm.models.convnext.ConvNeXtBlock:
                _.adapter_attn =  RepAdapter2D(in_features=_.norm.weight.shape[0], hidden_dim=dim, scale=s)
                _.s = s
                bound_method = forward_convnext_attn_adapter.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_RepAdapter(_, method, dim, s, args=args, set_forward=set_forward)

def set_RepWeight(model, method, dim=8, s=1, args=None):
    import ipdb; ipdb.set_trace()
    if method == 'repblock':
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block or type(_) == timm.models.swin_transformer.SwinTransformerBlock:
                if _.adapter_attn.groups>1:
                    weight_B=sparse2dense(_.adapter_attn.conv_B.weight.squeeze(),_.adapter_attn.groups)
                else:
                    weight_B=_.adapter_attn.conv_B.weight.squeeze()
                attn_weight,attn_bias=reparameterize(_.adapter_attn.conv_A.weight.squeeze().T,weight_B.T,
                                        _.adapter_attn.conv_A.bias,_.adapter_attn.conv_B.bias,_.s,do_residual=True)
                qkv_weight,qkv_bias=reparameterize(attn_weight.T,_.attn.qkv.weight.T,
                                                attn_bias, _.attn.qkv.bias)

                with torch.no_grad():
                    _.attn.qkv.weight.copy_(qkv_weight)
                    _.attn.qkv.bias.copy_(qkv_bias)

                if _.adapter_mlp.groups>1:
                    weight_B=sparse2dense(_.adapter_mlp.conv_B.weight.squeeze(),_.adapter_mlp.groups)
                else:
                    weight_B=_.adapter_mlp.weight_B.squeeze()

                mlp_weight,mlp_bias=reparameterize(_.adapter_mlp.conv_A.weight.squeeze().T,weight_B.T,
                                        _.adapter_mlp.conv_A.bias,_.adapter_mlp.conv_B.bias,_.s,do_residual=True)
                fc_weight,fc_bias=reparameterize(mlp_weight.T,_.mlp.fc1.weight.T,
                                              mlp_bias, _.mlp.fc1.bias)
                with torch.no_grad():
                    _.mlp.fc1.weight.copy_(fc_weight)
                    _.mlp.fc1.bias.copy_(fc_bias)
            # elif type(_) == timm.models.swin_transformer.SwinTransformerBlock:
            #     _.adapter_attn = RepAdapter(in_features=_.dim,hidden_dim=dim,scale=s)
            #     _.adapter_mlp = RepAdapter(in_features=_.dim,hidden_dim=dim,scale=s)
            #     _.s = s
            #     bound_method = forward_swin_block_adapter.__get__(_, _.__class__)
            #     setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_RepWeight(_, method, dim, s, args=args)

    else:
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block or type(_) == timm.models.swin_transformer.SwinTransformerBlock:
                if _.adapter_attn.groups>1:
                    weight_B=sparse2dense(_.adapter_attn.conv_B.weight.squeeze(),_.adapter_attn.groups)
                else:
                    weight_B=_.adapter_attn.conv_B.weight.squeeze()
                attn_weight,attn_bias=reparameterize(_.adapter_attn.conv_A.weight.squeeze().T,weight_B.T,
                                        _.adapter_attn.conv_A.bias,_.adapter_attn.conv_B.bias,_.s,do_residual=True)
                qkv_weight,qkv_bias=reparameterize(attn_weight.T,_.attn.qkv.weight.T,
                                                attn_bias, _.attn.qkv.bias)
                with torch.no_grad():
                    _.attn.qkv.weight.copy_(qkv_weight)
                    _.attn.qkv.bias.copy_(qkv_bias)
            # elif type(_) == timm.models.swin_transformer.SwinTransformerBlock:
            #     _.adapter_attn = RepAdapter(in_features=_.dim,hidden_dim=dim,scale=s)
            #     _.s = s
            #     bound_method = forward_swin_attn_adapter.__get__(_, _.__class__)
            #     setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_RepWeight(_, method, dim, s,args=args)
