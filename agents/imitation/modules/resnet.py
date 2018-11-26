from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from modules.convolution_lstm import ConvLSTMSeq

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    # def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 enable_progressive=False, progress_start=0, progress_chas=[], progress_inc=2, progress_kernel=3,
                 enable_lstm=False, flo_len=0, flo_mode="", flo_indices=[], batch_size=1, lstm_cha=256, lstm_hei=64, lstm_wid=64, dtype=torch.FloatTensor,
                 n_blocks=6, padding_type='reflect'):

        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.enable_progressive = enable_progressive
        self.progress_start = progress_start
        self.progress_chas = progress_chas
        self.progress_inc = progress_inc
        self.progress_kernel = progress_kernel
        self.progress_num = len(self.progress_chas) - 1

        self.enable_lstm = enable_lstm
        self.flo_len = flo_len
        self.flo_mode = flo_mode
        self.flo_indices = flo_indices
        self.batch_size = batch_size
        self.lstm_cha = lstm_cha
        self.lstm_hei = lstm_hei
        self.lstm_wid = lstm_wid
        self.dtype = dtype

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        # self.gpu_ids = gpu_ids


        if not self.enable_progressive:
            # before
            model = [nn.ReflectionPad2d(3),
                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                     norm_layer(ngf),
                     nn.ReLU(True)]

            n_downsampling = 2
            for i in range(n_downsampling):
                mult = 2**i
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]

            self.model_before = nn.Sequential(*model)

            # middle    # 256x64x64
            model = []
            mult = 2**n_downsampling
            for i in range(n_blocks):
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]

            self.model_middle = nn.Sequential(*model)

            # after
            model = []
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            model += [nn.ReflectionPad2d(3)]
            model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model += [nn.Tanh()]

            self.model_after = nn.Sequential(*model)
        else:
            # before & after
            self.from_rgb_blocks = nn.ModuleList()
            self.to_rgb_blocks = nn.ModuleList()
            self.from_rgb_convs = nn.ModuleList()
            self.to_rgb_convs = nn.ModuleList()
            for progress_id in range(self.progress_num):
                if progress_id < self.progress_start:
                    self.from_rgb_blocks.append(nn.Sequential())
                    self.to_rgb_blocks.append(nn.Sequential())
                else:
                    self.from_rgb_blocks.append(nn.Sequential(nn.Conv2d(self.input_nc, self.progress_chas[progress_id+1], kernel_size=1)))
                    self.to_rgb_blocks.append(nn.Sequential(nn.Conv2d(self.progress_chas[progress_id+1], self.output_nc, kernel_size=1)))
                self.from_rgb_convs.append(nn.Sequential(nn.ReflectionPad2d(self.progress_kernel//2),
                                                         nn.Conv2d(self.progress_chas[progress_id+1], self.progress_chas[progress_id], kernel_size=self.progress_kernel),
                                                         norm_layer(self.progress_chas[progress_id]),
                                                         nn.ReLU(True),
                                                         nn.ReflectionPad2d(self.progress_kernel//2),
                                                         nn.Conv2d(self.progress_chas[progress_id], self.progress_chas[progress_id], kernel_size=self.progress_kernel),
                                                         norm_layer(self.progress_chas[progress_id]),
                                                         nn.ReLU(True)))
                self.to_rgb_convs.append(nn.Sequential(nn.ReflectionPad2d(self.progress_kernel//2),
                                                       nn.Conv2d(self.progress_chas[progress_id], self.progress_chas[progress_id+1], kernel_size=self.progress_kernel),
                                                       norm_layer(self.progress_chas[progress_id+1]),
                                                       nn.ReLU(True),
                                                       nn.ReflectionPad2d(self.progress_kernel//2),
                                                       nn.Conv2d(self.progress_chas[progress_id+1], self.progress_chas[progress_id+1], kernel_size=self.progress_kernel),
                                                       norm_layer(self.progress_chas[progress_id+1]),
                                                       nn.ReLU(True)))

            # middle
            model = []
            for i in range(n_blocks):
                model += [ResnetBlock(self.progress_chas[0], padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]

            self.model_middle = nn.Sequential(*model)

        # if self.enable_lstm:
        #     # self.lstm = ConvLSTMCell(input_channels=self.lstm_cha, hidden_channels=self.lstm_cha, kernel_size=5)
        #     # self._reset_lstm_hidden_vb_episode()
        #     # self._reset_lstm_hidden_vb_rollout()
        #     # self.lstm = ConvLSTMSeq(flo_mode=self.flo_mode, input_channels=self.lstm_cha, hidden_channels=[self.lstm_cha], kernel_size=5, dtype=self.dtype)

    # NOTE: to be called at the beginning of each new episode, clear up the hidden state
    def _reset_lstm_hidden_vb_episode(self, training=True): # seq_len, batch_size, hidden_dim
        not_training = not training
        self.lstm_hidden_vb = (Variable(torch.zeros(self.batch_size, self.lstm_cha, self.lstm_hei, self.lstm_wid).type(self.dtype), volatile=not_training),
                               Variable(torch.zeros(self.batch_size, self.lstm_cha, self.lstm_hei, self.lstm_wid).type(self.dtype), volatile=not_training))

    # NOTE: to be called at the beginning of each rollout, detach the previous variable from the graph
    def _reset_lstm_hidden_vb_rollout(self):
        self.lstm_hidden_vb = (Variable(self.lstm_hidden_vb[0].data),
                               Variable(self.lstm_hidden_vb[1].data))

    def forward(self, input_vb):
        # input_vb: batch_size x cha x hei x wid
        if isinstance(input_vb.data, torch.cuda.FloatTensor):
            x_vb = nn.parallel.data_parallel(self.model_before, input_vb)
            x_vb = nn.parallel.data_parallel(self.model_middle, x_vb)
            output_vb = nn.parallel.data_parallel(self.model_after, x_vb)
        else:
            x_vb = self.model_before(input_vb)
            x_vb = self.model_middle(x_vb)
            output_vb = self.model_after(x_vb)
        return output_vb

    def forward_seq(self, input_vb, progress_id=0, alpha=1.):
        # print("================================================>     Generator", progress_id)
        # print("input     size --->", input_vb.size())
        # input_vb: (seq_len x batch_size) x cha x hei x wid
        if isinstance(input_vb.data, torch.cuda.FloatTensor):
            if not self.enable_progressive:
                # print("input_vb --->", input_vb.size())
                x_vb = nn.parallel.data_parallel(self.model_before, input_vb)
                # print("x_vb     --->", x_vb.size())
                x_vb = nn.parallel.data_parallel(self.model_middle, x_vb)
                # print("x_vb     --->", x_vb.size())
                if self.enable_lstm:
                    lstm_outs_vb = []
                    # for j in range(self.flo_len):
                    for flo_ind in range(len(self.flo_indices)):
                        j = self.flo_indices[flo_ind]   # NOTE: here j stores the flow to the frame that's (j+1) apart
                        # lstm_out_vb, _ = nn.parallel.data_parallel(self.lstm, x_vb[-(j+2):].unsqueeze(1))
                        lstm_out_vb, _ = self.lstm(x_vb[-(j+2):].unsqueeze(1))
                        lstm_outs_vb.append(lstm_out_vb)
                    output_vb = nn.parallel.data_parallel(self.model_after, torch.cat(lstm_outs_vb))
                else:
                    output_vb = nn.parallel.data_parallel(self.model_after, x_vb)
                    # print("output_vb--->", output_vb.size())
            else:
                # ----------------------------------------------------------------------
                if progress_id > self.progress_start:   # then we need to combine with the left stream
                    x_vb_0 = F.avg_pool2d(input_vb, kernel_size=self.progress_kernel, stride=self.progress_inc, padding=self.progress_kernel//2)
                    x_vb_0 = self.from_rgb_blocks[progress_id-1](x_vb_0)
                    # print("hidden_vb_0    --->", x_vb_0.size(), "from_rgb_blocks", progress_id-1)
                x_vb_1 = self.from_rgb_blocks[progress_id](input_vb)
                # print("hidden_vb_1    --->", x_vb_1.size(), "from_rgb_blocks", progress_id)
                if progress_id > 0: # NOTE: need to go through this convs
                    x_vb_1 = self.from_rgb_convs[progress_id](x_vb_1)
                    # print("hidden_vb_1    --->", x_vb_1.size(), "from_rgb_convs", progress_id)
                    x_vb_1 = F.avg_pool2d(x_vb_1, kernel_size=self.progress_kernel, stride=self.progress_inc, padding=self.progress_kernel//2)
                    # print("hidden_vb_1    --->", x_vb_1.size())
                if progress_id > self.progress_start:
                    x_vb = (1 - alpha) * x_vb_0 + alpha * x_vb_1
                    # print("   combined    --->", x_vb.size())
                else:
                    x_vb = x_vb_1
                    # print(" X combined    --->", x_vb.size())
                for i in range(progress_id-1, 0, -1):
                    x_vb = self.from_rgb_convs[i](x_vb)
                    x_vb = F.avg_pool2d(x_vb, kernel_size=self.progress_kernel, stride=self.progress_inc, padding=self.progress_kernel//2)
                    # print("    fromRGBcov --->", x_vb.size(), "from_rgb_convs", i)
                x_vb = self.from_rgb_convs[0](x_vb) # NOTE: always need to pass back through the 1st conv
                # print("    fromRGBcov --->", x_vb.size(), "from_rgb_convs", 0)
                # ----------------------------------------------------------------------
                x_vb = nn.parallel.data_parallel(self.model_middle, x_vb)
                # print("after   middle --->", x_vb.size())
                if self.enable_lstm:
                    lstm_outs_vb = []
                    # for j in range(self.flo_len):
                    for flo_ind in range(len(self.flo_indices)):
                        j = self.flo_indices[flo_ind]   # NOTE: here j stores the flow to the frame that's (j+1) apart
                        # lstm_out_vb, _ = nn.parallel.data_parallel(self.lstm, x_vb[-(j+2):].unsqueeze(1))
                        lstm_out_vb, _ = self.lstm(x_vb[-(j+2):].unsqueeze(1))
                        lstm_outs_vb.append(lstm_out_vb)
                    x_vb = torch.cat(lstm_outs_vb)
                    # print("after     lstm --->", x_vb.size())
                # ----------------------------------------------------------------------
                for i in range(0, progress_id):
                    x_vb = self.to_rgb_convs[i](x_vb) # NOTE: always need to pass back through the 1st conv
                    # print("      toRGBcov --->", x_vb.size(), "to_rgb_convs", i)
                    x_vb = F.upsample(x_vb, scale_factor=self.progress_inc, mode='nearest')
                    # print("after upsample --->", x_vb.size())
                if progress_id > self.progress_start:
                    x_vb_0 = self.to_rgb_blocks[progress_id-1](x_vb)
                    # print("hidden_vb_0    --->", x_vb_0.size(), "to_rgb_blocks", progress_id-1)
                x_vb_1 = self.to_rgb_convs[progress_id](x_vb)
                # print("hidden_vb_1    --->", x_vb_1.size(), "to_rgb_convs", progress_id)
                x_vb_1 = self.to_rgb_blocks[progress_id](x_vb_1)
                # print("hidden_vb_1    --->", x_vb_1.size(), "to_rgb_blocks", progress_id)
                if progress_id > self.progress_start:
                    x_vb = (1 - alpha) * x_vb_0 + alpha * x_vb_1
                    # print("   combined    --->", x_vb.size())
                else:
                    x_vb = x_vb_1
                    # print(" X combined    --->", x_vb.size())
                return x_vb
        else:
            if not self.enable_progressive:
                x_vb = self.model_before(input_vb)
                x_vb = self.model_middle(x_vb)
                if self.enable_lstm:
                    lstm_outs_vb = []
                    # for j in range(self.flo_len):
                    for flo_ind in len(flo_indices):
                        j = flo_indices[flo_ind]     # NOTE: here j stores the flow to the frame that's (j+1) apart
                        lstm_out_vb, _ = self.lstm(x_vb[-(j+2):].unsqueeze(1))
                        lstm_outs_vb.append(lstm_out_vb)
                    output_vb = self.model_after(torch.cat(lstm_outs_vb))
                else:
                    output_vb = self.model_after(x_vb)
            else:
                pass
        # print("output    size --->", input_vb.size())
        return output_vb

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
