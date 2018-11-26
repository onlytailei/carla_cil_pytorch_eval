from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from agents.imitation.modules.resnet import ResnetGenerator


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and (classname.find('ConvLSTM') == -1 and classname.find('ConvLSTMCell') == -1 and classname.find('ConvLSTMSeq') == -1):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return norm_layer

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False,
             enable_progressive=False, progress_start=0, progress_chas=[], progress_inc=2, progress_kernel=3,
             enable_lstm=False, flo_len=0, flo_mode="", flo_indices=[], batch_size=1, lstm_cha=256, lstm_hei=64, lstm_wid=64, dtype=torch.FloatTensor):#, gpu_ids=[]):
    netG = None
    # use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    # if use_gpu:
    #     assert(torch.cuda.is_available())
    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                               enable_progressive=enable_progressive, progress_start=progress_start, progress_chas=progress_chas, progress_inc=progress_inc, progress_kernel=progress_kernel,
                               enable_lstm=enable_lstm, flo_len=flo_len, flo_mode=flo_mode, flo_indices=flo_indices, batch_size=batch_size, lstm_cha=lstm_cha, lstm_hei=lstm_hei, lstm_wid=lstm_wid, dtype=dtype,
                               n_blocks=9)#, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                               enable_progressive=enable_progressive, progress_start=progress_start, progress_chas=progress_chas, progress_inc=progress_inc, progress_kernel=progress_kernel,
                               enable_lstm=enable_lstm, flo_len=flo_len, flo_mode=flo_mode, flo_indices=flo_indices, batch_size=batch_size, lstm_cha=lstm_cha, lstm_hei=lstm_hei, lstm_wid=lstm_wid, dtype=dtype,
                               n_blocks=6)#, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    # if len(gpu_ids) > 0:
    #     netG.cuda(device_id=gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False,
             enable_progressive=False, progress_start=0, progress_chas=[], progress_inc=2, progress_kernel=3,
             enable_round_input_D=False):#, gpu_ids=[]):
    netD = None
    # use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    # if use_gpu:
    #     assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   enable_progressive=enable_progressive, progress_start=progress_start, progress_chas=progress_chas, progress_inc=progress_inc, progress_kernel=progress_kernel,
                                   enable_round_input_D=enable_round_input_D)#, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   enable_progressive=enable_progressive, progress_start=progress_start, progress_chas=progress_chas, progress_inc=progress_inc, progress_kernel=progress_kernel,
                                   enable_round_input_D=enable_round_input_D)#, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    # if use_gpu:
    #     netD.cuda(device_id=gpu_ids[0])
    netD.apply(weights_init)
    return netD

def define_styleD(input_nc, ndf, which_model_styleD, n_layers_D=3, norm='batch', use_sigmoid=False,
                  enable_progressive=False, progress_start=0, progress_chas=[], progress_inc=2, progress_kernel=3,
                  enable_round_input_D=False):#, gpu_ids=[]):
    netD = None
    # use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    # if use_gpu:
    #     assert(torch.cuda.is_available())
    if ("conv" in which_model_styleD):
        netD = StyleBasicDiscriminator(which_model_styleD, input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                       enable_progressive=enable_progressive, progress_start=progress_start, progress_chas=progress_chas, progress_inc=progress_inc, progress_kernel=progress_kernel,
                                       enable_round_input_D=enable_round_input_D)#, gpu_ids=gpu_ids)
    elif which_model_styleD == 'pool':
        netD = StylePoolDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                      enable_progressive=enable_progressive, progress_start=progress_start, progress_chas=progress_chas, progress_inc=progress_inc, progress_kernel=progress_kernel,
                                      enable_round_input_D=enable_round_input_D)#, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_styleD)
    # if use_gpu:
    #     netD.cuda(device_id=gpu_ids[0])
    netD.apply(weights_init)
    return netD

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 dtype=torch.FloatTensor, enable_stable=False):
                #  tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.dtype = dtype
        self.enable_stable = enable_stable
        # self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                if self.enable_stable:
                    if np.random.rand()>0.98: # TODO hardcode batch size is 10 when stable
                        real_tensor = (torch.rand(input.size(0),1,1,1)>0.5).type(self.dtype) * torch.ones(input.size()).type(self.dtype)
                    else:
                        real_tensor = torch.FloatTensor(input.size()).fill_(self.real_label).type(self.dtype)
                    real_tensor = real_tensor + ((torch.rand(input.size())-0.5)/4).type(self.dtype)
                else:
                    real_tensor = torch.FloatTensor(input.size()).fill_(self.real_label).type(self.dtype)
                # real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                if self.enable_stable:
                    if np.random.rand()>0.98:
                        fake_tensor = (torch.rand(input.size(0),1,1,1)>0.5).type(self.dtype) * torch.ones(input.size()).type(self.dtype)
                    else:
                        fake_tensor = torch.FloatTensor(input.size()).fill_(self.fake_label).type(self.dtype)
                    fake_tensor = fake_tensor + ((torch.rand(input.size())-0.5)/4).type(self.dtype)
                else:
                    fake_tensor = torch.FloatTensor(input.size()).fill_(self.fake_label).type(self.dtype)
                # fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input_list, target_is_real):
        #target_tensor = self.get_target_tensor(input, target_is_real)
        #return self.loss(input, target_tensor)
        output_loss=0
        for input in input_list:
            target_tensor = self.get_target_tensor(input, target_is_real)
            output_loss = output_loss + self.loss(input, target_tensor)
            #print(self.loss(input,target_tensor))
        return output_loss

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 enable_progressive=False, progress_start=0, progress_chas=[], progress_inc=2, progress_kernel=3,
                 enable_round_input_D=False):#, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        # self.gpu_ids = gpu_ids

        self.enable_progressive = enable_progressive
        self.progress_start = progress_start
        self.progress_chas = progress_chas
        self.progress_inc = progress_inc
        self.progress_kernel = progress_kernel
        self.progress_num = len(self.progress_chas) - 1

        self.enable_round_input_D = enable_round_input_D

        self.input_nc = input_nc
        self.use_sigmoid = use_sigmoid

        if not self.enable_progressive:
            kw = 4
            padw = int(np.ceil((kw-1)//2))
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

            nf_mult = 1
            nf_mult_prev = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2**n, 8)
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]

            nf_mult_prev = nf_mult
            nf_mult = min(2**n_layers, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=1, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

            if use_sigmoid:
                sequence += [nn.Sigmoid()]

            self.model = nn.Sequential(*sequence)
        else:
            self.from_rgb_blocks = nn.ModuleList()
            self.from_rgb_convs = nn.ModuleList()
            for progress_id in range(self.progress_num):
                if progress_id < self.progress_start:
                    self.from_rgb_blocks.append(nn.Sequential())
                else:
                    self.from_rgb_blocks.append(nn.Sequential(nn.Conv2d(self.input_nc, self.progress_chas[progress_id+1], kernel_size=1)))
                self.from_rgb_convs.append(nn.Sequential(nn.ReflectionPad2d(self.progress_kernel//2),
                                                         nn.Conv2d(self.progress_chas[progress_id+1], self.progress_chas[progress_id], kernel_size=self.progress_kernel),
                                                         norm_layer(self.progress_chas[progress_id]),
                                                         nn.ReLU(True),
                                                         nn.ReflectionPad2d(self.progress_kernel//2),
                                                         nn.Conv2d(self.progress_chas[progress_id], self.progress_chas[progress_id], kernel_size=self.progress_kernel),
                                                         norm_layer(self.progress_chas[progress_id]),
                                                         nn.ReLU(True)))

    def forward(self, input_vb):
        # input_vb: batch_size x cha x hei x wid
        if isinstance(input_vb.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input_vb)
        else:
            return self.model(input_vb)

    def forward_seq(self, input_vb, progress_id=0, alpha=1.):
        # print("================================================> Discriminator", progress_id)
        # print("input     size --->", input_vb.size())
        # input_vb: (seq_len x batch_size) x cha x hei x wid
        if self.enable_round_input_D:
            input_vb = (torch.round((input_vb / 2. + 0.5) * 255.) / 255. - 0.5) * 2.
        if isinstance(input_vb.data, torch.cuda.FloatTensor):
            if not self.enable_progressive:
                output_vb = nn.parallel.data_parallel(self.model, input_vb)
            else:
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
                if self.use_sigmoid:
                    output_vb = F.sigmoid(x_vb)
                else:
                    output_vb = x_vb
        else:
            if not self.enable_progressive:
                output_vb = self.model(input_vb)
            else:
                pass
        # print("output    size --->", output_vb.size())
        return [output_vb]

# Defines the PatchGAN discriminator with the specified arguments.
class StyleBasicDiscriminator(nn.Module):
    def __init__(self, styleD_model, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 enable_progressive=False, progress_start=0, progress_chas=[], progress_inc=2, progress_kernel=3,
                 enable_round_input_D=False):#, gpu_ids=[]):
        super(StyleBasicDiscriminator, self).__init__()
        # self.gpu_ids = gpu_ids

        self.styleD_model = styleD_model
        self.enable_progressive = enable_progressive
        self.progress_start = progress_start
        self.progress_chas = progress_chas
        self.progress_inc = progress_inc
        self.progress_kernel = progress_kernel
        self.progress_num = len(self.progress_chas) - 1

        self.enable_round_input_D = enable_round_input_D

        self.input_nc = input_nc
        self.use_sigmoid = use_sigmoid

        self.kernel_size = 3
        self.input_chas = [64, 128, 256, 512]

        if not self.enable_progressive:
            self.gram0 = nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                       nn.Conv2d(1, 32, kernel_size=self.kernel_size, stride=1),
                                       #nn.Conv2d(1, 128, kernel_size=self.kernel_size, stride=1),
                                       nn.LeakyReLU(0.2, True))
            self.gram1 = nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                       nn.Conv2d(1, 32, kernel_size=self.kernel_size, stride=1),
                                       #nn.Conv2d(1, 64, kernel_size=self.kernel_size, stride=1),
                                       nn.LeakyReLU(0.2, True))
            self.gram2 = nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                       nn.Conv2d(1, 32, kernel_size=self.kernel_size, stride=1),
                                       nn.LeakyReLU(0.2, True))
            self.gram3 = nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                       nn.Conv2d(1, 32, kernel_size=self.kernel_size, stride=2),
                                       nn.LeakyReLU(0.2, True))

            self.conv1 =  nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                       nn.Conv2d(64, 32, kernel_size=self.kernel_size, stride=2))
                                       #nn.Conv2d(64, 64, kernel_size=self.kernel_size, stride=2),
                                       #nn.LeakyReLU(0.2, True))

            self.conv2 =  nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                        nn.Conv2d(64, 32, kernel_size=self.kernel_size, stride=2))
                                        #nn.Conv2d(128, 128, kernel_size=self.kernel_size, stride=2),
                                        #nn.LeakyReLU(0.2, True))
            self.conv3 =  nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                        nn.Conv2d(64, 32, kernel_size=self.kernel_size, stride=2))
                                        #nn.Conv2d(256, 256, kernel_size=self.kernel_size, stride=2),
                                        #nn.LeakyReLU(0.2, True))
            self.conv4 =  nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                        nn.Conv2d(32, 1, kernel_size=self.kernel_size, stride=1))
                                        #nn.Conv2d(256, 1, kernel_size=self.kernel_size, stride=1))

            # for input len 2
            if (self.styleD_model == "conv024") or (self.styleD_model == "conv014"):
                self.convgram2_1 = nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                            nn.Conv2d(32, 32, kernel_size=self.kernel_size, stride=2),
                                            nn.LeakyReLU(0.2, True))
                self.convgram2_2 = nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                            nn.Conv2d(64, 32, kernel_size=self.kernel_size, stride=2),
                                            nn.LeakyReLU(0.2, True))
                self.convgram2_3 = nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                            nn.Conv2d(32, 1, kernel_size=self.kernel_size, stride=1),
                                            nn.LeakyReLU(0.2, True))

            # for input len 1
            elif self.styleD_model == "conv04":
                self.convgram1_1 =  nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                            nn.Conv2d(1, 64, kernel_size=self.kernel_size, stride=2),
                                            nn.LeakyReLU(0.2, True))
                                            #nn.Conv2d(256, 1, kernel_size=self.kernel_size, stride=1))
                self.convgram1_2 =  nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                            nn.Conv2d(64, 64, kernel_size=self.kernel_size, stride=1),
                                            nn.LeakyReLU(0.2, True))
                                            #nn.Conv2d(256, 1, kernel_size=self.kernel_size, stride=1))
                self.convgram1_3 =  nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                            nn.Conv2d(64, 1, kernel_size=self.kernel_size, stride=1))
                                            #nn.Conv2d(256, 1, kernel_size=self.kernel_size, stride=1))


            elif self.styleD_model == "conv01234":
                self.convgram3 = nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                               nn.Conv2d(32, 1, kernel_size=self.kernel_size))
                                               #nn.Sigmoid()) # 32x 256 x256
                self.convgram2 = nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                               nn.Conv2d(32, 1, kernel_size=self.kernel_size))
                                               #nn.Sigmoid()) # 32x 256 x256
                self.convgram1 = nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                               nn.Conv2d(32, 1, kernel_size=self.kernel_size))
                                               #nn.Sigmoid()) # 64x 128 x128
                self.convgram0 = nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2),
                                               nn.Conv2d(32, 1, kernel_size=self.kernel_size))
                                               #nn.Sigmoid()) # 128x 64 x64
            else:
                raise NotImplementedError("wrong styld D model")

        else:
            raise NotImplementedError("not implemented with progressive mode")

    def forward(self, input_vb):
        # input_vb: batch_size x cha x hei x wid
        if isinstance(input_vb.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input_vb)
        else:
            return self.model(input_vb)

    def forward_seq(self, input_vb, progress_id=0, alpha=1.):
        # print("================================================> Discriminator", progress_id)
        #print("input     size --->", len(input_vb))
        #print("input     size --->", input_vb[0].size())
        #print("input     size --->", input_vb[1].size())
        #print("input     size --->", input_vb[2].size())
        #print("input     size --->", input_vb[3].size())


        if len(input_vb)==4:
            x3_vb = self.gram3(input_vb[3].unsqueeze(1))
            x2_vb = self.gram2(input_vb[2].unsqueeze(1))
            x1_vb = self.gram1(input_vb[1].unsqueeze(1))
            x0_vb = self.gram0(input_vb[0].unsqueeze(1))
            x2_vb_prime = self.conv1(torch.cat((x3_vb, x2_vb), dim=1))
            x1_vb_prime = self.conv2(torch.cat((x2_vb_prime, x1_vb), dim=1))
            x0_vb_prime = self.conv3(torch.cat((x1_vb_prime, x0_vb), dim=1))
            x_vb = self.conv4(x0_vb_prime)

            if self.styleD_model == "conv4":
                if self.use_sigmoid:
                    output_vb = [F.sigmoid(x_vb)]
                else:
                    output_vb = [x_vb]
            elif self.styleD_model == "conv01234":
                x3_output = self.convgram3(x3_vb)
                x2_output = self.convgram2(x2_vb)
                x1_output = self.convgram1(x1_vb)
                x0_output = self.convgram0(x0_vb)
                if self.use_sigmoid:
                    output_vb = [
                            F.sigmoid(x_vb),
                            F.sigmoid(x0_output),
                            F.sigmoid(x1_output),
                            F.sigmoid(x2_output),
                            F.sigmoid(x3_output)]
                else:
                    output_vb = [x_vb, x0_output, x1_output, x2_output, x3_output]

        elif len(input_vb) == 2:
            if self.styleD_model == "conv014":
                x1_vb = self.gram3(input_vb[1].unsqueeze(1))  #32 64 64
                x0_vb = self.gram0(input_vb[0].unsqueeze(1))  #32 64 64

                x0_out = self.convgram2_2(torch.cat((x0_vb, x1_vb),dim=1)) # 64 64 64
                x0_out = self.convgram2_3(x0_out) #1 32 32

            elif self.styleD_model == "conv024":
                x1_vb = self.gram3(input_vb[1].unsqueeze(1))  #32 128 128
                x0_vb = self.gram0(input_vb[0].unsqueeze(1))  #32 64 64

                x1_mid = self.convgram2_1(x1_vb) # 32 64 64
                x0_out = self.convgram2_2(torch.cat((x0_vb, x1_mid),dim=1)) # 64 64 64
                x0_out = self.convgram2_3(x0_out) #1 32 32

            if self.use_sigmoid:
                output_vb = [F.sigmoid(x0_out)]
            else:
                output_vb = [x0_out]

        elif len(input_vb) == 1:
            if self.styleD_model == "conv04":
                x0_vb = self.convgram1_1(input_vb[0].unsqueeze(1))
                x0_vb = self.convgram1_2(x0_vb)
                x0_vb = self.convgram1_3(x0_vb)

                if self.use_sigmoid:
                    output_vb = [F.sigmoid(x0_vb)]
                else:
                    output_vb = [x0_vb]
        else:
            raise NotImplementedError("input length mush be 1 2 4")
        #print (output_vb[0].size())

        return output_vb
        # # x
        # print(x3_vb.size())
        # print(x2_vb.size())
        # print(x1_vb.size())
        # print(x0_vb.size())
        # print(x2_vb_prime.size())
        # print(output.size())


        # if self.use_sigmoid:
        #     output_vb = [
        #             F.sigmoid(x_vb),
        #             F.sigmoid(x0_output),
        #             F.sigmoid(x1_output),
        #              F.sigmoid(x2_output),
        #             F.sigmoid(x3_output)]
        # else:
        #     output_vb = [x_vb, x0_output, x1_output, x2_output, x3_output]
        # return output_vb

class StylePoolDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 enable_progressive=False, progress_start=0, progress_chas=[], progress_inc=2, progress_kernel=3,
                 enable_round_input_D=False):#, gpu_ids=[]):
        super(StylePoolDiscriminator, self).__init__()
        # self.gpu_ids = gpu_ids

        self.enable_progressive = enable_progressive
        self.progress_start = progress_start
        self.progress_chas = progress_chas
        self.progress_inc = progress_inc
        self.progress_kernel = progress_kernel
        self.progress_num = len(self.progress_chas) - 1

        self.enable_round_input_D = enable_round_input_D

        self.input_nc = input_nc
        self.use_sigmoid = use_sigmoid

        self.kernel_size = 1
        self.avg_kernel_size=2
        self.input_chas = [64, 128, 256, 512]

        if not self.enable_progressive:
            self.gram0 = nn.Sequential(nn.Conv2d(1, 128, kernel_size=self.kernel_size, stride=1),
                                       nn.LeakyReLU(0.2, True))
            self.gram1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=self.kernel_size, stride=1),
                                       nn.LeakyReLU(0.2, True))
            self.gram2 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=self.kernel_size, stride=1),
                                       nn.LeakyReLU(0.2, True))
            self.gram3 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=self.kernel_size, stride=1),
                                       nn.AvgPool2d(self.avg_kernel_size, stride=2),
                                       nn.LeakyReLU(0.2, True))

            #self.conv1 =  nn.Conv2d(64, 64, kernel_size=self.kernel_size, stride=1)
            #self.conv2 =  nn.Conv2d(128, 128, kernel_size=self.kernel_size, stride=1)
            #self.conv3 =  nn.Conv2d(256, 256, kernel_size=self.kernel_size, stride=1)
            #self.conv4 =  nn.Conv2d(256, 1, kernel_size=self.kernel_size, stride=1)


            self.conv1 =  nn.Sequential(nn.Conv2d(64, 64, kernel_size=self.kernel_size, stride=1),
                                       nn.LeakyReLU(0.2, True))
            self.conv2 =  nn.Sequential(nn.Conv2d(128, 128, kernel_size=self.kernel_size, stride=1),
                                        nn.LeakyReLU(0.2, True))
            self.conv3 =  nn.Sequential(nn.Conv2d(256, 256, kernel_size=self.kernel_size, stride=1),
                                        nn.LeakyReLU(0.2, True))
            self.conv4 =  nn.Conv2d(256, 1, kernel_size=self.kernel_size, stride=1)

        else:
            pass

    def forward(self, input_vb):
        # input_vb: batch_size x cha x hei x wid
        if isinstance(input_vb.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input_vb)
        else:
            return self.model(input_vb)

    def forward_seq(self, input_vb, progress_id=0, alpha=1.):
        # print("================================================> Discriminator", progress_id)
        #print("input     size --->", len(input_vb))
        #print("input     size --->", input_vb[0].size())
        #print("input     size --->", input_vb[1].size())
        #print("input     size --->", input_vb[2].size())
        #print("input     size --->", input_vb[3].size())
        x3_vb = self.gram3(input_vb[3].unsqueeze(1))
        x2_vb = self.gram2(input_vb[2].unsqueeze(1))
        x1_vb = self.gram1(input_vb[1].unsqueeze(1))
        x0_vb = self.gram0(input_vb[0].unsqueeze(1))
        #print(x3_vb.size())
        #print(x2_vb.size())
        #print(x1_vb.size())
        #print(x0_vb.size())
        # print(x2_vb_prime.size())
        # print(output.size())
        x2_vb_prime = self.conv1(F.avg_pool2d(torch.cat((x3_vb, x2_vb), dim=1),
            self.avg_kernel_size, stride=2))
        x1_vb_prime = self.conv2(F.avg_pool2d(torch.cat((x2_vb_prime, x1_vb), dim=1),
            self.avg_kernel_size, stride=2))
        x0_vb_prime = self.conv3(F.avg_pool2d(torch.cat((x1_vb_prime, x0_vb), dim=1),
            self.avg_kernel_size, stride=2))
        x_vb = self.conv4(x0_vb_prime)

        # # x


        if self.use_sigmoid:
            output_vb = F.sigmoid(x_vb)
        else:
            output_vb = x_vb
        return [output_vb]
