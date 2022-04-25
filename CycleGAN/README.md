CycleGAN for style transfer

## Reference

* [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593v7.pdf)
* https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

## CycleGAN概述

![CycleGAN_show](./photos/CycleGAN_show.png)

GAN模型由Generator和Discriminator两个网络模型组成，在unconditional GAN中，G输入random noise输出图片，D用于区分图片来源于G还是数据集，两者对抗学习，最终期望G可以生成与数据集中图片无法区分开的图片。

在image translation任务中，则是给定domain X、Y，期望G将X映射至与Y接近的分布。但在仅仅使用 the adversarial objective 的情况下，往往会出现许多x映射到同样的y的情况，从而失去的模型的多样性。

> However, such a translation does not guarantee that an individual input x and output y are paired up in a meaningful way – there are infinitely many mappings G that will induce the same distribution over ŷ. Moreover, in practice,we have found it **difficult to optimize the adversarial objective in isolation**: standard procedures often lead to the well-known problem of **mode collapse**, where all input images map to the same output image and the optimization fails to make progress .(paper introduction paragraph 6)

为解决这个问题，CycleGAN引入了“cycle consistent”，引入F为Y到X的映射，期望X由G映射到Y再由F映射回来后保持不变，基于此得到“cycle consistency loss”

![cycle consistency](./photos/cycle_consistency.png)

因此loss由 adversarial loss 和 cycle consistency loss 两部分组成。

adversarial loss 和GAN中一致（F映射的loss形式相同）

<img src="https://latex.codecogs.com/svg.image?\begin{aligned}\mathcal{L}_{\mathrm{GAN}}\left(G,&space;D_{Y},&space;X,&space;Y\right)&space;&=\mathbb{E}_{y&space;\sim&space;p_{\text&space;{data&space;}}(y)}\left[\log&space;D_{Y}(y)\right]&plus;\mathbb{E}_{x&space;\sim&space;p_{\text&space;{data&space;}}(x)}\left[\log&space;\left(1-D_{Y}(G(x))\right]\right.\end{aligned}" />

cycle consistency loss 如下

<img src="https://latex.codecogs.com/svg.image?\begin{aligned}\mathcal{L}_{\text&space;{cyc&space;}}(G,&space;F)&space;&=\mathbb{E}_{x&space;\sim&space;p_{\text&space;{data&space;}}(x)}\left[\|F(G(x))-x\|_{1}\right]&plus;\mathbb{E}_{y&space;\sim&space;p_{\text&space;{data&space;}}(y)}\left[\|G(F(y))-y\|_{1}\right]\end{aligned}" />

因此完整的loss如下

<img src="https://latex.codecogs.com/svg.image?\begin{aligned}\mathcal{L}\left(G,&space;F,&space;D_{X},&space;D_{Y}\right)&space;&=\mathcal{L}_{\mathrm{GAN}}\left(G,&space;D_{Y},&space;X,&space;Y\right)&space;\\&&plus;\mathcal{L}_{\mathrm{GAN}}\left(F,&space;D_{X},&space;Y,&space;X\right)&space;\\&&plus;\lambda&space;\mathcal{L}_{\mathrm{cyc}}(G,&space;F)\end{aligned}" />

特别地，在 Photo generation from paintings (见5.2 Applications)任务中，为了保持生成前后的图片色彩构成的一致性，引入了 identity loss，如下

<img src="https://latex.codecogs.com/svg.image?\mathcal{L}_{\text&space;{identity&space;}}(G,&space;F)=\mathbb{E}_{y&space;\sim&space;p_{\text&space;{data&space;}}(y)}\left[\|G(y)-y\|_{1}\right]&plus;&space;&space;&space;\mathbb{E}_{x&space;\sim&space;p_{\text&space;{data&space;}}(x)}\left[\|F(x)-x\|_{1}\right]" />

在实现中，对于adversarial loss，使用均方误差代替了log。

> we replace the negative log likelihood objective by a least-squares loss [35]. This loss is more stable during training and generates higher quality results.(paper 4.Implementation training details)

![adversarial loss](./photos/Adversarial_loss.jpg)

在模型方面，G由若干卷积层和residual blocks构成。D使用了70X70 Patch GANs。

> For the discriminator networks we use 70 × 70 PatchGANs [22, 30, 29], which aim to classify whether 70 × 70 overlapping image patches are real or fake. Such a patch-level discriminator architecture has fewer parameters than a full-image discriminator and can work on arbitrarily-sized images in a fully convolutional fashion.(paper 4.Implementation Network Architecture)

## Code

### Generator Model

![G model](./photos/G_model.jpg)

```python
'''in cycle_gan_model.py---class CycleGANModel(BaseModel)---__init__'''
self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
# same as self.netG_B

'''in networks.py---define_G(...)'''
net = None
norm_layer = get_norm_layer(norm_type=norm)

if netG == 'resnet_9blocks':
    net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, 							use_dropout=use_dropout, n_blocks=9)
    
return init_net(net, init_type, init_gain, gpu_ids)  # net to gpu and init weights


'''in nerwork.py---Class ResnetGenerator(nn.Module)---__init__(...)'''
model = [nn.ReflectionPad2d(3),
         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
         norm_layer(ngf),
         nn.ReLU(True)]

n_downsampling = 2
for i in range(n_downsampling):  # add downsampling layers
    mult = 2 ** i
    model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
              norm_layer(ngf * mult * 2),
              nn.ReLU(True)]

mult = 2 ** n_downsampling
for i in range(n_blocks):       # add ResNet blocks
	model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

for i in range(n_downsampling):  # add upsampling layers
    mult = 2 ** (n_downsampling - i)
    model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                 kernel_size=3, stride=2,
                                 padding=1, output_padding=1,
                                 bias=use_bias),
              norm_layer(int(ngf * mult / 2)),
              nn.ReLU(True)]
    model += [nn.ReflectionPad2d(3)]
    model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
    model += [nn.Tanh()]

self.model = nn.Sequential(*model)

'''in network.py---class ResnetBlock(nn.Module)---__init__(...)'''
super(ResnetBlock, self).__init__()
self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

'''in network.py---class ResnetBlock(nn.Module)---build_conv_block(...)'''
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

conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
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
conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

return nn.Sequential(*conv_block)
    
'''in network.py---class ResnetBlock(nn.Module)---forward()'''
def forward(self, x):
    """Forward function (with skip connections)"""
    out = x + self.conv_block(x)  # add skip connections
    return out
    
```





