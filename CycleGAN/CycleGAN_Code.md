# CycleGAN_Code



## model

### generator 

use ResNet9 by default

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



### discrimnator 

