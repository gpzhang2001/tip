构建好网络之后，对网络往往缺乏直观的认识，可以通过网络可视化对网络结构，各部分参数进行剖析，以便于进一步认识并可以根据此进行有针对性的修改。

这里直接给出实例代码，简单修改自己的参数就可以直接使用。

```python
import torch
from thop import clever_format, profile
from torchsummary import summary


if __name__ == "__main__":
    input_shape     = [640, 640]
    # your_body_param
    backbone        = ''
    # ....
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m       = your_body_net(your_body_param,
                       backbone=backbone).to(device)
    summary(m, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)
		# 如果只考虑乘法，这里可以不用乘2
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))

   
```

打印结果：

```shell
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 320, 320]           6,912
       BatchNorm2d-2         [-1, 64, 320, 320]             128
              SiLU-3         [-1, 64, 320, 320]               0
              Conv-4         [-1, 64, 320, 320]               0
             Focus-5         [-1, 64, 320, 320]               0
            Conv2d-6        [-1, 128, 160, 160]          73,728
       BatchNorm2d-7        [-1, 128, 160, 160]             256
     ........................................................
       
================================================================
Total params: 47,056,765
Trainable params: 47,056,765
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.69
Forward/backward pass size (MB): 4194304002730.40
Params size (MB): 179.51
Estimated Total Size (MB): 4194304002914.60
```

