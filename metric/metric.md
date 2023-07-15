[toc]



**主要是一些常用的评价指标，这里的程序均是经过验证的，可以直接使用**

# PSNR&SSIM

在图像处理中，二者是非常常见的评价指标。

## 概念

（填坑中...）

[PSNR&SSIM概念](./psnr&ssim.md)

## 使用

```python
import cv2
import numpy as np
import math


def psnr(img1, img2):
    mse = np.mean((img1/255. - img2/255.) ** 2)
    if mse < 1.0e-10:
        return 100
    pixel_max = 1
    return 20 * math.log10(pixel_max / math.sqrt(mse))


origina_img = cv2.imread('lena.png')
noise_img = cv2.imread('lena_gaussian.png')

result = psnr(origina_img, noise_img)
print(result)

```

```python
from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import cv2


def ssim(image1, image2, K, window_size, L):
    _, channel1, _, _ = image1.size()
    _, channel2, _, _ = image2.size()
    channel = min(channel1, channel2)

    # gaussian window generation
    sigma = 1.5  # default
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    # define constants
    # * L = 255 for constants doesn't produce meaningful results; thus L = 1
    # C1 = (K[0]*L)**2;
    # C2 = (K[1]*L)**2;
    C1 = K[0] ** 2
    C2 = K[1] ** 2

    mu1 = F.conv2d(image1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(image2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(image1 * image1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(image2 * image2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(image1 * image2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


if __name__ == "__main__":
    # opencv image load
    I1 = cv2.imread('lena.png')
    I2 = cv2.imread('lena_gaussian.png')

    # tensors
    I1 = torch.from_numpy(np.rollaxis(I1, 2)).float().unsqueeze(0) / 255.0
    I2 = torch.from_numpy(np.rollaxis(I2, 2)).float().unsqueeze(0) / 255.0
    # print(I1.size(), I2.size()) # returns torch([1,3,256,256])

    # tensor.autograd.Variable (Automatic differentiation variable)
    I1 = Variable(I1, requires_grad=True)
    I2 = Variable(I2, requires_grad=True)

    # default constants
    K = [0.01, 0.03]
    L = 255
    window_size = 11

    ssim_value = ssim(I1, I2, K, window_size, L)

    print(ssim_value.data)

```

# MSE

在计算psnr的时候已经计算了mse，可以看一下psnr的代码。

# LPIPS

也称为感知损失

## 概念

（填坑中...）

[lpips](./lpips.md)

## 使用

### install

```shell
git clone https://github.com/richzhang/PerceptualSimilarity
cd PerceptualSimilarity
pip install -r requirements.txt
```

then,the following code is all you need.

```shell
python lpips_2imgs.py -p0 imgs/ex_ref.png -p1 imgs/ex_p0.png --use_gpu
python lpips_2dirs.py -d0 imgs/ex_dir0 -d1 imgs/ex_dir1 -o imgs/example_dists.txt --use_gpu
python lpips_1dir_allpairs.py -d imgs/ex_dir_pair -o imgs/example_dists_pair.txt --use_gpu
```

# entropy（DE）

# EME

# AB

# PixDist

# LOE

# △E

