#!/usr/bin/env python
__authors__ = ["Tobias Uelwer", "Thomas Germer"]
__date__ = "2022/06/27"
__license__ = "MIT"

import numpy as np
import imageio
from shock import chromatic_shock, blur
import matplotlib.pyplot as plt

img = np.array(imageio.imread('kunstsammlung.jpg'))/255.0
img_gray = blur(img[200:700, 300:800], 1.2)

plt.figure()
plt.subplot(121)
plt.imshow(img_gray, cmap='gray')
plt.title('Input image')
plt.axis('off')
plt.subplot(122)
plt.imshow(chromatic_shock(img_gray), cmap='gray')
plt.title('Shock-filtered image')
plt.axis('off')
plt.savefig('result.png')
plt.tight_layout()
plt.show()