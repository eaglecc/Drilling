import numpy as np
import torch

size = 5
attn_shape = (1,size,size)
mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
print(mask)

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())