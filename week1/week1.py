import torch
import torchvision
import cv2
import cv2 #opencv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""
#test opencv
img = cv2.imread('cat.jpg')
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

print(torch.__version__)
print(torchvision.__version__)

#test pytorch
print(torch.__version__)
print(torchvision.__version__)
#test numpy & matplotlib &pandas
x = np.arange(1, 10)
y = x**2

plt.figure(1)
plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('test')
plt.show()

data = {'col1': x, 'col2': y}
df = pd.DataFrame(data)
df.to_csv('test.csv', header = 0, index = None)

