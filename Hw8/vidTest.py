
import cv2
import numpy as np
import glob

# Extract images from folder
imdir = './VideoSurvallience/'
ext = ['bmp']    # Add image formats here

files = []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
images = np.array([cv2.imread(file) for file in files])

# Extract the R,G,B data separately.
red = images[:,:,:,0]
green = images[:,:,:,1]
blue = images[:,:,:,2]

# Visualize test
# plt.imshow(red[0], cmap='Reds')
# plt.show()

# Reshape each colored array
# MR = red.reshape(3055, 19200).T
# MG = green.reshape(3055, 19200).T
# MB = blue.reshape(3055, 19200).T
# print(MR.shape)


#   Make Videos
size = (160,120)
background_vid = cv2.VideoWriter('vidtester.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), 10, size)
for i in range(len(red)):
    img =np.dstack((blue[i], green[i], red[i]))
    background_vid.write(img)

background_vid.release()