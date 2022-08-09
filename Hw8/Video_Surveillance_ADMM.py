### Homework 8
# ADMM Robust PCA for Video Surveillence 
#
# Princess Tara Zamani
# 11/21/2021

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA

def main():
    # Initialize
    p = 1e-2
    
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
    MR = red.reshape(3055, 19200).T
    MG = green.reshape(3055, 19200).T
    MB = blue.reshape(3055, 19200).T
    print(MR.shape)

    # Roadmap: Run ADMM on R,G,B channels seperately then assemble them together
    # Loop through M = MR, MG, MB

    # Lists containing final L and S values for MR, MG, MB
    L_list = []
    S_list = []

    # Implementation per M = MR, MG, MB value
    for M in [MR, MG, MB]:

        # Divide data into 5 smaller matrices, each with dimensioin 19200 x 611
        [M1, M2, M3, M4, M5] = [M[:,:200], M[:,611:1222], M[:,1222:1833], M[:,1833:2444], M[:,2444:]]
        lam = 1/np.sqrt(19200) #1/np.sqrt(19200) # might need tuning

        L_concat = np.zeros((M1.shape[0], 1))  
        S_concat = np.zeros((M1.shape[0], 1)) 
        
        # Loop m = M1, ..., M5 and do ADMM alg to perform separation
        for m in [M1]: #, M2, M3, M4, M5]:
            L = m
            S = np.zeros((m.shape[0], m.shape[1]))
            V = np.zeros((m.shape[0], m.shape[1])) 
            for t in range(1): #len(m)-1):
                # Update L_t+1
                L_inner_calc = m - S - V/p
                u, s, v = LA.svd(L_inner_calc) # Do partial SVD instead of first 100 terms
                # sigma = LA.diagsvd(s, L_inner_calc.shape[0], L_inner_calc.shape[1]) 
                # L_next = np.matmul(np.matmul(u, np.maximum(sigma - 1/p, 0)), v.T) # No np.diag function
                sigma = np.maximum(s - 1/p, 0) 
                L_next = np.matmul(np.matmul(u, LA.diagsvd(sigma, L_inner_calc.shape[0], L_inner_calc.shape[1]) ), v.T) # No np.diag function

                # Update S_t+1
                S_inner_calc = m - L_next - V/p
                tao = lam*1/p
                S_next = np.copy(S)
                S_next[S_inner_calc > tao] = S_inner_calc [S_inner_calc > tao]- tao
                S_next[S_inner_calc < -tao] = S_inner_calc[S_inner_calc < -tao] + tao
                S_next[np.abs(S_inner_calc) <= tao] = 0

                # Update V_t+1
                V_next = V + p*(L_next + S_next - m)

                L = L_next
                S = S_next
                V = V_next

            # Append L and S for this m subset (concatenation)
            L_concat = np.concatenate((L_concat, L), axis=1)
            S_concat = np.concatenate((S_concat, S), axis=1)
    
        L_list.append(L_concat[:,1:])
        S_list.append(S_concat[:,1:])

    # We now have the separate background matrices (L_list) and moving object matrices (S_list)
    # Reshape columns back to 160x120 images
    photonum = 200
    L_list = np.reshape(L_list, (np.shape(L_list)[0], 120, 160, photonum))
    S_list = np.reshape(S_list, (np.shape(S_list)[0], 120, 160, photonum))

    # Reshape to be 3, photonum, 120, 160
    L_list = np.transpose(L_list, axes=(0,3,1,2))
    S_list = np.transpose(S_list, axes=(0,3,1,2))

    # Assemble them into an RBG background video an RGB moving object video
    #   Background
    size = (160,120)
    background_vid = cv2.VideoWriter('background.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), 10, size)
    for i in range(photonum):
        img =np.dstack((L_list[2][i], L_list[1][i], L_list[0][i])) # image in BGR 
        background_vid.write(np.uint8(img))
    background_vid.release()

    #   Moving Objects
    size = (160,120)
    movingObj_vid = cv2.VideoWriter('movingObjects.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), 10, size)
    for i in range(photonum):
        img =np.dstack((S_list[2][i], S_list[1][i], S_list[0][i]))
        movingObj_vid.write(np.uint8(img))
    movingObj_vid.release()

    







if __name__=='__main__':
    main()
