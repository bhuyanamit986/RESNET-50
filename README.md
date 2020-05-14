# RESNET-50
# Introduction

---

### *The Problem of very deep neural network*
 - The main benefit of a very deep network is that it can represent very complex functions. It can also learn features at many different levels of abstraction, from edges (at the shallower layers, closer to the input) to very complex features (at the deeper layers, closer to the output).
 - However, using a deeper network doesn't always help. A huge barrier to training them is vanishing gradients: very deep networks often have a gradient signal that goes to zero quickly, thus making gradient descent prohibitively slow.
 - More specifically, during gradient descent, as you backprop from the final layer back to the first layer, you are multiplying by the weight matrix on each step, and thus the gradient can decrease exponentially quickly to zero (or, in rare cases, grow exponentially quickly and "explode" to take very large values).
 - During training, you might therefore see the magnitude (or norm) of the gradient for the shallower layers decrease to zero very rapidly as training proceeds.
 
### *Components of a residual network*
 
#### - The Identinty Block

First component of main path:

The first CONV2D has  *F1*  filters of shape *(1,1)* and a stride of *(1,1)*. Its padding is "valid" and its name should be conv_name_base + '2a'. Use 0 as the seed for the random initialization.
The first BatchNorm is normalizing the 'channels' axis. Its name should be bn_name_base + '2a'.
Then apply the ReLU activation function. This has no name and no hyperparameters.
Second component of main path:

The second CONV2D has  *F2*  filters of shape  *(f,f)*  and a stride of *(1,1)*. Its padding is "same" and its name should be conv_name_base + '2b'. Use 0 as the seed for the random initialization.
The second BatchNorm is normalizing the 'channels' axis. Its name should be bn_name_base + '2b'.
Then apply the ReLU activation function. This has no name and no hyperparameters.
Third component of main path:

The third CONV2D has  *F3*  filters of shape *(1,1)* and a stride of *(1,1)*. Its padding is "valid" and its name should be conv_name_base + '2c'. Use 0 as the seed for the random initialization.
The third BatchNorm is normalizing the 'channels' axis. Its name should be bn_name_base + '2c'.
Note that there is no ReLU activation function in this component.
Final step:

The X_shortcut and the output from the 3rd layer X are added together.
Then apply the ReLU activation function. This has no name and no hyperparameters.

#### The Convolutional Block

First component of main path:

The first CONV2D has  *F1*  filters of shape (1,1) and a stride of (s,s). Its padding is "valid" and its name should be conv_name_base + '2a'. Use 0 as the glorot_uniform seed.
The first BatchNorm is normalizing the 'channels' axis. Its name should be bn_name_base + '2a'.
Then apply the ReLU activation function. This has no name and no hyperparameters.
Second component of main path:

The second CONV2D has  *F2*  filters of shape *(f,f)* and a stride of *(1,1)*. Its padding is "same" and it's name should be conv_name_base + '2b'. Use 0 as the glorot_uniform seed.
The second BatchNorm is normalizing the 'channels' axis. Its name should be bn_name_base + '2b'.
Then apply the ReLU activation function. This has no name and no hyperparameters.
Third component of main path:

The third CONV2D has  *F3*  filters of shape *(1,1)* and a stride of *(1,1)*. Its padding is "valid" and it's name should be conv_name_base + '2c'. Use 0 as the glorot_uniform seed.
The third BatchNorm is normalizing the 'channels' axis. Its name should be bn_name_base + '2c'. Note that there is no ReLU activation function in this component.
Shortcut path:

The CONV2D has  *F3* filters of shape *(1,1)* and a stride of *(s,s)*. Its padding is "valid" and its name should be conv_name_base + '1'. Use 0 as the glorot_uniform seed.
The BatchNorm is normalizing the 'channels' axis. Its name should be bn_name_base + '1'.
Final step:

The shortcut and the main path values are added together.
Then apply the ReLU activation function. This has no name and no hyperparameters.

---

# Building the Model

---

 - Zero-padding pads the input with a pad of (3,3)
 - Stage 1:
The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). Its name is "conv1".
BatchNorm is applied to the 'channels' axis of the input.
MaxPooling uses a (3,3) window and a (2,2) stride.
 - Stage 2:
The convolutional block uses three sets of filters of size [64,64,256], "f" is 3, "s" is 1 and the block is "a".
The 2 identity blocks use three sets of filters of size [64,64,256], "f" is 3 and the blocks are "b" and "c".
 - Stage 3:
The convolutional block uses three sets of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
The 3 identity blocks use three sets of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
 - Stage 4:
The convolutional block uses three sets of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
The 5 identity blocks use three sets of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
 - Stage 5:
The convolutional block uses three sets of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
The 2 identity blocks use three sets of filters of size [512, 512, 2048], "f" is 3 and the blocks are "b" and "c".
 - The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
 - The 'flatten' layer doesn't have any hyperparameters or name.
 - The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation. Its name should be 'fc' + str(classes).
 
 _______________________________________________________________________________________
 
 Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 64, 64, 3)     0                                            
____________________________________________________________________________________________________
zero_padding2d_1 (ZeroPadding2D) (None, 70, 70, 3)     0           input_1[0][0]                    
____________________________________________________________________________________________________
conv1 (Conv2D)                   (None, 32, 32, 64)    9472        zero_padding2d_1[0][0]           
____________________________________________________________________________________________________
bn_conv1 (BatchNormalization)    (None, 32, 32, 64)    256         conv1[0][0]                      
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 32, 32, 64)    0           bn_conv1[0][0]                   
____________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)   (None, 15, 15, 64)    0           activation_4[0][0]               
____________________________________________________________________________________________________
res2a_branch2a (Conv2D)          (None, 15, 15, 64)    4160        max_pooling2d_1[0][0]            
____________________________________________________________________________________________________
bn2a_branch2a (BatchNormalizatio (None, 15, 15, 64)    256         res2a_branch2a[0][0]             
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 15, 15, 64)    0           bn2a_branch2a[0][0]              
____________________________________________________________________________________________________
res2a_branch2b (Conv2D)          (None, 15, 15, 64)    36928       activation_5[0][0]               
____________________________________________________________________________________________________
bn2a_branch2b (BatchNormalizatio (None, 15, 15, 64)    256         res2a_branch2b[0][0]             
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 15, 15, 64)    0           bn2a_branch2b[0][0]              
____________________________________________________________________________________________________
res2a_branch2c (Conv2D)          (None, 15, 15, 256)   16640       activation_6[0][0]               
____________________________________________________________________________________________________
res2a_branch1 (Conv2D)           (None, 15, 15, 256)   16640       max_pooling2d_1[0][0]            
____________________________________________________________________________________________________
bn2a_branch2c (BatchNormalizatio (None, 15, 15, 256)   1024        res2a_branch2c[0][0]             
____________________________________________________________________________________________________
bn2a_branch1 (BatchNormalization (None, 15, 15, 256)   1024        res2a_branch1[0][0]              
____________________________________________________________________________________________________
add_2 (Add)                      (None, 15, 15, 256)   0           bn2a_branch2c[0][0]              
                                                                   bn2a_branch1[0][0]               
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 15, 15, 256)   0           add_2[0][0]                      
____________________________________________________________________________________________________
res2b_branch2a (Conv2D)          (None, 15, 15, 64)    16448       activation_7[0][0]               
____________________________________________________________________________________________________
bn2b_branch2a (BatchNormalizatio (None, 15, 15, 64)    256         res2b_branch2a[0][0]             
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 15, 15, 64)    0           bn2b_branch2a[0][0]              
____________________________________________________________________________________________________
res2b_branch2b (Conv2D)          (None, 15, 15, 64)    36928       activation_8[0][0]               
____________________________________________________________________________________________________
bn2b_branch2b (BatchNormalizatio (None, 15, 15, 64)    256         res2b_branch2b[0][0]             
____________________________________________________________________________________________________
activation_9 (Activation)        (None, 15, 15, 64)    0           bn2b_branch2b[0][0]              
____________________________________________________________________________________________________
res2b_branch2c (Conv2D)          (None, 15, 15, 256)   16640       activation_9[0][0]               
____________________________________________________________________________________________________
bn2b_branch2c (BatchNormalizatio (None, 15, 15, 256)   1024        res2b_branch2c[0][0]             
____________________________________________________________________________________________________
add_3 (Add)                      (None, 15, 15, 256)   0           bn2b_branch2c[0][0]              
                                                                   activation_7[0][0]               
____________________________________________________________________________________________________
activation_10 (Activation)       (None, 15, 15, 256)   0           add_3[0][0]                      
____________________________________________________________________________________________________
res2c_branch2a (Conv2D)          (None, 15, 15, 64)    16448       activation_10[0][0]              
____________________________________________________________________________________________________
bn2c_branch2a (BatchNormalizatio (None, 15, 15, 64)    256         res2c_branch2a[0][0]             
____________________________________________________________________________________________________
activation_11 (Activation)       (None, 15, 15, 64)    0           bn2c_branch2a[0][0]              
____________________________________________________________________________________________________
res2c_branch2b (Conv2D)          (None, 15, 15, 64)    36928       activation_11[0][0]              
____________________________________________________________________________________________________
bn2c_branch2b (BatchNormalizatio (None, 15, 15, 64)    256         res2c_branch2b[0][0]             
____________________________________________________________________________________________________
activation_12 (Activation)       (None, 15, 15, 64)    0           bn2c_branch2b[0][0]              
____________________________________________________________________________________________________
res2c_branch2c (Conv2D)          (None, 15, 15, 256)   16640       activation_12[0][0]              
____________________________________________________________________________________________________
bn2c_branch2c (BatchNormalizatio (None, 15, 15, 256)   1024        res2c_branch2c[0][0]             
____________________________________________________________________________________________________
add_4 (Add)                      (None, 15, 15, 256)   0           bn2c_branch2c[0][0]              
                                                                   activation_10[0][0]              
____________________________________________________________________________________________________
activation_13 (Activation)       (None, 15, 15, 256)   0           add_4[0][0]                      
____________________________________________________________________________________________________
res3a_branch2a (Conv2D)          (None, 8, 8, 128)     32896       activation_13[0][0]              
____________________________________________________________________________________________________
bn3a_branch2a (BatchNormalizatio (None, 8, 8, 128)     512         res3a_branch2a[0][0]             
____________________________________________________________________________________________________
activation_14 (Activation)       (None, 8, 8, 128)     0           bn3a_branch2a[0][0]              
____________________________________________________________________________________________________
res3a_branch2b (Conv2D)          (None, 8, 8, 128)     147584      activation_14[0][0]              
____________________________________________________________________________________________________
bn3a_branch2b (BatchNormalizatio (None, 8, 8, 128)     512         res3a_branch2b[0][0]             
____________________________________________________________________________________________________
activation_15 (Activation)       (None, 8, 8, 128)     0           bn3a_branch2b[0][0]              
____________________________________________________________________________________________________
res3a_branch2c (Conv2D)          (None, 8, 8, 512)     66048       activation_15[0][0]              
____________________________________________________________________________________________________
res3a_branch1 (Conv2D)           (None, 8, 8, 512)     131584      activation_13[0][0]              
____________________________________________________________________________________________________
bn3a_branch2c (BatchNormalizatio (None, 8, 8, 512)     2048        res3a_branch2c[0][0]             
____________________________________________________________________________________________________
bn3a_branch1 (BatchNormalization (None, 8, 8, 512)     2048        res3a_branch1[0][0]              
____________________________________________________________________________________________________
add_5 (Add)                      (None, 8, 8, 512)     0           bn3a_branch2c[0][0]              
                                                                   bn3a_branch1[0][0]               
____________________________________________________________________________________________________
activation_16 (Activation)       (None, 8, 8, 512)     0           add_5[0][0]                      
____________________________________________________________________________________________________
res3b_branch2a (Conv2D)          (None, 8, 8, 128)     65664       activation_16[0][0]              
____________________________________________________________________________________________________
bn3b_branch2a (BatchNormalizatio (None, 8, 8, 128)     512         res3b_branch2a[0][0]             
____________________________________________________________________________________________________
activation_17 (Activation)       (None, 8, 8, 128)     0           bn3b_branch2a[0][0]              
____________________________________________________________________________________________________
res3b_branch2b (Conv2D)          (None, 8, 8, 128)     147584      activation_17[0][0]              
____________________________________________________________________________________________________
bn3b_branch2b (BatchNormalizatio (None, 8, 8, 128)     512         res3b_branch2b[0][0]             
____________________________________________________________________________________________________
activation_18 (Activation)       (None, 8, 8, 128)     0           bn3b_branch2b[0][0]              
____________________________________________________________________________________________________
res3b_branch2c (Conv2D)          (None, 8, 8, 512)     66048       activation_18[0][0]              
____________________________________________________________________________________________________
bn3b_branch2c (BatchNormalizatio (None, 8, 8, 512)     2048        res3b_branch2c[0][0]             
____________________________________________________________________________________________________
add_6 (Add)                      (None, 8, 8, 512)     0           bn3b_branch2c[0][0]              
                                                                   activation_16[0][0]              
____________________________________________________________________________________________________
activation_19 (Activation)       (None, 8, 8, 512)     0           add_6[0][0]                      
____________________________________________________________________________________________________
res3c_branch2a (Conv2D)          (None, 8, 8, 128)     65664       activation_19[0][0]              
____________________________________________________________________________________________________
bn3c_branch2a (BatchNormalizatio (None, 8, 8, 128)     512         res3c_branch2a[0][0]             
____________________________________________________________________________________________________
activation_20 (Activation)       (None, 8, 8, 128)     0           bn3c_branch2a[0][0]              
____________________________________________________________________________________________________
res3c_branch2b (Conv2D)          (None, 8, 8, 128)     147584      activation_20[0][0]              
____________________________________________________________________________________________________
bn3c_branch2b (BatchNormalizatio (None, 8, 8, 128)     512         res3c_branch2b[0][0]             
____________________________________________________________________________________________________
activation_21 (Activation)       (None, 8, 8, 128)     0           bn3c_branch2b[0][0]              
____________________________________________________________________________________________________
res3c_branch2c (Conv2D)          (None, 8, 8, 512)     66048       activation_21[0][0]              
____________________________________________________________________________________________________
bn3c_branch2c (BatchNormalizatio (None, 8, 8, 512)     2048        res3c_branch2c[0][0]             
____________________________________________________________________________________________________
add_7 (Add)                      (None, 8, 8, 512)     0           bn3c_branch2c[0][0]              
                                                                   activation_19[0][0]              
____________________________________________________________________________________________________
activation_22 (Activation)       (None, 8, 8, 512)     0           add_7[0][0]                      
____________________________________________________________________________________________________
res3d_branch2a (Conv2D)          (None, 8, 8, 128)     65664       activation_22[0][0]              
____________________________________________________________________________________________________
bn3d_branch2a (BatchNormalizatio (None, 8, 8, 128)     512         res3d_branch2a[0][0]             
____________________________________________________________________________________________________
activation_23 (Activation)       (None, 8, 8, 128)     0           bn3d_branch2a[0][0]              
____________________________________________________________________________________________________
res3d_branch2b (Conv2D)          (None, 8, 8, 128)     147584      activation_23[0][0]              
____________________________________________________________________________________________________
bn3d_branch2b (BatchNormalizatio (None, 8, 8, 128)     512         res3d_branch2b[0][0]             
____________________________________________________________________________________________________
activation_24 (Activation)       (None, 8, 8, 128)     0           bn3d_branch2b[0][0]              
____________________________________________________________________________________________________
res3d_branch2c (Conv2D)          (None, 8, 8, 512)     66048       activation_24[0][0]              
____________________________________________________________________________________________________
bn3d_branch2c (BatchNormalizatio (None, 8, 8, 512)     2048        res3d_branch2c[0][0]             
____________________________________________________________________________________________________
add_8 (Add)                      (None, 8, 8, 512)     0           bn3d_branch2c[0][0]              
                                                                   activation_22[0][0]              
____________________________________________________________________________________________________
activation_25 (Activation)       (None, 8, 8, 512)     0           add_8[0][0]                      
____________________________________________________________________________________________________
res4a_branch2a (Conv2D)          (None, 4, 4, 256)     131328      activation_25[0][0]              
____________________________________________________________________________________________________
bn4a_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4a_branch2a[0][0]             
____________________________________________________________________________________________________
activation_26 (Activation)       (None, 4, 4, 256)     0           bn4a_branch2a[0][0]              
____________________________________________________________________________________________________
res4a_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_26[0][0]              
____________________________________________________________________________________________________
bn4a_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4a_branch2b[0][0]             
____________________________________________________________________________________________________
activation_27 (Activation)       (None, 4, 4, 256)     0           bn4a_branch2b[0][0]              
____________________________________________________________________________________________________
res4a_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_27[0][0]              
____________________________________________________________________________________________________
res4a_branch1 (Conv2D)           (None, 4, 4, 1024)    525312      activation_25[0][0]              
____________________________________________________________________________________________________
bn4a_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4a_branch2c[0][0]             
____________________________________________________________________________________________________
bn4a_branch1 (BatchNormalization (None, 4, 4, 1024)    4096        res4a_branch1[0][0]              
____________________________________________________________________________________________________
add_9 (Add)                      (None, 4, 4, 1024)    0           bn4a_branch2c[0][0]              
                                                                   bn4a_branch1[0][0]               
____________________________________________________________________________________________________
activation_28 (Activation)       (None, 4, 4, 1024)    0           add_9[0][0]                      
____________________________________________________________________________________________________
res4b_branch2a (Conv2D)          (None, 4, 4, 256)     262400      activation_28[0][0]              
____________________________________________________________________________________________________
bn4b_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4b_branch2a[0][0]             
____________________________________________________________________________________________________
activation_29 (Activation)       (None, 4, 4, 256)     0           bn4b_branch2a[0][0]              
____________________________________________________________________________________________________
res4b_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_29[0][0]              
____________________________________________________________________________________________________
bn4b_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4b_branch2b[0][0]             
____________________________________________________________________________________________________
activation_30 (Activation)       (None, 4, 4, 256)     0           bn4b_branch2b[0][0]              
____________________________________________________________________________________________________
res4b_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_30[0][0]              
____________________________________________________________________________________________________
bn4b_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4b_branch2c[0][0]             
____________________________________________________________________________________________________
add_10 (Add)                     (None, 4, 4, 1024)    0           bn4b_branch2c[0][0]              
                                                                   activation_28[0][0]              
____________________________________________________________________________________________________
activation_31 (Activation)       (None, 4, 4, 1024)    0           add_10[0][0]                     
____________________________________________________________________________________________________
res4c_branch2a (Conv2D)          (None, 4, 4, 256)     262400      activation_31[0][0]              
____________________________________________________________________________________________________
bn4c_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4c_branch2a[0][0]             
____________________________________________________________________________________________________
activation_32 (Activation)       (None, 4, 4, 256)     0           bn4c_branch2a[0][0]              
____________________________________________________________________________________________________
res4c_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_32[0][0]              
____________________________________________________________________________________________________
bn4c_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4c_branch2b[0][0]             
____________________________________________________________________________________________________
activation_33 (Activation)       (None, 4, 4, 256)     0           bn4c_branch2b[0][0]              
____________________________________________________________________________________________________
res4c_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_33[0][0]              
____________________________________________________________________________________________________
bn4c_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4c_branch2c[0][0]             
____________________________________________________________________________________________________
add_11 (Add)                     (None, 4, 4, 1024)    0           bn4c_branch2c[0][0]              
                                                                   activation_31[0][0]              
____________________________________________________________________________________________________
activation_34 (Activation)       (None, 4, 4, 1024)    0           add_11[0][0]                     
____________________________________________________________________________________________________
res4d_branch2a (Conv2D)          (None, 4, 4, 256)     262400      activation_34[0][0]              
____________________________________________________________________________________________________
bn4d_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4d_branch2a[0][0]             
____________________________________________________________________________________________________
activation_35 (Activation)       (None, 4, 4, 256)     0           bn4d_branch2a[0][0]              
____________________________________________________________________________________________________
res4d_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_35[0][0]              
____________________________________________________________________________________________________
bn4d_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4d_branch2b[0][0]             
____________________________________________________________________________________________________
activation_36 (Activation)       (None, 4, 4, 256)     0           bn4d_branch2b[0][0]              
____________________________________________________________________________________________________
res4d_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_36[0][0]              
____________________________________________________________________________________________________
bn4d_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4d_branch2c[0][0]             
____________________________________________________________________________________________________
add_12 (Add)                     (None, 4, 4, 1024)    0           bn4d_branch2c[0][0]              
                                                                   activation_34[0][0]              
____________________________________________________________________________________________________
activation_37 (Activation)       (None, 4, 4, 1024)    0           add_12[0][0]                     
____________________________________________________________________________________________________
res4e_branch2a (Conv2D)          (None, 4, 4, 256)     262400      activation_37[0][0]              
____________________________________________________________________________________________________
bn4e_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4e_branch2a[0][0]             
____________________________________________________________________________________________________
activation_38 (Activation)       (None, 4, 4, 256)     0           bn4e_branch2a[0][0]              
____________________________________________________________________________________________________
res4e_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_38[0][0]              
____________________________________________________________________________________________________
bn4e_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4e_branch2b[0][0]             
____________________________________________________________________________________________________
activation_39 (Activation)       (None, 4, 4, 256)     0           bn4e_branch2b[0][0]              
____________________________________________________________________________________________________
res4e_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_39[0][0]              
____________________________________________________________________________________________________
bn4e_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4e_branch2c[0][0]             
____________________________________________________________________________________________________
add_13 (Add)                     (None, 4, 4, 1024)    0           bn4e_branch2c[0][0]              
                                                                   activation_37[0][0]              
____________________________________________________________________________________________________
activation_40 (Activation)       (None, 4, 4, 1024)    0           add_13[0][0]                     
____________________________________________________________________________________________________
res4f_branch2a (Conv2D)          (None, 4, 4, 256)     262400      activation_40[0][0]              
____________________________________________________________________________________________________
bn4f_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4f_branch2a[0][0]             
____________________________________________________________________________________________________
activation_41 (Activation)       (None, 4, 4, 256)     0           bn4f_branch2a[0][0]              
____________________________________________________________________________________________________
res4f_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_41[0][0]              
____________________________________________________________________________________________________
bn4f_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4f_branch2b[0][0]             
____________________________________________________________________________________________________
activation_42 (Activation)       (None, 4, 4, 256)     0           bn4f_branch2b[0][0]              
____________________________________________________________________________________________________
res4f_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_42[0][0]              
____________________________________________________________________________________________________
bn4f_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4f_branch2c[0][0]             
____________________________________________________________________________________________________
add_14 (Add)                     (None, 4, 4, 1024)    0           bn4f_branch2c[0][0]              
                                                                   activation_40[0][0]              
____________________________________________________________________________________________________
activation_43 (Activation)       (None, 4, 4, 1024)    0           add_14[0][0]                     
____________________________________________________________________________________________________
res5a_branch2a (Conv2D)          (None, 2, 2, 512)     524800      activation_43[0][0]              
____________________________________________________________________________________________________
bn5a_branch2a (BatchNormalizatio (None, 2, 2, 512)     2048        res5a_branch2a[0][0]             
____________________________________________________________________________________________________
activation_44 (Activation)       (None, 2, 2, 512)     0           bn5a_branch2a[0][0]              
____________________________________________________________________________________________________
res5a_branch2b (Conv2D)          (None, 2, 2, 512)     2359808     activation_44[0][0]              
____________________________________________________________________________________________________
bn5a_branch2b (BatchNormalizatio (None, 2, 2, 512)     2048        res5a_branch2b[0][0]             
____________________________________________________________________________________________________
activation_45 (Activation)       (None, 2, 2, 512)     0           bn5a_branch2b[0][0]              
____________________________________________________________________________________________________
res5a_branch2c (Conv2D)          (None, 2, 2, 2048)    1050624     activation_45[0][0]              
____________________________________________________________________________________________________
res5a_branch1 (Conv2D)           (None, 2, 2, 2048)    2099200     activation_43[0][0]              
____________________________________________________________________________________________________
bn5a_branch2c (BatchNormalizatio (None, 2, 2, 2048)    8192        res5a_branch2c[0][0]             
____________________________________________________________________________________________________
bn5a_branch1 (BatchNormalization (None, 2, 2, 2048)    8192        res5a_branch1[0][0]              
____________________________________________________________________________________________________
add_15 (Add)                     (None, 2, 2, 2048)    0           bn5a_branch2c[0][0]              
                                                                   bn5a_branch1[0][0]               
____________________________________________________________________________________________________
activation_46 (Activation)       (None, 2, 2, 2048)    0           add_15[0][0]                     
____________________________________________________________________________________________________
res5b_branch2a (Conv2D)          (None, 2, 2, 512)     1049088     activation_46[0][0]              
____________________________________________________________________________________________________
bn5b_branch2a (BatchNormalizatio (None, 2, 2, 512)     2048        res5b_branch2a[0][0]             
____________________________________________________________________________________________________
activation_47 (Activation)       (None, 2, 2, 512)     0           bn5b_branch2a[0][0]              
____________________________________________________________________________________________________
res5b_branch2b (Conv2D)          (None, 2, 2, 512)     2359808     activation_47[0][0]              
____________________________________________________________________________________________________
bn5b_branch2b (BatchNormalizatio (None, 2, 2, 512)     2048        res5b_branch2b[0][0]             
____________________________________________________________________________________________________
activation_48 (Activation)       (None, 2, 2, 512)     0           bn5b_branch2b[0][0]              
____________________________________________________________________________________________________
res5b_branch2c (Conv2D)          (None, 2, 2, 2048)    1050624     activation_48[0][0]              
____________________________________________________________________________________________________
bn5b_branch2c (BatchNormalizatio (None, 2, 2, 2048)    8192        res5b_branch2c[0][0]             
____________________________________________________________________________________________________
add_16 (Add)                     (None, 2, 2, 2048)    0           bn5b_branch2c[0][0]              
                                                                   activation_46[0][0]              
____________________________________________________________________________________________________
activation_49 (Activation)       (None, 2, 2, 2048)    0           add_16[0][0]                     
____________________________________________________________________________________________________
res5c_branch2a (Conv2D)          (None, 2, 2, 512)     1049088     activation_49[0][0]              
____________________________________________________________________________________________________
bn5c_branch2a (BatchNormalizatio (None, 2, 2, 512)     2048        res5c_branch2a[0][0]             
____________________________________________________________________________________________________
activation_50 (Activation)       (None, 2, 2, 512)     0           bn5c_branch2a[0][0]              
____________________________________________________________________________________________________
res5c_branch2b (Conv2D)          (None, 2, 2, 512)     2359808     activation_50[0][0]              
____________________________________________________________________________________________________
bn5c_branch2b (BatchNormalizatio (None, 2, 2, 512)     2048        res5c_branch2b[0][0]             
____________________________________________________________________________________________________
activation_51 (Activation)       (None, 2, 2, 512)     0           bn5c_branch2b[0][0]              
____________________________________________________________________________________________________
res5c_branch2c (Conv2D)          (None, 2, 2, 2048)    1050624     activation_51[0][0]              
____________________________________________________________________________________________________
bn5c_branch2c (BatchNormalizatio (None, 2, 2, 2048)    8192        res5c_branch2c[0][0]             
____________________________________________________________________________________________________
add_17 (Add)                     (None, 2, 2, 2048)    0           bn5c_branch2c[0][0]              
                                                                   activation_49[0][0]              
____________________________________________________________________________________________________
activation_52 (Activation)       (None, 2, 2, 2048)    0           add_17[0][0]                     
____________________________________________________________________________________________________
avg_pool (AveragePooling2D)      (None, 1, 1, 2048)    0           activation_52[0][0]              
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2048)          0           avg_pool[0][0]                   
____________________________________________________________________________________________________
fc6 (Dense)                      (None, 6)             12294       flatten_1[0][0]                  
____________________________________________________________________________________________________
Total params: 23,600,006
Trainable params: 23,546,886
Non-trainable params: 53,120

---

# *Note* : It was trained on CPU so I trained it for 2 epochs only. I loaded a pretrained model which was already trained on GPU which saves time. This notebook was to give intution about how ResNET works.

---

# Reference

---

This notebook presents the ResNet algorithm due to He et al. (2015). The implementation here also took significant inspiration and follows the structure given in the GitHub repository of Francois Chollet:

 - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - Deep Residual Learning for Image Recognition (2015)
 - Francois Chollet's GitHub repository: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
