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

The second CONV2D has  F*F2*  filters of shape  *(f,f)*  and a stride of *(1,1)*. Its padding is "same" and its name should be conv_name_base + '2b'. Use 0 as the seed for the random initialization.
The second BatchNorm is normalizing the 'channels' axis. Its name should be bn_name_base + '2b'.
Then apply the ReLU activation function. This has no name and no hyperparameters.
Third component of main path:

The third CONV2D has  *F3*  filters of shape *(1,1)* and a stride of *(1,1)*. Its padding is "valid" and its name should be conv_name_base + '2c'. Use 0 as the seed for the random initialization.
The third BatchNorm is normalizing the 'channels' axis. Its name should be bn_name_base + '2c'.
Note that there is no ReLU activation function in this component.
Final step:

The X_shortcut and the output from the 3rd layer X are added together.
Then apply the ReLU activation function. This has no name and no hyperparameters.
