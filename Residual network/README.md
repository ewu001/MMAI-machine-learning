Playground for Residual Network following the architecture of ResNet50

Residual network runs on skip connections to pass previous layer's activation directly into later layer before its activation step as a residual with the purpose of fighting vanishing gradient in deep convolution networks. 

By doing do, gradients can be directly backpropogated to earlier layers. 

After compile,  ResNet is trained to 2 separate data sets, one is MNIST with 70,000 images, the other is hand digit sign data set with 10,80 images.
