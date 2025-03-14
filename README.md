# ECE-GY-7123-Deep-Learning-Project-1

This project includes all files and implementations for Project-1 where we had to build ResNet model on CIFAR-10 datasets with less than 5M parameters.

The final configurations of our model are:
- Number of layers: 3
- Blocks in each layer: [4,5,3]
- Input channels to each layer: [64,128,256]
- Filter Size: 3x3
- Kernel Size: 1x1
- Average pooling size: 8
With the following hyperparameters:
- Batch Size: 128
- Optimizer: Lookahead + SGD
- Learning Rate: 0.1
- Momentum: 0.9
- Weight Decay: 0.0005
- Annealing Cosine: 200 epochs
- Learning Rate Decay: by 0.1
- Total Epochs: 200
- Lookahead Alpha: 0.5
  
Resulting in train accuracy of 97.65, validation accuracy
of 96.95 and test accuracy of 87.3

