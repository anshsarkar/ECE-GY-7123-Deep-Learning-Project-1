# ECE-GY-7123-Deep-Learning-Project-1

**Team GradeIsAllYouNeed**  
- Ansh Sarkar (as20363@nyu.edu)
- Princy Doshi (pd2672@nyu.edu)
- Simran Kucheria (sk11645@nyu.edu)
---

## Project Overview

The goal of this project was to train and test a ResNet model on the CIFAR-10 dataset with **less than 5 million parameters**. The constraints included no pre-trained models allowed. Our best model achieved a **test accuracy of 96.95%** with **4,992,586 parameters**.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/anshsarkar/ECE-GY-7123-Deep-Learning-Project-1)

---

## Model Architecture

We implemented a variant of the ResNet architecture using **BasicBlocks**, which consist of two convolutional layers with batch normalization and ReLU activation. The final model configuration is as follows:

- **Number of layers**: 3
- **Blocks in each layer**: [4, 5, 3]
- **Input channels to each layer**: [64, 128, 256]
- **Filter Size**: 3x3
- **Kernel Size**: 1x1
- **Average pooling size**: 8

### Hyperparameters:
- **Batch Size**: 128
- **Optimizer**: Lookahead + SGD
- **Learning Rate**: 0.1
- **Momentum**: 0.9
- **Weight Decay**: 0.0005
- **Total Epochs**: 200
- **Lookahead Alpha**: 0.5

---

## Methodology

### Data Augmentation
We experimented with various data augmentation techniques to improve model performance:
- **Random Crop** and **Random Horizontal Flip** were used to introduce spatial variability.
- **Cutout** was employed to mask random square regions, forcing the model to focus on broader contextual patterns.
- **Mixup** was tested but ultimately not included in the final implementation as it did not yield significant improvements.

### Optimizers and Regularization
- **Lookahead Optimizer**: Used to stabilize training and speed up convergence.
- **Cosine Annealing LR**: Dynamically adjusted the learning rate over epochs.
- **L2 Regularization**: Applied through weight decay to prevent overfitting.
- **Label Smoothing**: Improved model generalization by creating "soft" labels.

---

## Results

Our final model achieved:
- **Training Accuracy**: 97.65%
- **Test Accuracy**: 96.95%

The model began to converge around **200 epochs**, with the accuracy plateauing at this point.

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- DeVries, T., & Taylor, G. W. (2017). [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552).
- Zhang, M. R., Lucas, J., Hinton, G., & Ba, J. (2019). [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610).

---

## Figures

- **Training Loss vs Epochs**: ![Training Loss](training_loss.png)
- **Testing Loss vs Epochs**: ![Testing Loss](testing_loss.png)
- **Accuracy vs Epochs**: ![Accuracy](accuracy.png)
