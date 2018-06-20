# Disentangling Multiple Conditional Inputs in GANs

This is a Tensorflow implementation of our paper "Disentangling Multiple Conditional Inputs in GANs". It is tested with Tensorflow 1.8 (Python 3.6). We modified the code from [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://github.com/tkarras/progressive_growing_of_gans) and used it without progressive growing.

### Overview

The process of fashion design requires extensive amount of knowledge in creation and production of garments. Therefore, a well-structured and agile design process is crucial. A machine-assisted design approach that combines human experience with deep learning can help designers to rapidly visualize an original garment and can save time on design iteration cycles. 

In this paper, we propose a method that disentangles the effects of multiple input conditions in Generative Adversarial Networks (GANs). In particular, we demonstrate our method in controlling color, texture, and shape of a generated garment image for computer-aided fashion design. To disentangle the effect of input attributes, we customize conditional GANs with consistency loss functions. For more information, check out our paper.

### Usage

Before running the code, please check `dataset_tool.py` and `dataset.py` files and make sure that you modify them for your purposes. Creating and loading the dataset will depend on the task. After that, you can adjust the training parameters by modifying the `config.py` file.

After creating your datasets and adjusting your hyperparameters, the training can be performed by running the following code:

```
python train.py
```

### Examples

#### Color Control
![](examples/color-control.png)

#### Texture Control
![](examples/texture-control.png)

#### Shape Control
![](examples/shape-control.png)
