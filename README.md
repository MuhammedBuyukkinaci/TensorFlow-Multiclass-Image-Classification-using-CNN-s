# TensorFlow-Multiclass-Image-Classification-using-CNN-s
This is a multiclass image classification project using Convolutional Neural Networks and TensorFlow API (no Keras) on Python.

[Read all story in Turkish](https://medium.com/@mubuyuk51/tensorflow-ile-%C3%A7ok-s%C4%B1n%C4%B1fl%C4%B1-multi-class-resim-s%C4%B1n%C4%B1fland%C4%B1rma-f56c3605aff6).
# Dependencies

```pip install -r requirements.txt```

or

```pip3 install -r requirements.txt```

# Training
Training on GPU:

```python multiclass_classification_gpu.py ```

Training on CPU:

```python multiclass_classification_cpu.py ```

# Notebook
Download .ipynb file from [here](https://github.com/MuhammedBuyukkinaci/My-Jupyter-Files-1/blob/master/Multiclass_CNN.ipynb) and run

```jupyter lab ``` or ```jupyter notebook ```

# Data
No MNIST or CIFAR-10. 

This is a repository containing datasets of 5200 training images of 4 classes and 1267 testing images.No problematic image.

Download .rar extension version from [here](
https://www.dropbox.com/s/30n7ge8dxhs3doi/multiclass_datasets.rar?dl=0) or .zip extension version from [here](
https://www.dropbox.com/s/20jkiactn0k5sss/multiclass_datasets_zip.zip?dl=0). It is 67 MB.

Extract files from multiclass_datasets.rar. Then put it in TensorFlow-Multiclass-Image-Classification-using-CNN-s folder.
train_data_bi.npy is containing training photos with labels.

test_data_bi.npy is containing 1267 testing photos with labels.

Classes are chair & kitchen & knife & saucepan.

Classes are equal(1300 glass - 1300 kitchen - 1300 knife- 1300 saucepan) on training data. 

# CPU or GPU
I trained on GTX 1050. 1 epoch lasted 45 seconds approximately.

If you are using CPU, which I do not recommend, change the lines below:
```
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
with tf.Session(config=config) as sess:
```
to
```
with tf.Session() as sess:
```
# Architecture

AlexNet is used as architecture. 5 convolution layers and 3 Fully Connected Layers with 0.5 Dropout Ratio. 60 million Parameters.
![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Image-Classification-Convolutional-Neural-Networks/blob/master/alexnet_architecture.png) 

# Results
Accuracy score reached 89% on CV after 30 epochs. Test accuracy is around 88%.
![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Multiclass-Image-Classification-using-CNN-s/blob/master/mc_results.png)

# Predictions
Predictions for first 64 testing images are below. Titles are  the predictions of our Model.

![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Multiclass-Image-Classification-using-CNN-s/blob/master/mc_preds.png)
