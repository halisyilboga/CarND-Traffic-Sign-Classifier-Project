### Project: Build a Traffic Sign Recognition

In this project, convolutional neural networks (CNN) is applied to  German Traffic Sign Dataset and trained modified version of LeNet network to classify them. After the model was trained, the model was tested on  7 images which founded from internet.


[//]: # (Image References)

[image1]: result/sampleImages.png "sampleImages "
[image2]: result/sampleImages_augmented.png "sampleImages augmented image"

[image4]: result/sample_new.png "sample new"
[image5]: result/sample_new_result.png "sample_new_result"

[image6]: result/visualize.png "visualize layers"

[image7]: result/histogramOfDataset.png "histogramOfDataset"
[image8]: result/sample_new_result1.png "sample_new_result"
[image9]: result/sample_new_result2.png "sample_new_result"
[image10]: result/sample_new_result3.png "sample_new_result"
[image11]: result/sample_new_result4.png "sample_new_result"
[image12]: result/sample_new_result5.png "sample_new_result"
[image13]: result/sample_new_result7.png "sample_new_result"
---

sample images are belof for German image dataset
 ![alt text][image1]

dataset class distrubition  is below
 ![alt text][image7]


By the help of keras image generate class images are augmented. Random augmenteation is used on images and the size of the dataset is increased by the 3 times of real database.  For avoiding different result a method written to save the all dataset to folder and read from there when needed.

```
X_train_ex = []
y_train_ex = []

def create_data(X_train_ex, y_train_ex, n=3):
    for i in range(0, X_train.shape[0]):
        for j in range(0,n): # Generate n new random transfer images.
            X_train_ex.append(datagen.random_transform(X_train[i,:], seed=1))
            y_train_ex.append(y_train[i])

try:
    with open("train_augmented.pkl", "rb") as f:
            X_train_ex, y_train_ex = pickle.load(f)
            X_train_ex = np.array(X_train_ex)
            y_train_ex = np.array(y_train_ex)

except OSError as err:
    print("train_augment")    

    create_data(X_train_ex,y_train_ex)
    X_train_ex = np.concatenate( (X_train_ex, X_train))
    y_train_ex = np.concatenate( (y_train_ex, y_train))

    with open("train_augmented.pkl", 'wb') as f:
        pickle.dump([X_train_ex,y_train_ex], file=f)

```
sample augmented images are below/
![alt text][image2]
and keras ImageDataGenerator class support

I use keras data generate methods to generate new images on existing ones. According to documentation It can  augment different images according to fallowing features.

```
'theta': Float. Rotation angle in degrees.
'tx': Float. Shift in the x direction.
'ty': Float. Shift in the y direction.
'shear': Float. Shear angle in degrees.
'zx': Float. Zoom in the x direction.
'zy': Float. Zoom in the y direction.
'flip_horizontal': Boolean. Horizontal flip.
'flip_vertical': Boolean. Vertical flip.
'channel_shift_intencity': Float. Channel shift intensity.
'brightness': Float. Brightness shift intensity.
```


---
### Architecture
**Layer 1:**  Convolutional. The output shape should be 28x28x24.

**Activation** Relu activation function is used.

**Pooling** The output shape should be 14x14x24


**Layer 2:** Convolutional. The output shape should be 10x10x32.

**Activation** Relu activation function is used.

**Pooling** The output shape should be 5x5x32.

**Flatten**  Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.

**Layer 3:** Fully Connected. This have 800 outputs.

**Activation** Relu activation function is used.
**dropout** keep_rate 0.8

**Layer 4:** Fully Connected. This have 400 outputs.

**Activation** Relu activation function is used.
**dropout** keep_rate 0.8

**Layer 5:** Fully Connected. This have 240 outputs.

**Activation** Relu activation function is used.
**dropout** keep_rate 0.8

**Layer 6:** Fully Connected. This have 120 outputs.

**Activation** Relu activation function is used.
**dropout** keep_rate 0.8

**Layer 7:** Fully Connected (Logits). This have 43 outputs.
Which after each fully connected network dropout out layer is used for regularization.



---

Training is done with Adamoptimizer and different learning rate is checked for optimum result. Extended version of the dataset is used for swowing result of generating augmented images. It can expand the capability of the network in term of scale, rotation etc. Approximately 95 validation and test accuracy is achieved.


---

7 sample image is downloaded from internet and labelled accordingly. Below sample images.

![alt text][image4]

and network classification result can be seen below.



![alt text][image8]

![alt text][image5]

![alt text][image9]

![alt text][image10]

![alt text][image11]

![alt text][image12]

![alt text][image13]


and visualization result can be seen below conv layer one and conv layer 2 is used for visualization.


![alt text][image6]
