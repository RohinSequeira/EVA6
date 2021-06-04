# EVA6_Session5_incremental_models

## Objective

We are trying to build models to identify digits in the MNIST dataset in an iterative manner so that we can paint a picture of how we went from laying the foundation to adding different layers to the final product. Our end result has to satisfy the following conditions:

1. Test accuracy of 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
2. Less than or equal to 15 Epochs
3. Less than 10000 Parameters (additional points for doing this in less than 8000 pts)

Target needs to be achieved in 3 steps or more, so as to explain the process properly. _**Target**_ , **_Result_** and **_Analysis_** provided for each iteration.

Click on the title of each step to access the notebook associated with the experiment.



## [Step 0: Creating a basic foundation to build upon]()

We have implemented our learning to implement all the basic concepts that help form a good foundation of a deep learning model like Batch Normalization, Max Pooling, GAP and Fully Convolutional 1x1 layers along with Convolutional layers.

**_Note_:** The various components were added based on experience so as to reduce the number of experiement that will be needed to perform and record :).

No. of Parameters | Train Accuracy | Test Accuracy|
------------------|----------------|--------------|
|     8152        |     99.09      |  98.99 (14th Epoch), 98.22 (15th Epoch)


### _Summary_
Although the training accuracy looks good, Model is overfitting. In the 15th epoch, we have observed over 0.8% difference between the train and test accuracies, suggesting overfitting.

We think we have achieved our goal of creating a good base model. Adding regularization in the form of dropout might help reduce the overfitting, which we will try in the next step.


## [Step 1: Apply regularization to reduce overfitting]()

Apply regularization to bring overfitting within acceptable range.  

No. of Parameters | Train Accuracy | Test Accuracy|
------------------|----------------|--------------|
|     8152        |     99.02      |  98.91 (13th Epoch), 98.66 (15th Epoch)


### _Summary_

Dropout Regularization(6%) has been added at the end of first and second convolution blocks (randomly), which helped reduce overfitting as expected. The model still needs to learn more efficiently without overfitting and to help the model identify the test images, for which we can try Data Augmentation.

## [Step 2: Using Data Augmentation]()

Use Data Augmentation - Random Rotation of Training Images to help model identify images better while testing and also hopefully help reduce overfitting.

No. of Parameters | Train Accuracy | Test Accuracy|
------------------|----------------|--------------|
|     8152        |     98.77      |  98.87 (14th Epoch), 98.35 (15th Epoch)


### _Summary_

Data Augmentation has helped the model reduce overfitting. But, overall accuracy has reduced, which is acceptable for now.

## [Step 3: Using Dynamic Learning Rate]()

Use LR scheduler - ReduceLROnPlateau to dynamically update the Learning Rate. This should hopefully help us cross the accuracy of both testing and training beyond 99.*%.

No. of Parameters | Train Accuracy | Test Accuracy|
------------------|----------------|--------------|
|     8152        |     99.44      |  98.41 (14th Epoch), 98.37 (15th Epoch)

### _Summary_

Bringing in the LR scheduler has definitely helped in inceasing the training & test accuracy, and we even achieved our target of 99.41% in the 13th & 14th epochs, but the accuracy dropped down to 99.37% in the 15th epoch.

## [Step 4: Playing with the Optimizer]()

Even after successive runs, the model failed to hold on after reaching the target of 99.4% test accuracy.
Replace optimizer to see if Adam can help sustain the accuracy obtained.

No. of Parameters | Train Accuracy | Test Accuracy|
------------------|----------------|--------------|
|     8152        |     99.45      |  99.53 (14th Epoch), 99.49 (15th Epoch)

### _Summary_

The usage of Adam optimizer in place of SGD has proved useful not only in achieving the target of 99.4%, but also pushed it a bit closer to 99.5% consistently over the last 3 epochs (13,14 & 15).  
This is where we stop our experiment.


### Collaborators

Abhiram Gurijala  
Arijit Ganguly  
Rohin Sequeira  

