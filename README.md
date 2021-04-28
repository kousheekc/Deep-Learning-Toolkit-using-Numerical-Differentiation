# Deep Learning Toolkit from Scratch Using Numerical Differentiation
Toolkit for building and training deep learning models. Built from scratch.

To train your model simply modify the train.py script using your custom hyperparemeters.

## Requirements:
* The only requirement for this project is **Numpy**

If you would like to use the plotting functionality you will need
* **Seaborn**
* **Tensorboard**
* **Pandas**

## Overview of the training process
* Compute the estimated output by forward propagation
* Compute loss using the estimated output and the desired output
* Compute the gradient tensor
* Update the parameters by taking a small step in the opposite direction of the gradient tensor

## Gradient Calculation
To calculate the gradient I have implemented 6 methods to numerically compute the derivative of a function at a point with respect to a variable.
* Forward difference
* Backward difference
* Three point endpoint 
* Three point midpoint
* Five point endpoint
* Five point midpoint

## Results
The following results show the training process while using the Iris flower classification dataset. The training was done using all 6 methods of differentiation and was finally compared to a model built and trained on Pytorch. 

As you can see from the results the differentiation formula that was used did not significantly affect the outcome for this specific dataset. In comparison to a model trained in Pytorch, both performed with similar accuracies on the training and testing datasets, however the Pytorch model converged within 20 epochs while the other model took 100 epochs to converge. In addition, the time taken to train one epoch was much shorter on the Pytorch model.


