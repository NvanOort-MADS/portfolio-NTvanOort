# Convolutional Neural Network
This report outlines the experiments I conducted and results that were achieved in trying to optimize a Convolutional Neural Network (CNN) on the Fashion-MNIST dataset. Furthermore, I will reflect on the chosen experiments, how I thought they would impact the model and what actually happened.

## Experiments
Just like the Fully Connected Neural Network hypertuning experiments, I decided to start finetuning this CNN by establishing a baseline. The baseline was created by running the notebook on MLflow from the lessons. This resulted in three separate models by leveraging Hyperopt and those results are outlined in the next paragraph.
After the baseline, I will continue with the structure for manual hypertuning - that was provided during the lessons - by discovering the optimal architecture in terms of Width & Depth. Followed by applying regularization techniques if necessaray and closing of with the best suited learning speed in terms op optimization.

### Baseline (Hyperopt - 3 runs)
As mentioned, the baseline was established using Hyperopt with the following search space:

```Python
search_space = {
    'filters' : scope.int(hp.quniform('filters', 16, 128, 8)),
    'kernel_size' : scope.int(hp.quniform('kernel_size', 2, 5, 1)),
    'num_layers' : scope.int(hp.quniform('num_layers', 1, 10, 1)),
}
```
**Results**
- Run 1: 88 filters, 4 layers → test loss 0.5893, accuracy 79.88% (best baseline)
- Run 2: 96 filters, 4 layers
- Run 3: 64 filters, 4 layers
- **Observation:** Train loss (0.6459) > test loss (0.5893) → underfitting

The baseline tells us that the number of filters falls outside the numerical range for factors of 2 to deliver the optimal number of filters.

To understand the models capabilities over a longer period of time, without changing the models capacity, I tried increasing the epochs from 3 -> 20. Additionally, I ran the 20 epochs model on the underfitting baseline with 64 filters to even better understand how extra training data would influence a model with low capacity. Additional training data enabled the model to already reach **86%** accuracy, whilst training en test scores getting more alined at **33%** for train and **37%** for test loss.

![Baseline (88 filters & 3 epochs) vs. Extended training (64 & 20)](./img/Loss_test_baseline.svg)
*Test loss: 0.5893 vs. Test loss: 0.3766*

![Baseline (88 filters & 3 epochs) vs. Extended training (64 & 20)](./img/Loss_train_baseline.svg)
*Train loss: 0.6459 vs. Train loss: 0.3332*

![Baseline (88 filters & 3 epochs) vs. Extended training (64 & 20)](./img/metric_Accuracy_baseline.svg)
*Accuracy: 0.7988 vs. Accuracy: 0.8627*

### Experiment 2: Architecture (Width & Depth)
Before experimenting with the models capacity, I wanted to understand what it was working with during the model with 64 filters and 20 epochs training duration. Therefore, I went through disecting the nn.Sequencial class steps and calculated the parameters inbetween each layer.

This gave me insight into the effect the convolutions and pooling operations had on the input data. At the end of the four layers I saw that the images were only 2x2 in size remaining. This was the first time in hypertuning Neural Networks that I gained an intuïtive feeling by thinking that 2x2 is probably a too small image size for the model to perform well on. Therefore, I wanted to try removing the last layer for the fifth experiment (second after de first three baselines). This enabled the model to work width an image size of 6x6 before flattening fo the Fully Connected (FC) layer.

After running this with 64 filters and 20 epochs as well, the model immediately improved significantly reaching **90%** accuracy, whilst also improving on the loss function for both train and test! This confirmed that my intuition was correct and that gave a good feeling towards becoming better at understanding networks and improving in manual hypertuning.

![Architecture (64 filters & 20 epochs)](./img/Loss_test_architecture.svg)
*Test loss: 0.2659*

![Architecture (64 filters & 20 epochs)](./img/Loss_train_architecture.svg)
*Train loss: 0.2354*

![Architecture (64 filters & 20 epochs)](./img/metric_Accuracy_architecture.svg)
*Accuracy: 0.9038*

### Expertiment 3: Regularization
Even though the models capacity is improved, it does not show any signs of overfitting and thus regularization techniques are unnecessary. However, since techniques like Dropout and Batch Normalization are new to me, I am curious to see how these are applied in action and maybe some difference will be shown. Therefore I researched where to and when to use Batch Norm and Dropout. This thaught me that there are studies mentioning that using both techniques causes disharmony as Dropout disturbs the batch statistics used by Batch Norm. However, this is not entirely clear and thus some studies counter this. Anyhow, I followed some use cases and tried the following approach:

1. Add Batch Norm before activation function (as per the original paper)
2. Add Dropout in the Fully Connected layer (if the models still overfits after step 1)
3. Add Dropout in the CNN after activation function

As the task is not too complex, regularisation techniques did not much for the loss and accuracy scores but it is gooed to know how to apply these techniques in the future for when the task requires it.

### Experiment 4: Optimization
Optimization still feels difficult to fully grasp and understand what setting best fit each algorithm and what algorithm best fits the architecture. Therefore, I did some research into what optimization algorithm to choose for a CNN architecture as it differs a lot into how parameters are calculated and created compared to previous exercise about the FCNN. 
I learned that adaptive optimization techniques suite the CNN architecture better as weight are shared over the kernels and each layer learns different patters (in terms of Fashion-MNIST where the first layer detects lines, the second curves, the third shapes, the fourth object, etc.), and therefore require different learning steps.

The best model so far was tested on the baseline optimizer, which was Adam. So, I wanted to see if another technique would find better results and through the research I learned about AdamW, which is an improvement on Adam where the weight decay is applied on the weights directly instead of via de gradient. This should come at the benefit of applying consistency over the parameters instead of changing with the gradient size.

I tested some variations of AdamW in terms of LR and Weigth Decay but unfortunately didn't improve the accuracy and stuck around 89 - 90% accuracy. Probably because of the stronger regularization that AdamW brings to the table and the model already had enough regularization thanks to the Dropout/ Batch Norm layers in the architecture.

### Final Model
The biggest improvement was eliminating a layer in the architecture to enable the model to learn more spatial information and therefore reach an **Accuracy of 90%**

Find the [notebook](./mlflow.ipynb) and the [instructions](./instructions.md)

[Go back to Homepage](../README.md)