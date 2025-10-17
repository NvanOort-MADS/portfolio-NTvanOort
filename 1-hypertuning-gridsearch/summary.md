# Fully Connected (FC) Neural Network
This report outlines the experiments I conducted and results that were achieved in trying to optimize a fully connected neural network on the Fashion-MNIST dataset. Furthermore, I will reflect on the chosen experiments, how I thought they would impact the model and what actually happened.

## Quick Recap
This exercise was supposed to be completed wihtin a week, but I took five additional weeks to complete it. Maybe the exercise is not even complete, but I accepted the results for as far as I got (which can probably be better). I think I have restarted this exercise three of four times and spending hours or even days trying to understand which combinations go together and how they relate to changing some hyperparameter to one direction and how that influences another hyperparameter, which order to try different hyperparameters, etc.

From a conceptual point I'd like to believe I understand when a model is too simple in thus underfits or tends to overfit and which hyperparameters you can choose to affect those symptoms. For example, increasing the number of units to prevent underfitting or meddling with the batchsize, epochs and learning rate to find the best generalization settings though optimization. 
But the main reason I was stuck on this exercise for so long was that I wanted to complete this before moving on to other architectures which would introduce even more parameter options. I found out that I was struggling the most with understanding how deep and how wide the network should be. I just wasn't getting a feeling for it or intuitively understanding what should work and what not. Because there are so many possibilities, from starting with a hidden layer bigger than the input features to starting smaller and expanding the units with each layer, or even keep the same number of unit but making the network deeper. 

After trying a lot of experiments and doing lot's of reading on hypertuning neural networks, I've come to understand that there is not really a rule of thumb, a calculation or something similar. Only some heuristics based on the input features and output, maybe the parameters, and that leaves hypertuning relatively abstract which I find difficult to grasp and therefore difficult to gets a sense of what is suitable in the endless landscape of hypertuning.

## Experiments
From the reading I did I found that many mention the learning rate as one of the most important hyperparameters, and thus is often the first one that's tuned. However, during the hypertuning lessen we were provided with a structure in which out teacher usually performs manual hypertuning. This structure looks as follows, and is also what I decided to use for these experiments as a guide:

1. **Architecture**
2. **Width & Depth**
3. **Regularization**
4. **Optimization**

I created eight models in the process of finding the optimal hyperparameter settings to find the best generalizing model. One of which was the model as set up in the notebook. This seemed as a nice baseline consisting of:

- Input layer (28 * 28 = 784 features)
- Hidden layer (256 units)
- Hidden layer (256 units)
- Output layer (10 classes)

With inbetween some non-linear layers with a ReLU actionvation function.

### Architecture + Width & Depth
Since this exercise is focused on a fully connected neural network, the choice for the best suiting architecture became obsolete. Therefore, the first tuning was done by playing around with the width and depth. I started by decreasing the width of the second hidden layer to get closer the output layer and thus require the model to handle compression. This resulted in an accuracy score of **.805** which was a bit lower compared to the baseline model with a accuracy score of **.817** indicating that this model was to simple for the task at hand.

Since increasing the number of units does not benefit the model when it's expected to output 10 classes, I wanted to test an additional hidden layer and tried another layer with 256 units. Unfortunately, the additional layer scored almost equal to the baseline, meaning that two hidden layers of enough units were already able to model to data well enough. 

These models were still only trained on 3 epochs with a batchsize of 64. Given that the accuracy didn't increase too much, I wanted to let the model process more examples thus increasing the number of epochs to 10. Showing the model the data more times enabled it to learn more and this resulted in an increase of .03 in accuracy to **.0847**

### Optimization
For this exercise it was not yet necessary to apply regulatization, so I refrained from that area to limit the number of possibilities. With the depht and width of the model not improving much which changes in different directions, I went back the baseline model of two hidden layers - with 256 units each - and see how the learning rate and optimization algorithm would influence the models ability to learn.

The baseline used Adam as the algorithm with a lr of 1e-3. From the theorie I learned that often the ranges are tested with a factor of 10, so I wanted to see what a lr of 1e-2 and 1e-4 would do. With 10 epochs the higher learning rate resulted in overshooting by jumping out of a minima and the lower lr showed a steadier results without getting stuck or overshooting in the available epochs.
Lastly, just to verify I chose one extra smaller learning rate of 1e-5, but this appeared to equal the 1e-4 lr and took longer per epoch to train. This is ofcourse logical as smaller steps take longer to train, but given that there was barely any difference in accuracy it was better to stick with 1e-4.

Finally, I wanted to understand if another algoritm would make a difference and therefore I went with Stochastic Gradient Descent (SGD), to see what a less advanced algorithm - compared to Adam which in essence is an improvement upon SGD - would let the model achieve in terms of generalization. SGD was already able to achieve **.862** on accuracy with the same lr as the baseline model with Adam, which is **0.02** higher compared to the baseline Adam 1e-3.After that, I tested the same as for Adam by trying one more lr and decided to go with 1e-2 to see what a higher step would find.
Although the accuracy didn't improve, the train time per epoch went lower.

### Final Model
As the goal is to find the simplest model that generalizes best to new data, I ended with the following model:

- Two hidden layers with 256 units
- Ten epochs
- SGD with 1e-2 learning rate

**Accuracy = 0.8669**

## Closing Remarks
Since it took me a long time to wrap up this exercise, I did not really follow the rules in terms of setting up a hypothesis and performing experiment based on that. I want to pick this up in the following exercise to see if all the reading and tuning helped in any way.

Find the [notebook](./notebook.ipynb) and the [instructions](./instructions.md)

[Go back to Homepage](../README.md)
