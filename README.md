# berenet
Matrix based neural network library.

> This is a library I threw together while learning about neural networks. So far it can create and trains (with backpropagation)
a regular ANN and a basic sequence-sequence RNN with a few tuning knobs available to customize training. Those will be covered further down. Currently this library is made for Python 3+.

<hr>

![Image of a Neural Network](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Artificial_neural_network.svg/2000px-Artificial_neural_network.svg.png)

_**NOTE:** This library used the matrix functionality provided by numpy. If you do not have numpy, get it from [here](https://www.scipy.org/scipylib/download.html), or just run "pip install numpy" in terminal (only tested on Ubuntu)._

## Table of Contents
* [Installing](#installing)
* [ANN Example: Training an OR Gate!](#example-training-an-or-gate)
* [RNN Example: Testing recurrence!](#example-testing-recurrence)
* [More Functionality :muscle:](#more-functionality)
  * [Saving/Loading Models](#saving-loading-models)
  * [Staying in the Know](#staying-in-the-know)
  * [Tuning knobs :zap:](#tuning-knobs)
    * [The Train Method](#the-train-method)
    * [The Predict Method](#the-predict-method)
    * [The Constructor](#the-constructor)

<a name="installing"></a>
## Installing
#### The Easy Way (only tested on Ubuntu)
This method is the pip method. All you have to do is fire up terminal and run
```
pip install berenet
```
And voilà, you're good to go, just test it by running
```
from berenet import BereNet, BerecurreNet
```

#### The More Annoying Way
Download the git repo! After that just put the downloaded files into your project, and import them and you're good to go!

<a name="example-training-an-or-gate"></a>
## ANN Example: Training an OR Gate!
In case you don't know, an OR gate is a type of switch in the hardware universe. The basic idea is that it has 2 inputs, and 1 output. If either of the inputs are 1, the output is also a 1. Otherwise the output is a 0.

```
# Import BereNet and numpy
from berenet import BereNet
import numpy as np

# Initialize the training data as a numpy array
# Here the rows are examples, while the columns are inputs to the network
# There must be the same number of columns as input neurons
training_data = np.array((
    (1, 0),
    (1, 1),
    (0, 1),
    (0, 0)
), dtype=np.float64)

# Initialize the target outputs for the neural network
# There must be the same number of rows here as in training_data
# There must also be the same number of columns as output neurons for the network
targets = np.array((
  (1,),
  (1,),
  (1,),
  (0,)
), dtype=np.float64)

# Create the neural network
# This network will have 2 input neurons, 2 hidden neurons, and 1 output neuron
nn = BereNet([2, 2, 1])

# Train the network
# First argument is the training data (will be split into 4 arrays with 1 minibatch each, as specified above)
# Second argument is self-explanatory
# Third argument is the learning rate
# Fourth argument is the number of epochs to train for
nn.train(training_data, targets, 0.1, 10000)
```
Now let's use our newly trained model to make a prediction!
```
# Print out the prediction for all 4 examples we trained on
print(nn.predict(training_data))
```
The output should look something like this: 
```
[[ 0.97617963]
 [ 0.99576899]
 [ 0.97617959]
 [ 0.04974998]]
```
So this means that our network thinks that the outputs for the first 3 examples are 1's, and for the last one a 0. As you can see from the target values we gave it, that's correct! :ok_hand:

<a name="example-testing-recurrence"></a>
## RNN Example: Testing recurrence!
So here we will take a look at how to use the recurrent neural network to learn some random pattern in some made-up data. The goal is just to demonstrate the functionality of the RNN, and introduce you to how it works rather than build the next Ultron, so we'll keep the dataset fun size for now.

```
# The RNN class is called BerecurreNet
from berenet import BerecurreNet
import numpy as np

# Initialize the training data as a numpy array
# The RNN data has not only instances and features, it also has time steps so its shape will be a little different
# Here the rows are still examples, the columns are still inputs to the network, however you can probably tell that
# the rows are split up into "chunks". These chunks are time steps, the first being time step 1, and then 2, and so on...
# There must be the same number of columns as input neurons, and the same number of "chunks" as time steps
# And finally, to have multiple examples, you must put multiple rows into each chunk, ensuring that each chunk has the same number of rows
training_data = np.array((
  (
    (1, 0, 0, 0),
  ),
  (
    (0, 1, 0, 0),
  ),
  (
    (0, 0, 1, 0),
  ),
  (
    (0, 0, 1, 0),
  )
), dtype=np.float64)

# Initialize the target outputs for the neural network
# This data is in the same format, and same shape as the training data
# Each chunk corresponds to outputs for that time step respective to the above input
# There must be the same number of rows here as in training_data
# There must also be the same number of columns as output neurons for the network
targets = np.array((
  ((0, 1, 0, 0)),
  ((0, 0, 1, 0)),
  ((0, 0, 1, 0)),
  ((0, 0, 0, 1)),
), dtype=np.float64)

# Create the neural network
# The first argument is the list of neurons for each layer.
# This network will have 4 input neurons, 10 hidden (recurrent) neurons, and 4 output neurons
rnn = BerecurreNet([4, 10, 4])

# Train the network
# First argument is the training data
# Second argument is self-explanatory
# Third argument is the learning rate
# Fourth argument is the number of epochs to train for
rnn.train(training_data, targets, 1, 1000)
```
Now let's use our newly trained model to make a prediction!
```
# Print out the prediction for all 4 examples we trained on
print(rnn.predict(training_data, reset=True, round_digits=1))
```
The output should look something like this: 
```
[[[0.  1.  0.  0. ]]
 [[0.  0.  1.  0. ]]
 [[0.  0.  0.9 0.1]]
 [[0.  0.  0.1 0.9]]]
```
If we compare this with our desired output, well, its not perfect, but its definitely not far off!

<a name="more-functionality"></a>
## More Functionality :muscle:
Below is some documentation describing some cool features that you can play around with in this library. Not all features are implemented for both classes, so under each title will be a line indicating the supported classes.
<a name="saving-loading-models"></a>
#### Saving/Loading Models
**_Supported Classes: [BereNet, BerecurreNet]_**  
To save a model, simply call the `save('or_gate.berenet')` method, passing in a file name inside with which the model will be saved. The saving mechanism is just using pickle, so nothing special here... Maybe we should teach it how to save itself :laughing:
```
nn.save('or_gate.berenet')
```
To load back a saved model, call the static method `load('or_gate.berenet')`, once again, passing in the file name of the saved network.
```
nn = BereNet.load('or_gate.berenet')
```
Or, for its RNN counterpart:
```
rnn = BerecurreNet.load('or_gate.berenet')
```
<a name="staying-in-the-know"></a>
#### Staying in the Know
##### Verbosity
**_Supported Classes: [BereNet]_**  
The BereNet class has an included variable called `verbosity`, which contains letters representing different things the network should print out during training. There's a handy method that will print out everything this field can do. Let's call it:
```
nn.show_verbosity_legend()
```
Gives us:
```
Verbosity Legend:
m is to show mean squared error everytime it changes
s is to show sample metrics
e is to show epochs
n is to show minibatch number with every epoch
```
So, to expand on what the list says:
* `nn.verbosity += 'm'` will make it print out the mean squared error (later referred to as MSE) after every epoch (usually smaller than actual size), and once at the end (more accurate). To calculate this error it uses either validation data that was passed through the `train` method, or the test data otherwise. The reason it prints only the MSE instead of the accuracy score is because it is very difficult measure accuracy when the goal of the network is unknown, and the MSE is a decent representation of whether or not the network is improving.
* `nn.verbosity += 's'` will just show some information after dividing the data into the specified minibatches, such as number of examples, number of minibatches, and the size of each minibatch.
* The default for verbosity is just 'e', which means that when training, the network will print out what epoch it is currently on. This is so that when taking a while to train you have some sort of idea of how much progress has been made, rather than sitting and waiting in suspense, not knowing how much longer it's going to be.
* `nn.verbosity += 'n'` This is the same as 'e', except it will print out what minibatch it is on in the epoch
##### Summary
**_Supported Classes: [BereNet, BerecurreNet]_**  
The train method also has a summarize parameter. Very simply, when set to true, the class will print a few lines about the neural network that has been contructed. To see how it works, lets see what setting `summarize=True` will result in for our RNN example above:
```
rnn.train(X, targets, 1, 1000, summarize=True)
```
Will print:
```
INFO:root:Type: Recurrent
INFO:root:  Layers: 3
INFO:root:  Recurrent Layers: 1
INFO:root:  Feed Forward Layers: 2
INFO:root:  Total Neurons: 18
INFO:root:  Total Weights: 90
INFO:root:Data
INFO:root:  Time Steps: 4
INFO:root:  Instances per time step: 1
INFO:root:  Total Instances: 4
INFO:root:  Data Shape: (4, 1, 4)
INFO:root:Parameters
INFO:root:  Learning Rate: 1
INFO:root:  Epochs: 1000
```
<a name="tuning-knobs"></a>
#### Tuning knobs :zap:

<a name="the-train-method"></a>
###### The Train Method
You've already seen how to change the learning rate and epochs, but there's a few more tricks up BereNet's sleeve!
For one, there are many optional parameters for the `train` method that we skipped in our simple example above.  
**_Supported Classes: [BereNet, BerecurreNet]_**  
Up first, lets see how we can set the minibatch size for our neural network:
```
nn.train(training_data, targets, 0.1, 10000, minibatch_size=100)
```
So now our neural network will split our training data into batches of 100 (default is 1), and calculate the gradient for each instance in the batch before updating the weights.  
**_Supported Classes: [BereNet, BerecurreNet]_**  
Let's take a look at the validation data option:
```
nn.train(training_data, targets, 0.1, 10000, validation_data=valid_data, validation_targets=valid_targets)
```
All this does is make the MSE calculation use the validation data rather than the training data when `nn.verbosity` contains the letter 'm'.  
**_Supported Classes: [BereNet, BerecurreNet]_**  
Up next, something more interesting perhaps, momentum!
```
nn.train(training_data, targets, 0.1, 10000, momentum=0.9)
```
This is fairly self-explanatory if you know what momentum is. Fortunately, if you don't, I have a link for you that describes it well! And as a bonus it also describes the next 2 parameters that I will talk about, so check it out for reference!
https://www.willamette.edu/~gorr/classes/cs449/momrate.html  
**_Supported Classes: [BereNet]_**  
And now, something slightly more niche, the "Bold Driver" algorithm for learning rate optimization. To use it just set the flag to `True` like this:
```
nn.train(training_data, targets, 0.1, 10000, bold_driver=True)
```
Keep in mind that this algorithm is designed for full-batch learning, so unless your minibatch size is the same as your sample size, it will give you a warning.  
**_Supported Classes: [BereNet, BerecurreNet]_**  
Next up, annealing, applied like so:
```
nn.train(training_data, targets, 0.1, 10000, annealing_schedule=1000000)
```
`annealing_schedule` is the value for T in the annealing formula "µ<sub>new</sub> = µ<sub>old</sub>/(1 + epoch/T)" where µ is the learning rate. This value is best obtained by trial-and-error, just like the learning rate, and heavily relies on the number of epochs.  
**_Supported Classes: [BereNet]_**  
Now, we have L2 regularization, which basically combats overfitting by manipulating how the weights are changed. Anyways, to enable it... Well, you guessed it, set the optional parameter to your preferred λ (the regularization strength). 
```
nn.train(training_data, targets, 0.1, 10000, l2_regularizer=0.05)
```
Keep in mind, for this one, the higher the value, the more it will focus on preventing overfitting, and less on optimizing the error, and vice-versa. The key is to find a good balance (trial-and-error is your friend!) between focussing on error optimization and the prevention of overfitting.  
**_Supported Classes: [BerecurreNet]_**  
Lastly, we can set the the `track_error` parameter to an integer less than the our epoch count. This will print the mean squared error as many times as the parameter's values.
```
rnn.train(training_data, targets, 0.1, 1000, track_error=10)
```
This will print the MSE 10 times during training, or every 100 epochs. This parameter is useful when you want to see your neural network improving over time, or would like to debug some weird behaviour.
<a name="the-predict-method"></a>
###### The Predict Method
**_Supported Classes: [BereNet, BerecurreNet]_**  
The predict method has 2 straight-forward optional parameters. `softmax_output` and `round` will give you a probability distribution of your outputs using the softmax function, and round all of your numbers to a certain decimal respectively. For example
```
print nn.predict(training_data, softmax_output=True, round=2)
```
will produce the output
```
[[ 1.]
 [ 1.]
 [ 1.]
 [ 1.]]
```
The reason they are all 1 is because there is only 1 output. This is mostly useful for multiclass classification, since it helps you get an obvious answer.  
**_Supported Classes: [BerecurreNet]_**  
For the BerecurreNet class, we also have a `reset` parameter, which you no doubt have noticed in our example above. This parameter is necessary to reset the state of each of the recurrent layers, so that when you go in for your prediction it doesn't use old values from the recurrent units. This is optional as sometimes you want to pick up where you left off, and have your time sequence be saved until your next training session.
```
rnn.predict(training_data, reset=True)
```
<a name="the-constructor"></a>
###### The Constructor
**_Supported Classes: [BereNet, BerecurreNet]_**  
Other than the layer configuration and the minibatch size, the constructor can also take in an optional `functions` argument as a list of activation functions for each layer. This argument can be passed in 2 different ways.  
Way #1:
```
nn = BereNet([2, 2, 1], functions=[BereNet.ARCTAN])
```
This way will initialize the network to have the first layer with an identity function, and the rest with the arctan function. So with this method, you pass in a list with 1 of the available activation functions, and it will set all but the first layer to use them. The defualt is `functions=[BereNet.LOGISTIC]`.  
Way #2:
```
nn = BereNet([2, 2, 1], functions=[BereNet.RELU, BereNet.SOFTSIGN, BereNet.TANH])
```
This way allows for more customization, but with more work. You must pass in a list with the same size as the layer configuration list, with each index in the functions list corresponding to the respective index in the layers configuration.  
By now you're probably wondering well what functions can I use then? :confused: Well wonder no more!
The functions are:
* IDENTITY
* LOGISTIC
* TANH
* ARCTAN
* SOFTSIGN
* RELU

**_Supported Classes: [BereNet, BerecurreNet]_**  
You can also specify how many bias neurons you want to have at each layer:
```
rnn = BerecurreNet([1, 5, 5, 1], biases=[1, 2, 1])
```
This line will add 1 bias neuron to the input layer, 2 to the second layer, and 1 to the third. The default is having 1 bias neuron for each layer except the output layer.  
**_Supported Classes: [BerecurreNet]_**  
In order to specify which layers you want to have the recurrent property, you can use the `recurrent_layers` contructor parameter.
```
rnn = BerecurreNet([3, 30, 20, 10, 1], recurrent_layers=[False, True, False, True, False])
```
By default all layers excpect the input and output layers are recurrenct.
