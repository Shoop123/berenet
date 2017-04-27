# berenet
Matrix based neural network library.

> This is a library I threw together while learning about neural networks. So far it creates and trains (with backpropogation)
a regular ANN with a few tuning knobs available to cutomize training. Those will be covered further down.

<hr>

_**NOTE:** This library used the matrix functionality provided by numpy. If you do not have it installed follow this link and get it:
https://www.scipy.org/scipylib/download.html  
or just run "pip install numpy" in terminal (only tested on Ubuntu)._

## Example: Training an OR Gate!

In case you don't know, an OR gate is a type of switch in the hardware universe. The basic idea is that it has 2 inputs, and 1 output. If either of the inputs are 1, the output is also a 1. Otherwise the output is a 0.

```
# Import the BereNet and numpy (as np) classes from berenet
from berenet import BereNet, np

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
# The first argument is the list of neurons for each layer.
# This network will have 2 input neurons, 2 hidden neurons, and 1 output neuron
# The second agument is the size of the mini batches
nn = BereNet([2, 2, 1], 1)

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
print nn.predict(training_data)
```
The output should look something like this: 
```
[[ 0.97617963]
 [ 0.99576899]
 [ 0.97617959]
 [ 0.04974998]]
```
So this means that our network thinks that the outputs for the first 3 examples are 1's, and for the last one a 0. as you can see from the target values we gave it, that's correct! :ok_hand:

## More Functionality :muscle:

#### Saving/Loading Models
To save a model, simple call the `save('or_gate.net')` method, passing in a file name inside with which the model with be saved. The saving mechanism is just using pickle, so nothing special here... Maybe we should teach it how to save itself :laughing:
```
nn.save('or_gate.net')
```
To load back a saved model, call the static method `load('or_gate.net')`, once again, passing in the file name of the saved network.
```
nn = BereNet.load('or_gate.net')
```

#### Staying in the know
The BereNet class has an included variable called `verbosity`, which contains letters representing different things the network should print out during training. I don't expect anyone to memorize these option (even though there are only a few), so there's a method that will print out everything this field can do. Let's call it:
```
nn.show_verbosity_legend()
```
Gives us:
```
Verbosity Lengend:
m is to show mean squared error every time it changes
s is to show sample metrics
e is to show epochs
n is to show minibatch number with every epoch
```
So, to expand on what the list says:
* `nn.verbosity += 'm'` will make it print out the mean squared error (later referred to as MSE) after every epoch (less accurate), and once at the end (more accurate). To calculate this error it uses either validation data that was passed through the `train` method, or the test data otherwise. The reason it prints only the MSE instead of the accuracy score is because it is very difficult measure accuracy when the goal of the network is unkown, and the MSE is a decent representation of whether or not the netowrk is improving.
* `nn.verbosity += 's'` will just show some information after dividing the data into the specified minibatches, such as number of examples, number of minibatches, and the size of each minibatch.
* The default for verbosity is just 'e', which means that when training, the network will print out what epoch it is currently on
* `nn.verbosity += 'n'` This is the same as 'e', except it will print out what minibatch it is on in the epoch
