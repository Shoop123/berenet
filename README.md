# berenet
Matrix based neural network library.

> This is a library I threw together while learning about neural networks. So far it creates and trains (with backpropogation)
a regular ANN with a few tuning knobs available to cutomize training. Those will be covered further down.

<hr>

**This library used the matrix functionality provided by numpy. If you do not have it install follow this link and get it:
https://www.scipy.org/scipylib/download.html
or just run "pip install numpy" in terminal (only tested in Ubuntu).**

## Example
'''
\# OR Gate

\# Import the BereNet and numpy (as np) classes from berenet
from berenet import BereNet, np

#Initialize the training data as a numpy array
training_data = np.array((
		(1, 0),
		(1, 1),
		(0, 1),
		(0, 0)
), dtype=np.float64)

#Initialize the target outputs for the neural network
targets = np.array((
  (1,),
  (1,),
  (1,),
  (0,)
), dtype=np.float64)

#Create the neurual network
#The first argument is a list of neurons per layer. So this network will have 2 input neurons, 2 hidden neurons, and 1 output neuron
nn = BereNet([2, 2, 1], 1)
nn.verbosity += 'm'

nn.train(training_data, targets, 0.1, 10000, bold_driver=True)

print nn.predict(training_data)
'''
