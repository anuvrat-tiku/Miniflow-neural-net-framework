**Implementing the graph structure of a neural net in Miniflow**

***MiniFlow has two methods to help you define and then run values through your graphs: topological_sort() and forward_pass().***

**In order to define your network, you'll need to define the order of operations for your nodes. Given that the input to some node depends on the outputs of others, you need to flatten the graph in such a way where all the input dependencies for each node are resolved before trying to run its calculation. This is a technique called a topological sort.**

**Files**


1. miniflow.py - Implementation of the layers, forward and back prop, stochastic gradient descent.


2. net.py - Creates a layer and tests that stochastic gradient descent works as expected (with gradients calculated via correct back prop)

