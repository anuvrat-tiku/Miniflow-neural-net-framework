**Implementing the graph structure of a neural net in Miniflow**

***MiniFlow has two methods to help you define and then run values through your graphs: topological_sort() and forward_pass().***

**In order to define your network, you'll need to define the order of operations for your nodes. Given that the input to some node depends on the outputs of others, you need to flatten the graph in such a way where all the input dependencies for each node are resolved before trying to run its calculation. This is a technique called a topological sort.**

