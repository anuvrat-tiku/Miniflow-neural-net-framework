import numpy as np

"""Graph architecture in Mini-flow. Mini-flow is a simplified architecture of TensorFlow. """

"""Node defines the base set of properties that every node holds. Each node might receive input from multiple other 
nodes EXCEPT for the input node. The input node does not take any input. Also each node creates a single output which 
is passed to the next node. """


class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Nodes from which this node will receive input values.
        self.inbound_nodes = inbound_nodes

        # Nodes to which this node will pass values
        self.outbound_nodes = []

        # For each node in inbound_nodes, assign this node as the outbound node to _that_ node.
        for x in inbound_nodes:
            x.outbound_nodes.append(self)

        # Each node will calculate a value that it propagates forward
        self.value = 0

        # Keys are the inputs to this node and their values are the partials of this node with respect to that input.

        self.gradients = {}

    """ Each node will be able to pass values forward and perform back-propagation. """

    def forward(self):
        for w in self.inbound_nodes:
            self.value += w.value


""" Define the Input class as the subclass of Node. It will inherit all properties from Node. """


class Input(Node):
    def __init__(self):
        # Input has no inbound nodes, so initialization required here.
        Node.__init__(self)

    """ Unlike other nodes, the input node does not calculate any value such as a linear combination (sum(w.x)).
     It simply holds the values like weights and biases. Values can either be set explicitly or through the forward method.
     """

    def forward(self, value=None):
        # Override the value only if one is passed
        if value is not None:
            self.value = value

    def backward(self):
        # An Input node has no inputs so the gradient (derivative)
        # is zero.
        # The key, `self`, is reference to this object.
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1


""" Add is another subclass of Node that performs all the calculation. This add function accepts n nodes as input. """


class Add(Node):
    def __init__(self, *args):
        """
        Notice the difference in the __init__ method, Add.__init__(self, [x, y]).
        Unlike the Input class, which has no inbound nodes,
        the Add class takes n inbound nodes and adds the values of those nodes.
        """
        lst = []
        for i in args:
            lst.append(i)

        Node.__init__(self, lst)

    def forward(self):
        # Set the value of this node (`self.value`) to the sum of its inbound_nodes.
        self.value = 0
        for w in self.inbound_nodes:
            self.value += w.value


"""
Class to define the linear combination of weights, input and biases. The add node simply adds the input.
But a real neural network can also improve its efficiency over time.
It does it using weights, which are the tuning knobs of the network.
y = ∑(w.x) + b
"""


class Linear(Node):
    # The weights and bias properties here are not
    # numbers, but rather references to other nodes.
    # The weight and bias values are stored within the
    # respective nodes.
    def __init__(self, inputs, weights, bias):
        # Vales passed to the constructor is an array of arrays. Inputs and weights are arrays and bias is an integer.
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        """
        Will define the linear combination of the weights, inputs and bias.
        :return: None
        """
        inputs = np.array(self.inbound_nodes[0].value)
        weights = np.array(self.inbound_nodes[1].value)
        bias = np.array(self.inbound_nodes[2].value)

        self.bias += np.dot(inputs, weights) + bias

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize a partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # Set the partial of the loss with respect to this node's weights.
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this node's bias.
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    def __init__(self, x):
        Node.__init__(self, [x])

    """Define the sigmoid function, S(x) = 1/1 + e^(-)"""

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def forward(self):
        input_value = self.inbound_nodes[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Sum the derivative with respect to the input over all the outputs.
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost


class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    # We reshape these to avoid possible matrix/vector broadcast errors.
    def forward(self):
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        m = self.inbound_nodes[0].value.shape[0]

        self.value = np.mean((y - a) ** 2)


"""
The topological_sort() function implements topological sorting using Kahn's Algorithm.
Topological_sort() returns a sorted list of nodes in which all of the calculations can run in series.
Topological_sort() takes in a feed_dict, which is how we initially set a value for an Input node.
"""


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value


def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Nodes.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for n in graph[::-1]:
        n.backward()


def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    Arguments:

        `trainables`: A list of `Input` nodes representing weights/biases.
        `learning_rate`: The learning rate.
    """
    # Performs SGD
    #
    # Loop over the trainables
    for t in trainables:
        # Change the trainable's value by subtracting the learning rate
        # multiplied by the partial of the cost with respect to this
        # trainable.
        partial = t.gradients[t]
        t.value -= learning_rate * partial
