class Network:
    """
    Class that holds references to layers to create a neural network
    """
    def __init__(self):
        """
        Initializes a layers attribute
        """
        self.layers = []

    def add(self, layer):
        """
        Adds a new layer to the network.

        Sublayers can *only* be added after their inputs have been added.
        (In other words, the DAG of the graph must be flattened and added in order from input to output)
        :param layer: The sublayer to be added
        """
        self.layers.append(layer)    

    def forward(self,input):
        """
        Compute the output of the network in the forward direction.
        :param input: A torch tensor that will serve as the input for this forward pass
        :return: A torch tensor with useful output (e.g., the softmax decisions)
        """
        
        # Set the input
        self.layers[0].set(input)
        
        for layer in self.layers:
            layer.forward()
            
        # Set the output
        self.output = self.layers[-1].output

    def backward(self):
        """
        Compute the gradient of the output of all layers through backpropagation over the 
        gradient tape.

        """
        for layer in self.layers:
            layer.clear_grad()
            
        for layer in self.layers[::-1]:
            layer.backward()

    def step(self, step_size):
        """
        Perform one step of the stochastic gradient descent algorithm
        based on the gradients that were previously computed by backward.

        """
        for layer in self.layers():
            layer.step(step_size)
