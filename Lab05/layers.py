import numpy as np
import torch

class Layer:
    """
    Responsible for modeling a single matrix in an Input
    """
    def __init__(self, output_shape, train=True):
        """
        :param output_shape (tuple): the shape of the output array.  When this is a single number,
        it gives the number of output neurons. When this is an array, it gives the dimensions 
        of the array of output neurons.
        :param train: Whether or not to update the gradients
        """
        if not isinstance(output_shape, tuple):
            output_shape = (output_shape,)
            
        self.train = train
            
        self.output_shape = output_shape
        self.grad = torch.zeros(self.output_shape, dtype=torch.float32)

    def accumulate_grad(self):
        """
        TODO: Add arguments as needed for this method.
        This method should accumulate its grad attribute with the value provided.
        """

    def clear_grad(self):
        """
        TODO: Add arguments as needed for this method.
        This method should clear grad elements. It should set the grad to the right shape 
        filled with zeros.
        """
        self.grad = torch.zeros(self.output_shape, dtype=torch.float32)

    def step(self):
        """
        TODO: Add arguments as needed for this method.
        Most tensors do nothing during a step so we simply do nothing in the default case.
        """
        pass

class Input(Layer):
    """
    Responsible for modeling a single matrix in a Linear layer
    """
    def __init__(self, output_shape, train=True):
        """
        :param output_shape (tuple): the shape of the output array.  When this is a single number,
        it gives the number of output neurons. When this is an array, it gives the dimensions 
        of the array of output neurons.
        :param train: Whether or not to update the gradients
        """
        Layer.__init__(self, output_shape, train) # TODO: Pass along any arguments to the parent's initializer here.

    def set(self, output):
        """
        :param output: Value of the matrix. If the shape of the matrix doesn't meet the expectations
        of the Input instance, an assertion error is raised
        """
        assert self.output_shape == output.shape
        self.output = output

    def randomize(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set the output of this input layer to random values sampled from the standard normal
        distribution (torch has a nice method to do this). Ensure that the output does not
        change size.
        """
        self.output = torch.rand(self.output_shape, dtype=torch.float32)

    def forward(self):
        """This layer's values do not change during forward propagation."""
        pass

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        This method does nothing as the Input layer should have already received its output
        gradient from the previous layer(s) before this method was called.
        """
        pass

    def step(self):
        """
        TODO: Add arguments as needed for this method.
        This method should have a precondition that the gradients have already been computed
        for a given batch.

        It should perform one step of stochastic gradient descent, updating the weights of
        this layer's output based on the gradients that were computed and a learning rate.
        """

class Linear(Layer):
    """
    Class responsible for modeling a Linear Layer with a ReLU activation function
    """
    def __init__(self, x, W, b, train=True):
        """
        :param x: The input matrix of the layer
        :param W: The weight matrix of the layer
        :param b: The biase matrix of the layer. If this doesn't equal the Input's expected shape,
        an assertion error is raised
        """
        Layer.__init__(self, b.output_shape, train) # TODO: Pass along any arguments to the parent's initializer here.
        self.x = x
        self.W = W
        self.b = b

    def forward(self):
        """
        Sets the layer's output based on the outputs of the layers that feed into it
        """
        self.output = (self.W.output @ self.x.output) + self.b.output

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """

class ReLU(Layer):
    def __init__(self, x, W, b, train=True):
        """
        :param x: The input matrix of the layer
        :param W: The weight matrix of the layer
        :param b: The biase matrix of the layer. If this doesn't equal the Input's expected shape,
        an assertion error is raised
        """
        Layer.__init__(self, b.output_shape, train) # TODO: Pass along any arguments to the parent's initializer here.
        self.x = x
        self.W = W
        self.b = b
        
    def ReLU(self, x):
        """
        :param x: The values to perform the ReLU activation function on
        """
        return x * (x > 0)

    def forward(self):
        """
        Sets the layer's output based on the outputs of the layers that feed into it after applying the
        ReLU activation function
        """
        self.output = self.ReLU((self.W.output @ self.x.output) + self.b.output)

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """

class MSELoss(Layer):
    """
    This is a good loss function for regression problems.

    It implements the MSE norm of the inputs.
    """
    def __init__(self, actual, predicted):
        """
        Mean squared error layer used as a loss function
        :param actual: the actual values
        :param predicted: the predicted values
        """
        Layer.__init__(self, (1, predicted.output_shape[1])) # TODO: Pass along any arguments to the parent's initializer here.
        self.actual = actual
        self.predicted = predicted

    def forward(self):
        """
        Computes the difference between the two matrices, squares the results, and finds the mean
        """
        self.output = torch.mean((self.actual.output - self.predicted.output)**2, dim=0)

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """

class Regularization(Layer):
    def __init__(self, input_tensor, train=False):
        """
        Computes regularization term in order to decrease overfitting
        :param input_tensor: the tensor to be regularized
        :param train: Whether to keep the gradients during back propagation
        """
        Layer.__init__(self, (1,1), train) # TODO: Pass along any arguments to the parent's initializer here.
        self.input_tensor = input_tensor

    def forward(self):
        """
        Squares the input and computes the sum
        """
        self.output = (self.input_tensor.output**2).sum()

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """

class Softmax(Layer):
    """
    This layer is an unusual layer.  It combines the Softmax activation and the cross-
    entropy loss into a single layer.

    The reason we do this is because of how the backpropagation equations are derived.
    It is actually rather challenging to separate the derivatives of the softmax from
    the derivatives of the cross-entropy loss.

    So this layer simply computes the derivatives for both the softmax and the cross-entropy
    at the same time.

    But at the same time, it has two outputs: The loss, used for backpropagation, and
    the classifications, used at runtime when training the network.

    TODO: Create a self.classifications property that contains the classification output,
    and use self.output for the loss output.

    See https://www.d2l.ai/chapter_linear-networks/softmax-regression.html#loss-function
    in our textbook.

    Another unusual thing about this layer is that it does NOT compute the gradients in y.
    We don't need these gradients for this lab, and usually care about them in real applications,
    but it is an inconsistency from the rest of the lab.
    """
    def __init__(self, actual, predicted):
        """
        Softmax and cross entropy loss comobined into a single layer
        :param actual: the actual values
        :param predicted: the predicted values
        """
        Layer.__init__(self, (1,1)) # TODO: Pass along any arguments to the parent's initializer here.
        self.actual = actual
        self.predicted = predicted
        
    def softmax(self):
        """
        Responsible for calculating the softmax output of a provided network layer
        """
        e = torch.exp(self.predicted.output - torch.max(self.predicted.output))
        self.classifications = e / e.sum()
        
    def cross_entropy(self):
        """
        Calculates the cross entropy loss for predicted vs. the actual values
        """
        self.softmax()
        #Small constant such that soft_max + epsilon is never 0
        epsilon = 1*10**-7
        L = (self.actual.output * torch.log(self.classifications + epsilon)).sum()
        return -1*L
        
        
    def forward(self):
        """
        Computes the cross entropy loss according to the supplied input
        """
        self.output = self.cross_entropy()

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """

class Sum(Layer):
    def __init__(self, left_input, right_input, train=True):
        """
        Layer that is responsible for adding the inputs from two layers
        :param left_input: the first input to be added
        :param right_input: the second input to be added
        """
        assert left_input.output_shape == right_input.output_shape
        Layer.__init__(self, left_input.output_shape, train) # TODO: Pass along any arguments to the parent's initializer here.
        self.left_input = left_input
        self.right_input = right_input
        
    def forward(self):
        """
        Computes element wise summation of the two tensors
        """
        self.output = self.left_input.output + self.right_input.output

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
