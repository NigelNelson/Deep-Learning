import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import warnings
import os.path
import nvidia_smi

EPOCHS = 50

# For simple regression problem
TRAINING_POINTS = 1000

# For fashion-MNIST and similar problems
DATA_ROOT = '/../../data/cs3450/data/'
FASHION_MNIST_TRAINING = '/../../data/cs3450/data/fashion_mnist_flattened_training.npz'
FASHION_MNIST_TESTING = '/../../data/cs3450/data/fashion_mnist_flattened_testing.npz'
CIFAR10_TRAINING = '/../../data/cs3450/data/cifar10_flattened_training.npz'
CIFAR10_TESTING = '/../../data/cs3450/data/cifar10_flattened_testing.npz'
CIFAR100_TRAINING = '/../../data/cs3450/data/cifar100_flattened_training.npz'
CIFAR100_TESTING = '/../../data/cs3450/data/cifar100_flattened_testing.npz'

def load_dataset_flattened(train=True,dataset='Fashion-MNIST',download=False):
    """
    :param train: True for training, False for testing
    :param dataset: 'Fashion-MNIST', 'CIFAR-10', or 'CIFAR-100'
    :param download: True to download. Keep to false afterwords to avoid unneeded downloads.
    :return: (x,y) the dataset. x is a numpy array where columns are training samples and
             y is a numpy array where columns are one-hot labels for the training sample.
    """
    if dataset == 'Fashion-MNIST':
        if train:
            path = FASHION_MNIST_TRAINING
        else:
            path = FASHION_MNIST_TESTING
        num_labels = 10
    elif dataset == 'CIFAR-10':
        if train:
            path = CIFAR10_TRAINING
        else:
            path = CIFAR10_TESTING
        num_labels = 10
    elif dataset == 'CIFAR-100':
        if train:
            path = CIFAR100_TRAINING
        else:
            path = CIFAR100_TESTING
        num_labels = 100
    else:
        raise ValueError('Unknown dataset: '+str(dataset))

    if os.path.isfile(path):
        print('Loading cached flattened data for',dataset,'training' if train else 'testing')
        data = np.load(path)
        x = torch.tensor(data['x'],dtype=torch.float32)
        y = torch.tensor(data['y'],dtype=torch.float32)
        pass
    else:
        class ToTorch(object):
            """Like ToTensor, only to a numpy array"""

            def __call__(self, pic):
                return torchvision.transforms.functional.to_tensor(pic)

        if dataset == 'Fashion-MNIST':
            data = torchvision.datasets.FashionMNIST(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        elif dataset == 'CIFAR-10':
            data = torchvision.datasets.CIFAR10(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        elif dataset == 'CIFAR-100':
            data = torchvision.datasets.CIFAR100(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        else:
            raise ValueError('This code should be unreachable because of a previous check.')
        x = torch.zeros((len(data[0][0].flatten()), len(data)),dtype=torch.float32)
        for index, image in enumerate(data):
            x[:, index] = data[index][0].flatten()
        labels = torch.tensor([sample[1] for sample in data])
        y = torch.zeros((num_labels, len(labels)), dtype=torch.float32)
        y[labels, torch.arange(len(labels))] = 1
        np.savez(path, x=x.detach().numpy(), y=y.detach().numpy())
    return x, y
	
	
dataset = 'Fashion-MNIST'
# dataset = 'CIFAR-10'
# dataset = 'CIFAR-100'

#x_train, y_train = create_linear_training_data()
#x_train, y_train = create_folded_training_data()
#points_train, insideness_train = create_square()
x_train, y_train = load_dataset_flattened(train=True, dataset=dataset, download=False)

# Move selected datasets to GPU
x_train = x_train.to(DEVICE)
y_train = y_train.to(DEVICE)

#x_test, y_test = create_linear_training_data()
x_test, y_test = load_dataset_flattened(train=False, dataset=dataset, download=False)

# Move the selected datasets to the GPU
x_test = x_test.to(DEVICE)
y_test = y_test.to(DEVICE)

class Layer:
    """
    Responsible for modeling a single matrix in an Input
    """
    def __init__(self, output_shape):
        """
        :param output_shape (tuple): the shape of the output array.  When this is a single number,
        it gives the number of output neurons. When this is an array, it gives the dimensions 
        of the array of output neurons.
        """
        if not isinstance(output_shape, tuple):
            output_shape = (output_shape,)
            
        self.output_shape = output_shape
        
        
class Input(Layer):
    """
    Responsible for modeling a single matrix in a Linear layer
    """
    def __init__(self, output_shape):
        """
        :param output_shape (tuple): the shape of the output array. Passed to parent's initializer
        """
        Layer.__init__(self, output_shape)

    def set(self, value):
        """
        :param value: Value of the matrix. If the shape of the matrix doesn't meet the expectations
        of the Input instance, an assertion error is raised
        """
        assert self.output_shape == value.shape
        self.output = value
        
    def forward(self):
        """This layer's values do not change during forward propagation."""
        pass


class LinearReLU(Layer):
    """
    Class responsible for modeling a Linear Layer with a ReLU activation function
    """
    def __init__(self, x, W, b):
        """
        :param x: The input matrix of the layer
        :param W: The weight matrix of the layer
        :param b: The biase matrix of the layer. If this doesn't equal the Input's expected shape,
        an assertion error is raised
        """
        Layer.__init__(self, b.output_shape) # TODO: Pass along any arguments to the parent's initializer here.
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
        self.output.retain_grad()
   

class Linear(Layer):
    """
    Class responsible for modeling a Linear Layer without an activation function
    """
    def __init__(self, x, W, b):
        """
        :param x: The input matrix of the layer
        :param W: The weight matrix of the layer
        :param b: The biase matrix of the layer. If this doesn't equal the Input's expected shape,
        an assertion error is raised
        """
        Layer.__init__(self, b.output_shape) # TODO: Pass along any arguments to the parent's initializer here.
        self.x = x
        self.W = W
        self.b = b
    

    def forward(self):
        """
        Sets the layer's output based on the outputs of the layers that feed into it
        """
        self.output = (self.W.output @ self.x.output) + self.b.output
        self.output.retain_grad()
        
class Network:
    """
    Class responsible for defining the behavior of a simple Neural Network with a single hidden layer
    """
    def __init__(self, input_rows, num_hidden_nodes, num_classes, dtype=torch.float32, device=torch.device('cuda:0')):
        """
        :param input_rows: The number of rows expected in the input of the network
        :param num_hidden_nodes: The number of nodes in the hidden layer desired
        :param dtype: The data type to be used with the PyTorch tensors
        :param device: The device desired to be used with the PyTorch tensors
        """
        # Define weights and bias matrices for input -> hidden layer
        W = torch.rand((num_hidden_nodes, input_rows), dtype=dtype, device=device, requires_grad=True)
        W.data *= 0.1
        b1 = torch.zeros((num_hidden_nodes,1), dtype=dtype, device=device, requires_grad=True)
        
        # Define weights and bias matrices for hidden layer -> ouput
        M = torch.rand((num_classes ,num_hidden_nodes), dtype=dtype, device=device, requires_grad=True)
        M.data *= 0.1
        b2 = torch.zeros((num_classes, 1), dtype=dtype, device=device, requires_grad=True)

        # Create Input instances for all matrices
        W_layer = Input((num_hidden_nodes, input_rows))
        W_layer.set(W)
        b1_layer = Input((num_hidden_nodes,1))
        b1_layer.set(b1)
        M_layer = Input((num_classes,num_hidden_nodes))
        M_layer.set(M)
        b2_layer = Input((num_classes,1))
        b2_layer.set(b2)

        # Create 1st layer with ReLU activation function
        x1_layer = Input(x_train.shape[0])
        linear_layer1 = LinearReLU(x1_layer, W_layer, b1_layer)
        
        # Create 2nd layer without activation function
        x2_layer = Input(b1_layer.output.shape[0])
        linear_layer2 = Linear(x2_layer, M_layer, b2_layer)
        
        # Assign class variables
        self.layer1 = linear_layer1
        self.layer2 = linear_layer2
    
    
    def L2(self, actual, predicted):
        """
        Returns the L2 loss of the supplied args
        :param actual: The true values
        :param predicted: The predicted values
        """
        return ((actual - predicted)**2).mean()
    
    def softmax(self, X):
        e = torch.exp(X - torch.max(X))
        return e / e.sum()
    
    def cross_entropy(self, actual, predicted):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector. 
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        soft_max = self.softmax(predicted)
        epsilon = 1*10**-7
        L = (actual * torch.log(soft_max + epsilon)).sum()
        return -1*L, soft_max
        
    def train(self, x_train, y_train, num_epochs, learning_rate, reg_const, batch_size):
        """
        Method responsible for training the Neural Network
        :param x_train: The X training data
        :param y_train: The y training labels
        :param num_epochs: Number of epochs to train for
        :param learning_rate: The rate at which parameters are adjusted
        :param reg_const: The regularization constant that scales the regularization term
        :param batch_size: The batch size used for training
        
        """
        # Adjust the x matrices according to the batch size
        self.layer1.x = Input((x_train.shape[0], batch_size))
        self.layer2.x = Input((self.layer1.b.output.shape[0], batch_size))
        
        for epoch in range(num_epochs):
            num_correct = 0
            num_samples = 0
            for i in range(x_train.shape[1]//batch_size):
                
                num_samples += batch_size
                
                # Get the correct locations to reference in the training and testing sets
                start_idx = i*batch_size
                end_idx = i*batch_size + batch_size

                # Populate the x matrix with the training samples in this batch
                self.layer1.x.set(x_train[:, start_idx : end_idx].reshape(x_train.shape[0], batch_size))
                self.layer1.forward()
                self.layer2.x.set(self.layer1.output)
                self.layer2.forward()
                
                true_labels = (y_train[:, start_idx : end_idx]).reshape(y_train.shape[0], batch_size)

                # Calculate the L2 loss using the output of layer 2 and the associated samples in
                # y_train
                loss, soft_max = self.cross_entropy(true_labels, self.layer2.output)
                
                num_correct += (torch.argmax(true_labels, dim=0) == torch.argmax(soft_max, dim=0)).sum().item()
                accuracy = num_correct / num_samples
                
#                 print(f'Num correct: {num_correct}')
#                 print(f'Num samples: {num_samples}')

                # Calculate the regularization term
                s1 = (self.layer1.W.output**2).sum()
                s2 = (self.layer2.W.output**2).sum()
                reg = reg_const*(s1 + s2)

                # Calculate the final cost term
                cost = loss + reg

                # Compute backpropagation with Autograd
                cost.backward()
            

                # Used to update parameters inplace
                with torch.no_grad():

                    # Adjust parameters according to gradients and the learning rate
                    self.layer1.W.output -= learning_rate * self.layer1.W.output.grad
                    self.layer1.b.output -= learning_rate * self.layer1.b.output.grad
                    self.layer2.W.output -= learning_rate * self.layer2.W.output.grad
                    self.layer2.b.output -= learning_rate * self.layer2.b.output.grad

                    # Zero the gradients
                    self.layer1.W.output.grad.zero_()
                    self.layer1.b.output.grad.zero_()
                    self.layer2.W.output.grad.zero_()
                    self.layer2.b.output.grad.zero_()
                    
            print(f'Epoch #{epoch + 1} Accuracy: {accuracy}, Loss: {loss.item()}')
            
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
    
    def test(self, x_test, y_test):
        """
        Method responsible for testing the Neural Network after it has been trained
        :param x_train: The X training data
        :param y_train: The training labels
        """
        # Recalibrate the 1st x layer according to the shape of the testing data
        self.layer1.x = Input(x_test.shape)
        self.layer1.x.set(x_test)
        
        self.layer1.forward()
        
        # Recalibrate the 2nd x layer according to the shape of the output of the 1st layer
        self.layer2.x = Input(self.layer1.output.shape)
        self.layer2.x.set(self.layer1.output)

        self.layer2.forward()
        
        true_labels = torch.argmax(y_test, dim=0)
        predicted = torch.argmax(self.softmax(self.layer2.output), dim=0)
        
        num_correct = (predicted == true_labels).sum().item()
        total_samples = y_test.shape[1]
        accuracy = num_correct / total_samples
    
        print(f'Testing Accuracy: {accuracy}')
 

input_rows = 784
num_hidden_nodes = 24
num_epochs = 20
learning_rate = .001
reg_const = 0
batch_size = 4
num_classes = 10

print(f'Input rows: {input_rows}')
print(f'Num hidden nodes: {num_hidden_nodes}')
print(f'Epochs: {num_epochs}')
print(f'Learning Rate: {learning_rate}')
print(f'Reg_const: {reg_const}')
print(f'Batch size: {batch_size}')
print()

network = Network(input_rows, num_hidden_nodes, num_classes)
%time network.train(x_train, y_train, num_epochs, learning_rate, reg_const, batch_size)


print()

network.test(x_test, y_test)