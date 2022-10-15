num_hidden_nodes = 2

W = torch.rand((num_hidden_nodes,2), dtype=torch.float32, device=device, requires_grad=True)
W.data *= 0.1
b1 = torch.zeros((num_hidden_nodes,1), dtype=torch.float32, device=device, requires_grad=True)

M = torch.rand((2,num_hidden_nodes), dtype=torch.float32, device=device, requires_grad=True)
M.data *= 0.1
b2 = torch.zeros((2,1), dtype=torch.float32, device=device, requires_grad=True)

W_layer = Input((num_hidden_nodes, 2))
W_layer.set(W)
b1_layer = Input((num_hidden_nodes,1))
b1_layer.set(b1)

M_layer = Input((2,num_hidden_nodes))
M_layer.set(M)
b2_layer = Input((2,1))
b2_layer.set(b2)

num_epochs = 500
learning_rate = .01
reg_const = 0.000001
batch_size = 8

x1_layer = Input((x_train.shape[0], batch_size))
linear_layer1 = LinearReLU(x1_layer, W_layer, b1_layer)

x2_layer = Input((b1_layer.output.shape[0], batch_size))
linear_layer2 = Linear(x2_layer, M_layer, b2_layer)


for epoch in range(num_epochs):
#     learning_rate /= 10**epoch
#     print(learning_rate)
    print(f'epoch #{epoch}')
    for i in range(x_train.shape[1]//batch_size):
        x1_layer.set(x_train[:, i*batch_size : (i*batch_size + batch_size)].reshape(x_train.shape[0], batch_size))

        linear_layer1.forward()

        x2_layer.set(linear_layer1.output)

        linear_layer2.forward()
        
        #(1/linear_layer2.output.numel()) * 
        
        mse = MSE((y_train[:,i*batch_size : (i*batch_size + batch_size)]).reshape(y_train.shape[0], batch_size),
                 linear_layer2.output)

        #mse = (((y_train[:,i*batch_size : (i*batch_size + batch_size)]).reshape(y_train.shape[0], batch_size) - linear_layer2.output)**2).sum()
        
        s1 = (W_layer.output**2).sum()
        s2 = (M_layer.output**2).sum()
        S = reg_const*(s1 + s2)
        
        cost = mse + S

        cost.backward()

        with torch.no_grad():

            W_layer.output -= learning_rate * W_layer.output.grad
            b1_layer.output -= learning_rate * b1_layer.output.grad

            M_layer.output -= learning_rate * M_layer.output.grad
            b2_layer.output -= learning_rate * b2_layer.output.grad

            W_layer.output.grad.zero_()
            b1_layer.output.grad.zero_()
            M_layer.output.grad.zero_()
            b2_layer.output.grad.zero_()
            
    print(mse.item())
    
    
    
def trainNN(num_epochs, learning_rate, reg_const, batch_size, num_hidden_nodes):

    W = torch.rand((num_hidden_nodes,2), dtype=torch.float32, device=device, requires_grad=True)
    W.data *= 0.1
    b1 = torch.zeros((num_hidden_nodes,1), dtype=torch.float32, device=device, requires_grad=True)

    M = torch.rand((2,num_hidden_nodes), dtype=torch.float32, device=device, requires_grad=True)
    M.data *= 0.1
    b2 = torch.zeros((2,1), dtype=torch.float32, device=device, requires_grad=True)

    W_layer = Input((num_hidden_nodes, 2))
    W_layer.set(W)
    b1_layer = Input((num_hidden_nodes,1))
    b1_layer.set(b1)

    M_layer = Input((2,num_hidden_nodes))
    M_layer.set(M)
    b2_layer = Input((2,1))
    b2_layer.set(b2)

    x1_layer = Input((x_train.shape[0], batch_size))
    linear_layer1 = LinearReLU(x1_layer, W_layer, b1_layer)

    x2_layer = Input((b1_layer.output.shape[0], batch_size))
    linear_layer2 = Linear(x2_layer, M_layer, b2_layer)


    for epoch in range(num_epochs):
        print(f'epoch #{epoch}')
        for i in range(x_train.shape[1]//batch_size):
            start_idx = i*batch_size
            end_idx = i*batch_size + batch_size
            
            x1_layer.set(x_train[:, start_idx : end_idx].reshape(x_train.shape[0], batch_size))

            linear_layer1.forward()

            x2_layer.set(linear_layer1.output)

            linear_layer2.forward()

            #(1/linear_layer2.output.numel()) * 

            mse = MSE((y_train[:, start_idx : end_idx]).reshape(y_train.shape[0], batch_size),
                     linear_layer2.output)

            s1 = (W_layer.output**2).sum()
            s2 = (M_layer.output**2).sum()
            S = reg_const*(s1 + s2)

            cost = mse + S

            cost.backward()

            with torch.no_grad():

                W_layer.output -= learning_rate * W_layer.output.grad
                b1_layer.output -= learning_rate * b1_layer.output.grad

                M_layer.output -= learning_rate * M_layer.output.grad
                b2_layer.output -= learning_rate * b2_layer.output.grad

                W_layer.output.grad.zero_()
                b1_layer.output.grad.zero_()
                M_layer.output.grad.zero_()
                b2_layer.output.grad.zero_()

        print(mse.item())
    return (linear_layer1, linear_layer2)
    
    
    
test_x1_layer = Input(x_test.shape)
test_x1_layer.set(x_test)
layer1.x = test_x1_layer

layer1.forward()

test_x2_layer = Input(layer1.output.shape)
test_x2_layer.set(layer1.output)
layer2.x = test_x2_layer

layer2.forward()

mse = MSE(y_test, layer2.output)
print(mse)

------


num_hidden_nodes = 2

W = torch.rand((num_hidden_nodes,2), dtype=torch.float32, device=device, requires_grad=True)
W.data *= 0.1
b1 = torch.zeros((num_hidden_nodes,1), dtype=torch.float32, device=device, requires_grad=True)

W2 = torch.rand((num_hidden_nodes,2), dtype=torch.float32, device=device, requires_grad=True)
W2.data *= 0.1
b3 = torch.zeros((num_hidden_nodes,1), dtype=torch.float32, device=device, requires_grad=True)

W3 = torch.rand((num_hidden_nodes,2), dtype=torch.float32, device=device, requires_grad=True)
W3.data *= 0.1
b4 = torch.zeros((num_hidden_nodes,1), dtype=torch.float32, device=device, requires_grad=True)

M = torch.rand((2,num_hidden_nodes), dtype=torch.float32, device=device, requires_grad=True)
M.data *= 0.1
b2 = torch.zeros((2,1), dtype=torch.float32, device=device, requires_grad=True)

W_layer = Input((num_hidden_nodes, 2))
W_layer.set(W)
b1_layer = Input((num_hidden_nodes,1))
b1_layer.set(b1)

W2_layer = Input((num_hidden_nodes, 2))
W2_layer.set(W2)
b3_layer = Input((num_hidden_nodes,1))
b3_layer.set(b3)

W3_layer = Input((num_hidden_nodes, 2))
W3_layer.set(W3)
b4_layer = Input((num_hidden_nodes,1))
b4_layer.set(b4)

M_layer = Input((2,num_hidden_nodes))
M_layer.set(M)
b2_layer = Input((2,1))
b2_layer.set(b2)

num_epochs = 16
learning_rate = .1
reg_const = 0
batch_size = 1

x1_layer = Input((x_train.shape[0], batch_size))
linear_layer1 = LinearReLU(x1_layer, W_layer, b1_layer)

x3_layer = Input((x_train.shape[0], batch_size))
linear_layer3 = LinearReLU(x3_layer, W2_layer, b3_layer)

x4_layer = Input((x_train.shape[0], batch_size))
linear_layer4 = LinearReLU(x4_layer, W3_layer, b4_layer)

x2_layer = Input((b1_layer.output.shape[0], batch_size))
linear_layer2 = Linear(x2_layer, M_layer, b2_layer)


for epoch in range(num_epochs):
#     learning_rate /= 10**epoch
#     print(learning_rate)
    print(f'epoch #{epoch}')
    for i in range(x_train.shape[1]//batch_size):
        x1_layer.set(x_train[:, i*batch_size : (i*batch_size + batch_size)].reshape(x_train.shape[0], batch_size))

        linear_layer1.forward()

        x3_layer.set(linear_layer1.output)

        linear_layer3.forward()
        
        x3_layer.set(linear_layer3.output)

        linear_layer4.forward()
        
        x2_layer.set(linear_layer4.output)

        linear_layer2.forward()
        
        #(1/linear_layer2.output.numel()) * 

        mse = (((y_train[:,i*batch_size : (i*batch_size + batch_size)]).reshape(y_train.shape[0], batch_size) - linear_layer2.output)**2).sum()
        
        s1 = (W_layer.output**2).sum()
        s2 = (M_layer.output**2).sum()
        S = reg_const*(s1 + s2)
        
        if mse.item() < 0.000000001:
            learning_rate /= 10
        
        cost = mse + S

        cost.backward()
        print(mse.item())

        with torch.no_grad():

            W_layer.output -= learning_rate * W_layer.output.grad
            b1_layer.output -= learning_rate * b1_layer.output.grad

            M_layer.output -= learning_rate * M_layer.output.grad
            b2_layer.output -= learning_rate * b2_layer.output.grad

            W_layer.output.grad.zero_()
            b1_layer.output.grad.zero_()
            M_layer.output.grad.zero_()
            b2_layer.output.grad.zero_()
            
            
            
    srun --partition=dgx --gpus=1 --cpus-per-gpu=16 singularity exec --nv -B /data:/data -B /data:/scratch/data /data/cs3450/pytorch21q4.4.sif python trainENetModel.py
    
    srun --partition=teaching --gpus=1 --cpus-per-gpu=2 singularity exec --nv -B /data:/data "/data/containers/msoe-tensorflow-20.07-tf2-py3.sif" /usr/local/bin/nvidia_entrypoint.sh pip uninstall tensorflow