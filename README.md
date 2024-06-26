# Optimx-RL -Numpy

Optimx-RL 是我受 PyTorch 启发而定制的深度强化学习框架。它是 pytroch 的简化版本，使用 numpy 从头开始​​实现了所有内容。它旨在提供更灵活和轻量级的实现，以了解深度神经网络的内部工作原理，满足线上实时更新模型的诉求。Optimx-RL 提供了深度学习框架中常见的一系列功能，包括各种层（线性层、激活层、CNN 等）、激活函数、损失函数和优化算法。该项目旨在让用户清楚地了解神经网络的功能以及如何使用 Python 和 NumPy 从头开始​​构建神经网络，同时也包含了实时特征的缓存及模型的异步存储功能。

# 主要特性和功能：

## 1. Layers：

Optimx-RL 提供不同类型的层，例如全连接层、卷积层等。每种层类型都实现为执行前向和后向传递，使用户能够创建复杂的神经网络架构。Optimx-RL 中的 Reshape 层旨在将 Conv(CNN) 层的输出转换为扁平形状，以便将数据传递到完全连接 (FCLayer) 层。
**用法：** optimxrl.Layers 包含不同类型的层，例如 FClayer、conv 层等，例如：
example:
```
optimxrl.Layers.FCLayer(input_size,output_size) or 
optimxrl.Layers.Conv(kernel_size,input_channels,output_channels) etc
```

## 2. 激活函数：

该框架包括流行的激活函数，如 ReLU、Sigmoid、Tanh、Leaky ReLU 等，允许用户将非线性引入其神经网络。请注意，我们将激活层视为单独的层，以保持模块化。

**用法：** optimxrl.ActivatoinFunctions 包含不同类型的激活函数，包括 ReLU、LeakyReLU、Sigmoid 和 Tanh。
```
optimxrl.ActivationFunctions.Sigmoid() or 
optimxrl.ActivationFunctions.ReLU() etc 
```

## 3. 网络：

我们可以使用 optimxrl 中的这个网络类创建具有不同层的神经网络。我们可以创建诸如 net=Network() 的网络，然后我们可以向网络添加层。比如 fc1、激活1、fc2、激活2 或 conv1、激活1、conv2、激活2、reshape 层、fc1、激活1 等。

## 4. 损失函数：

optimxrl 提供了各种损失函数，例如交叉熵、均方误差和二元交叉熵，这些函数用于评估训练期间模型的性能并计算每个层中每个权重的损失导数。

**用法：** optimxrl.LossFunctions 包含不同的损失函数类，每个损失函数类包含两个方法，.Loss() 方法计算预测中的错误，.backward() 方法计算每个层中每个权重的损失梯度。
示例：
```
optimxrl.LossFunctions.CrossEntropy().Loss(true_values,predictions) computes and returns overall loss or error in the predictions.
optimxrl.LossFunctions.CrossEntropy().backward(network,true_values,predictions) computes derivatives of loss function with respect to each weight in each layer.
```

## 5. 优化算法：

optimxrl 提供了不同的优化算法，如梯度下降、带动量的梯度下降、RMSprop 和 Adam，用于调整模型参数并最小化损失函数。

**用法：** optimxrl.Optimizers 包含不同的优化算法，如梯度下降、带动量的梯度下降、rmsprop 和 adam。每个优化器包含两个方法 .step() 更新权重，.zero_grad() 清除累积量（如动量等），您可以在执行 step() 后简单地 zero_grad，这样前一个 epoch 的动量就不会影响当前 epoch 的学习。

```
optimxrl.Optimizers.GradientDescent(network.layers,learning_rate) or 
optimxrl.Optimizers.Adam(network.layers,beta1,beta2,learning_rate) etc.

```
## 完整示例:

**Importing:**
```
import optimxrl
import optimxrl.nerwork.Network as Network 
from optimxrl.ActivationFunctions import Sigmoid,Tanh,ReLU,LeakyReLU,Softmax 
from optimxrl.Layers import FCLayer,Conv,Reshape 
from optimxrl.LossFunctions import BinaryCrossEntropy, CrossEntropy, MeanSquaredError
from optimxrl.Optimizers import GradientDescent, GradientDescentWithMomentum, RMSProp, Adam 

```
**Network:**
```
net=Network()
net.add_layers(Conv(kernel_size=3,input_depth=1,output_depth=5))
net.add_layers(Sigmoid())
net.add_layers(Conv(kernel_size=3,input_depth=5,output_depth=10))
net.add_layers(Sigmoid())
net.add_layers(Reshape((10,24,24),(10*24*24,1)))
net.add_layers(FCLayer(10*24*24,100))
net.add_layers(Sigmoid())
net.add_layers(FCLayer(100,10))
net.add_layers(Sigmoid())
```

**Training:**
```
learning_rate= 0.001
optimizer=GradientDescent(net.layers,learning_rate=learning_rate)
error=BinaryCrossEntropy()
epochs=5
for epoch in range(epochs):
    epoch_loss=[]
    acc=[]
    for images,labels in train_loader:
        data=images.reshape(images.shape[0],1,28,28)
        targets=to_one_hot(labels)
        data=data.numpy()
        targets=targets.numpy()
        outputs=net.predict(data)
        loss=error.Loss(targets,outputs)#returns loss
        epoch_loss.append(loss)
        acc.append(accuracy(np.argmax(outputs,axis=1),np.argmax(targets,axis=1)))
        error.backward(net,targets,outputs)#computes derivatives of loss with respect to each weight in each layer
        optimizer.step()# updates weights
        
    print(f"epoch:{epoch+1}, loss:{np.mean(epoch_loss)} , accuracy:{np.mean(acc)*100}%")
```
**output:**
```
epoch:1, loss:0.1297362714460865 , accuracy:80.28833333333333%
epoch:2, loss:0.056575114539481795 , accuracy:91.85%
epoch:3, loss:0.04424243689721972 , accuracy:93.47%
epoch:4, loss:0.03710513431695464 , accuracy:94.61333333333334%
epoch:5, loss:0.03215961753225156 , accuracy:95.37166666666667%
```
