# Part 1: EVA6 Backpropagation in Microsoft Excel
Backpropagation of a Dense Network, implemented in MS Excel as a part of EVA 6 course from The School of AI

Artificial Neural Networks (ANN) are a class of computing models that are designed to mimic the working of a human brain. The most fundamental implementation of ANN is a Fully Connected Network. The FC network consists of multiple layers viz. a. Input layer - where the the independent variables required for modelling are passed as input, b. Hidden layers (one or more) which work on the input variables to extract features, and finally c. Output layer that gives the prediction of the model.

Each layer consists of a set of nodes, known as "Neurons". Each neurons in any given layer is connected to the entire set of neurons in the following layer and hence the name - "Fully Connected Network". Consider a layer containing "n" neurons - each of the "n" neurons in this layer establishes connections to all the "m" neurons in the next layer, resulting in a total of "n * m" connections. Each connections is assigned a weight, that decides how much of the output from the neuron in the preceeding layer (Layer "l-1") should pass to the next layer (Layer "l"). All the incoming connections to a neuron are then summed up, resulting in a weighted sum of the inputs. An activation is applied on this weighted sum, which is called the "Activation Output" of the neuron. The activations that are most commonly used are ReLU (Rectified Linear Unit), Sigmoid and Tanh. 

Figure 1 below illustrates a simple Fully Connected Network, which has 1 input layer, 1 hidden layer and 1 output layer. There are 2 inputs (i1, i2) in the input layer, which is then connected to a hidden layer with 2 neurons. Each of these 2 neurons in the hidden layer (h1, h2) receive a weighted sum of the input values from the input layer. These values undergo activation (a_h1, a_h2) and are then further fully connected as weighted sum to the neurons in the output layer (o1, o2). The output layer then applies an activation (Sigmoid, in this case) and gives out the final predictions (a_o1, a_o2).

![image](https://user-images.githubusercontent.com/71654199/119672619-5fe25500-be58-11eb-82da-f705dd3f9eab.png)
> Figure 1: An example of a Fully Connected Network

The predictions a_o1, a_o2 for the given input values i1, i2 are then compared with the actual target values t1, t2. In this example, we use the "Mean Squared Error" as the function to quantify the loss (difference between actual and predicted values).

Loss / "Mean Squared Error" from output a_o1 = E1 = (1/2) * (t1 - a_o1)^2  
Loss / "Mean Squared Error" from output a_o2 = E2 = (1/2) * (t2 - a_o2)^2

Total Loss = E1 + E2

![image](https://user-images.githubusercontent.com/71654199/119672750-78526f80-be58-11eb-872d-83079269d1aa.png)

∂E_total/∂w5 = (a_o1 - t1) * a_o1 * (1 - a_o1) * a_h1  
∂E_total/∂w6 = (a_o1 - t1) * a_o1 * (1 - a_o1) * a_h2  
∂E_total/∂w7 = (a_o2 - t2) * a_o2 * (1 - a_o2) * a_h1  
∂E_total/∂w8 = (a_o2 - t2) * a_o2 * (1 - a_o2) * a_h2

∂E_total/∂w1 = ((a_o1 - t1)*a_o1*(1 - a_o1)*w5 + (a_o2 - t2)*a_o2*(1 - a_o2)*w7) * a_h1 * (1 - a_h1) * i1  
∂E_total/∂w2 = ((a_o1 - t1)*a_o1*(1 - a_o1)*w5 + (a_o2 - t2)*a_o2*(1 - a_o2)*w7) * a_h1 * (1 - a_h1) * i2  
∂E_total/∂w3 = ((a_o1 - t1)*a_o1*(1 - a_o1)*w6 + (a_o2 - t2)*a_o2*(1 - a_o2)*w8) * a_h2 * (1 - a_h2) * i1  
∂E_total/∂w4 = ((a_o1 - t1)*a_o1*(1 - a_o1)*w6 + (a_o2 - t2)*a_o2*(1 - a_o2)*w8) * a_h2 * (1 - a_h2) * i2

![image](https://user-images.githubusercontent.com/71654199/119672818-86a08b80-be58-11eb-91f3-7200fe9d0f2b.png)

![image](https://user-images.githubusercontent.com/71654199/119672983-a59f1d80-be58-11eb-939a-58219cd3fbea.png)

![image](https://user-images.githubusercontent.com/71654199/119671879-c31fb780-be57-11eb-9c09-4479afd09594.png)

