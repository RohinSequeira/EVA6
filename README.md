# EVA6 Backpropagation in Microsoft Excel
Backpropagation of a Dense Network, implemented in MS Excel as a part of EVA 6 course from The School of AI

An Artificial Neural Networks (ANN) are a class of computing models that are designed to mimic the working of a human brain. The most fundamental implementation of ANN is a Fully Connected Network. The FC network consists of multiple layers viz. a. Input layer - where in the the parameters required for modelling are passed as input, b. Hidden layers (one or more) which work on the input parameters to extract feature, and finally c. Output layer that gives the prediction of the model.

Each layer consists of a set of nodes, known as "Neurons". Each neurons in any given layer is connected to the entire set of neurons in the following layer and hence the network gets the name - "Fully Connected Network". A layer containing "n" neurons connectis to all the "m" neurons in the next layer, resulting in "n * m" connections. Each of these connections is assigned a weight, that decides how much of the 

Figure 1 below illustrates a simple Fully Connected Network

![image](https://user-images.githubusercontent.com/71654199/119672619-5fe25500-be58-11eb-82da-f705dd3f9eab.png)
                            Figure 1: An example of a Fully Connected Network

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

