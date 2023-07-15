# Description
Searching the best MLP Neural-Network arcitecture (NAS) to find a arcitecture which has best value for accuracy and minimum value for loss.
We use Genetic algorithms as our evolutionary algorithm to search for this arcitecture, to find best classified CFAR-10 images dataset .

*** So we have some Hyper-parameters : ***

-feature extraction network type ( which has one these values : VGG-11, ResNet-34, ResNet-18)

-Number of hidden layers of MLP arcitecture ( which has one these values : 0,1,2)

-Number of neurons in each existing hidden layer ( which has one these values : 10,20,30)

-Type of Activation function in each hidden layer ( which has one these values : Relu , Sigmoid)


*** Table of settings that we can change its values are as follows : ***

-Number of epochs (given 5)

-Number of evolutionary generations (given 10) 

-PopSize ( number of individuals in each population or generation ) (given 10)

-Number of evaluation executation to evaluate each individual (given 5).


*** Best MLP Neural network arcitecture that i'v got : ***

Number of layers : 2

Number of Neurons per layer : [20,10]

Activation function : ['relu','sigmoid']

Feature extraction network type : VGG-11

Best Accuracy :0.64 %

