import keras
import tensorflow as tf
import numpy as np
import torchvision.transforms as torch_transforms
from torchvision import datasets
import torch
import torchvision.models as models

# this class is our chromosome class which we make object from it --> each chrmosome refer to an NN architecture.
class NNArc:
    def __init__(self, num_layers, num_neurons_per_layer: list, activation_func: list, feature_extraction_net):
        
        #exaple to undrestand parameters better :
        #num_neurons_per_layer = 2 --> 
        #[10,20] = num_neurons_per_layer --> it means our architecture has 2 layers layer 1 has 10 neurons
        #['relu','sigmoid'] = activation_func --> activation function of layer 1 is relu

        self.num_layers = num_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        self.activation_func = activation_func
        self.feature_extraction_net = feature_extraction_net
        self.fitness_score = 0


    def evaluate_model(self):
        if self.fitness_score != 0:
            return self.fitness_score

        train_images, label_train, test_images, label_test = self.cifar10_loader()

        feature_train_images = self.extract_feature_by_network(train_images)
        feature_test_images = self.extract_feature_by_network(test_images)
        
        feature_train_images = np.array(feature_train_images)
        feature_test_images = np.array(feature_test_images)
        label_train = np.array(label_train)
        label_test = np.array(label_test)
        
        #print("size of feature_train is :",len(feature_train_images))
        #print("\n")

        model = self.create_model(feature_train_images)
        for _ in range(5) :
            #print("feature_train_images.shape :", feature_train_images.shape)
            #print("label_train.shape :",label_train.shape)
            model.fit(feature_train_images, label_train, epochs=5) #This line trains the model using the training data for 5 times. 
            _, acu = model.evaluate(feature_test_images, label_test) #This line evaluates the model on the test data for 1 time.
            self.fitness_score = self.fitness_score + acu
        
        return self.fitness_score/5

    # in this function creat a tensorflow model ( this code defines a method that creates a neural network model with customizable hidden layers and activation functions, and it is ready to be trained on a multi-class classification problem 
    def create_model(self, feature_train):

        model = tf.keras.Sequential()
        #input layer :
        feature_shape = np.array(feature_train).shape[1] #.shape give me a toupl which is like (100,512) with using .shape[1] i choos 512 to give it as a list to input 
        model.add(keras.Input(shape=(feature_shape))) # --> we must tell it that how many neurons has input of our network
        #       it depends on what kind of extraction network we use (resnet18 , ...)and how many features it gives us as output
                
        #hidden layer :
        for i in range(self.num_layers):
            model.add(keras.layers.Dense(self.num_neurons_per_layer[i], self.activation_func[i]))

        #output layer :
        model.add(keras.layers.Dense(10, activation='softmax'))
        #compile the model :
        model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
                   
        return model


    def extract_feature_by_network(self, images):
        #i could use : self.feature_extraction_net(pretraind=True) when i was using objects instead of string for feature_extraction_net
        selected_network = self.feature_extraction_net
    
        if selected_network == "Resnet_34":
        # Extract features using Resnet_34
            #features = extract_features_resnet_34(images)
            images = torch.stack(images)
            model = models.resnet34(pretrained=True)

            for param in model.parameters():
                param.requires_grad = False
            modules = list(model.children())[:-1]
            model = torch.nn.Sequential(*modules)
            features = model(images).numpy()

            features = features.reshape(features.shape[0], -1)

        elif selected_network == "Resnet_18":
        # Extract features using Resnet_18
            #features = extract_features_resnet_18(images)
            images = torch.stack(images)
            model = models.resnet18(pretrained=True)

            for param in model.parameters():
                param.requires_grad = False
            modules = list(model.children())[:-1]
            model = torch.nn.Sequential(*modules)
            features = model(images).numpy()

            features = features.reshape(features.shape[0], -1)
        
        elif selected_network == "Vgg_11":
        # Extract features using Vgg_11
            #features = extract_features_vgg_11(images)
            images = torch.stack(images)
            model = models.vgg11(pretrained=True)

            for param in model.parameters():
                param.requires_grad = False
            modules = list(model.children())[:-1]
            model = torch.nn.Sequential(*modules)
            features = model(images).numpy()

            features = features.reshape(features.shape[0], -1)

        return features  # features is a NumPy array of the extracted features from the input images
    
    def cifar10_loader(self):
        transform = torch_transforms.Compose([torch_transforms.ToTensor(),torch_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        train_set = datasets.CIFAR10('data', train=True,download=True, transform=transform)
                                     
        test_set = datasets.CIFAR10('data', train=False,download=True, transform=transform)
                                    
        train_images, label_train = self.x_and_y(train_set)
        test_images, label_test = self.x_and_y(test_set)

        return train_images, label_train, test_images, label_test

    def x_and_y(self, data):
        x = []
        y = []
        #c=0
        for image, label in data:
            x.append(image)
            y.append(label)
            #c=c+1
            #if c==500: #this means that we take just 100 images from CFAR10 ( to meke debuge easier ! )
            #    break
        return x, y