import random
from Arc import NNArc
import copy

num_neurons = [10,20,30]
feature_extraction_networks = ["Vgg_11","Resnet_34","Resnet_18"]
activation_functions = ['sigmoid', 'relu']
num_hidden_layers = [0,1,2]

class EvolutioaryAlgo:
    def __init__(self, mutation_rate, num_generations, pop_size, elite_percentage):
        self.mutation_rate = mutation_rate
        self.population = None
        self.num_generations = num_generations
        self.pop_size = pop_size                # the number of chromosoms of population
        self.elite_percentage = elite_percentage

    def run(self):
        self.population = self.generate_population()
        bestChromosomsOfAll = []
        for i in range(self.num_generations):
            for arc in self.population:
                 arc.fitness_score = arc.evaluate_model()
            """
            chromCounter = 1
            for arc in self.population:
                print("in generation number :",i)
                print("chromosom number :",chromCounter)
                print(arc.feature_extraction_net)
                print(arc.fitness_score)
                chromCounter=chromCounter+1
            """    
            best_chromosomes = self.select_parents(self.population) # choosing best of chromosoms based on their fitness 
            next_generation = best_chromosomes.copy() # copy best of chromosoms to next_generation list directly (based on elit_percentage number)
            counter=0
            while len(next_generation) <= self.pop_size: # self.pop_size - 1 is because i want to choose one of chromosoms randomly between non_elit_chrmosoms and elit_chrmosoms to add to next_generation list
                parent1 = random.choice(self.population)  #choosing parents from best_chromosomes to mutate and recom to make new chromosoms to add to next_generation list
                parent2 = random.choice(self.population) 
                if parent1.num_layers != 0 and parent2.num_layers != 0: # it means only the parents which have 1 or more layers can go for recombination
                    child1,child2 = self.crossover(copy.deepcopy(parent1), copy.deepcopy(parent2))
                    mutated_child1 = self.mutate(child1)
                    mutated_child2 = self.mutate(child2)
                    next_generation.append(mutated_child1)
                    next_generation.append(mutated_child2)
                    counter=0
                counter=counter+1
                if counter==(len(self.population)*len(self.population)): # for when all the chromosoms have 0 layers.
                    for _ in range(2):
                        num_layers = random.choice(num_hidden_layers)
                        num_neurons_per_layer = []
                        activation_func = []
                        feature_extraction_net = random.choice(feature_extraction_networks)
                        if num_layers != 0:
                            for i in range(num_layers):
                                activation_func.append(random.choice(activation_functions))
                                num_neurons_per_layer.append(random.choice(num_neurons))
                            chromosome = NNArc(num_layers, num_neurons_per_layer, activation_func, feature_extraction_net)
                            self.population.pop()
                            self.population.append(chromosome)
                        elif num_layers == 0 : #do not loose chance of 0 layer. 
                            chromosome = NNArc(num_layers, num_neurons_per_layer , activation_func, feature_extraction_net)
                            self.population.pop()
                            self.population.append(chromosome)   
            self.population=next_generation
            bestOfThisGeneration =  max(self.population, key=lambda x: x.fitness_score)
            bestChromosomsOfAll.append(bestOfThisGeneration) #workde with out using copy.deepcopy
        return bestChromosomsOfAll

    def mutate(self, chromosome):
        mutated_chromosome = chromosome
        if chromosome.num_layers == 0:
            num_neurons_per_layer=[]
            activation_func=[]
            feature_extraction_net = random.choice(feature_extraction_networks)
            new_num_layers = random.choice(num_hidden_layers)
            if(new_num_layers != 0):
                for i in range(new_num_layers):
                    activation_func.append(random.choice(activation_functions))
                    num_neurons_per_layer.append(random.choice(num_neurons))
            mutated_chromosome = NNArc(new_num_layers, num_neurons_per_layer , activation_func, feature_extraction_net)
            return mutated_chromosome
        for i in range(chromosome.num_layers):
            if random.random() < self.mutation_rate: 
                mutated_chromosome.num_neurons_per_layer[i] = random.choice(num_neurons)
            if random.random() < self.mutation_rate:
                mutated_chromosome.activation_func[i] = random.choice(activation_functions)
            if random.random() < self.mutation_rate:
                mutated_chromosome.feature_extraction_net = random.choice(feature_extraction_networks)
        return mutated_chromosome


    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, min(parent1.num_layers, parent2.num_layers)-1) # crossover_point is our child new num_layers
        num_neurons_per_layer = 0
        
        num_neurons_per_layer = parent1.num_neurons_per_layer[crossover_point]
        parent1.num_neurons_per_layer[crossover_point] = parent2.num_neurons_per_layer[crossover_point]
        parent2.num_neurons_per_layer[crossover_point] = num_neurons_per_layer 
        
        temp_activation_func = parent1.activation_func[crossover_point]
        parent1.activation_func[crossover_point]=parent2.activation_func[crossover_point]
        parent2.activation_func[crossover_point]=temp_activation_func

        feature_extraction_net = parent1.feature_extraction_net if random.random() < 0.5 else parent2.feature_extraction_net
        child1 = NNArc(parent1.num_layers, parent1.num_neurons_per_layer, parent1.activation_func, feature_extraction_net)
        
        feature_extraction_net = parent1.feature_extraction_net if random.random() < 0.5 else parent2.feature_extraction_net
        child2 = NNArc(parent2.num_layers, parent2.num_neurons_per_layer, parent2.activation_func, feature_extraction_net)
        return child1,child2

    # Select the best-performing chromosomes as parents for the next generation
    def select_parents(self, population):
        sorted_population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
        elite_size = int(self.elite_percentage * len(population))  # elite_size is number of chromosomes we pick up directly for next generation
        elite_chromosomes = sorted_population[:elite_size]
        return elite_chromosomes

    # Generate the initial population
    def generate_population(self):
        population = []

        for _ in range(self.pop_size):

            num_layers = random.choice(num_hidden_layers)
            num_neurons_per_layer = []
            activation_func = []
            feature_extraction_net = random.choice(feature_extraction_networks)
            if num_layers != 0:
                for i in range(num_layers):
                    activation_func.append(random.choice(activation_functions))
                    num_neurons_per_layer.append(random.choice(num_neurons))
                chromosome = NNArc(num_layers, num_neurons_per_layer, activation_func, feature_extraction_net)
                population.append(chromosome)
            elif num_layers == 0 :
                chromosome = NNArc(num_layers, num_neurons_per_layer , activation_func, feature_extraction_net)
                population.append(chromosome)
        return population


    def show_result(self,bestChromosomsOfAll):
        if not self.population:
            print("No individuals in the population.")
            return

        # Filter out None elements from the population
        filtered_population = [individual for individual in self.population if individual is not None]

        if not filtered_population:
            print("All individuals in the population are None.")
            return

        best_chromosome = max( bestChromosomsOfAll , key=lambda x: x.fitness_score)

        best_accuracy = best_chromosome.fitness_score

        print("Best MLP Neural Network Architecture:")
        print("Number of Layers:", best_chromosome.num_layers)
        print("Number of Neurons per Layer:", best_chromosome.num_neurons_per_layer)
        print("Activation Function:", best_chromosome.activation_func)
        print("Feature Extraction Network:", best_chromosome.feature_extraction_net)
        print("Accuracy:", best_accuracy)
