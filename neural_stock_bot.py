import numpy as np
import scipy.special as ss

# neural network class definition
class StockBot:

    # initialise the neural network
    def __init__(self, hiddennodes, nhiddenlayers, nhiddenoutputs, learningrate):
        # set number of nodes in each input, hidden, and output layer
        self.inodes = 5 + nhiddenoutputs
        self.hnodes = hiddennodes
        self.onodes = 5 + nhiddenoutputs

        # number of hidden layers & hidden in/outputs
        self.nhlayers = nhiddenlayers
        self.nhout = nhiddenoutputs

        # initial weight matricies
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.whh = [np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.hnodes)) for i in range(self.nhlayers - 1)]
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.wm = [self.wih] + self.whh + [self.who]

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: ss.expit(x)

        pass

    # train the neural network
    def train(self, inputs_list):

        next_hinputs = [0 for i in range(self.nhout)]
        for dat in range(0, len(inputs_list) - 1):
            # Cycles through two inputs in the time series, using backprop into the prior time element
            two_cycle_outputs = []
            two_cycle_targets = []
            
            for two_cycle in range(2):
                # append 0 * the number of hidden inputs to the input list, next_hinputs changes depending on time-step
                inputs = np.array(inputs_list[dat + two_cycle] + next_hinputs).T
                # output list for all layers to be used in backprop
                output_list = [inputs]
                # target for current time-step
                targets = np.array(inputs_list[dat + two_cycle + 1])
                # stores the targets for both time-steps considered
                two_cycle_targets.insert(0, targets)
                
                # calculates the first inputs for the first hidden layer
                hidden_inputs = np.dot(self.wih, inputs)
                hidden_outputs = self.activation_function(hidden_inputs)
                output_list.append(hidden_outputs)
                
                # loops through each layer, calculating the outputs for each and storing them
                current_layer = 0
                while current_layer < len(self.wm):
                    # calculates the inputs for the next layer, then the outputs through the activation function
                    hidden_inputs = np.dot(self.wm[current_layer], output_list[current_layer])
                    hidden_outputs = self.activation_function(hidden_inputs)

                    # appends the new outputs to the output list to be used for the next layer calculation
                    output_list.append(hidden_outputs)
                    # increments the layer pointer
                    current_layer += 1

                # the whole output list is appended to a more global storage to be used in backprop later
                two_cycle_outputs.insert(0, output_list)

                # the hidden outputs from the first time step are fed into the hidden inputs for the second
                if two_cycle == 0: 
                    next_hinputs = output_list[-1][-self.nhout :]
                    continue               

            # Multi-timestep backprop
            for two_cycle in range(2):
                if two_cycle == 0:
                    # second step final output
                    final_output = two_cycle_outputs[two_cycle][-1]
                    # targets, is second step targets and final outputs of the hidden variables, so error is 0 for these 
                    targets = two_cycle_targets[two_cycle] + final_output[-self.nhout :]
                    targets = targets.T
                    output_errors = targets - final_output
                else:
                    # output errors for first step come from the prior step
                    output_errors = two_cycle_targets[two_cycle] + current_errors[-self.nhout :]
                    
                # current errors variable to make the method more clear
                current_errors = output_errors
                current_layer = 1
                while current_layer < (len(self.wm) - 1):
                    # calculate the hidden errors
                    hidden_errors = np.dot(self.wm[-current_layer].T, current_errors)
                    #prior_errors = hidden_errors

                    # current & previous outputs used in updating the weights matricies
                    current_outputs = two_cycle_outputs[two_cycle][-current_layer]
                    prev_outputs = two_cycle_outputs[two_cycle][-(current_layer + 1)]
                    # updating the current weight matrix
                    self.wm[-current_layer] += self.lr * np.dot((current_errors * current_outputs * (1.0 - current_outputs)),
                                                                np.transpose(prev_outputs))

                    # current errors on the next loop are the current hidden errors
                    current_errors = hidden_errors
                    # increment the current_layer pointer
                    current_layer += 1
            
        pass

     
    # query the neural network
    def predict(self, inputs_list, stepsbehind, stepsahead):
        # list of predictions made given parameters
        predictions = []
        # only making predictions with stepsbehind steps, but also need steps ahead true values for comparisons
        inputs_list = inputs_list[-(stepsbehind + stepsahead) :]

        next_hinputs = [0 for i in range(self.nhout)]
        for dat in range(0, len(inputs_list) - stepsahead):    

            if dat < (len(inputs_list) - stepsahead):     
                # append 0 * the number of hidden inputs to the input list, next_hinputs changes depending on time-step
                inputs = np.array(inputs_list[dat] + next_hinputs).T
            else:
                inputs = np.array(next_hinputs).T
            
            # calculates the first inputs for the first hidden layer
            hidden_inputs = np.dot(self.wih, inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            
            # loops through each layer, calculating the outputs for each and storing them
            current_layer = 0
            while current_layer < len(self.wm):
                # calculates the inputs for the next layer, then the outputs through the activation function
                hidden_inputs = np.dot(self.wm[current_layer], hidden_outputs[current_layer])
                hidden_outputs = self.activation_function(hidden_inputs)

                # increments the layer pointer
                current_layer += 1

            if dat < (len(inputs_list) - stepsahead):
                # the hidden outputs from the first time step are fed into the hidden inputs for the second
                next_hinputs = hidden_outputs[-self.nhout :]
            else:
                # prediction steps utilise previous predictions
                next_hinputs = hidden_outputs
                predictions.append(next_hinputs[: -self.nhout])

        return predictions






