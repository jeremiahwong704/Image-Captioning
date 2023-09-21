import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
       
        

        #Embedded layer: transforms each input word into a vector of a certain shape before being fed as input to RNN
        self.embedded_layer =  nn.Embedding(self.vocab_size, self.embed_size)
        
        #LSTM layer
        self.LSTM = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, dropout = 0, batch_first = True)
        
        #Fully connected layer
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, features, captions):
        features = features.clone()
        embed = self.embedded_layer(captions[:,:-1]) #Don't need the last word, as there is no next word to predict from it
        
        inputs = torch.cat((features.unsqueeze(1), embed), 1) #Concatonate features with embed
        #Inputs shape is now (batch_size, 1, embed_size)
        
        lstm_out, _ = self.LSTM(inputs)
        #Now, we go from embed size to hidden size, giving us 
        #(batch_size, max_caption_length, hidden_size)
           
        x = self.fc(lstm_out)
        #finally, shape is (batch_size, max_caption_length, vocab_size)
        
        return x

    def sample(self, inputs, states=None, max_len=20):
        # Initialize a list to store the predicted word ids
        sampled_ids = []

        # If initial states are not provided, initialize them as zero tensors
        if states is None:
            states = (torch.zeros(self.num_layers, inputs.size(0), self.hidden_size).to(inputs.device),
                      torch.zeros(self.num_layers, inputs.size(0), self.hidden_size).to(inputs.device))

        # Iterate for `max_len` time steps
        for i in range(max_len):
            # Pass the inputs through the LSTM layer to get LSTM outputs and new states
            lstm_out, states = self.LSTM(inputs, states)

            # Map LSTM outputs to word scores using the fully connected layer
            outputs = self.fc(lstm_out.squeeze(1))

            # Get the index of the maximum value (predicted word) for the current time step
            _, predicted = outputs.max(1)

            # Append the predicted word index to the sampled_ids list
            sampled_ids.append(predicted.item())

            # Prepare the inputs for the next time step (use the predicted word as the input)
            inputs = self.embedded_layer(predicted).unsqueeze(1)

        return sampled_ids
    
    
    
    
    

    
    
    
    
    
    