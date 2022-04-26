import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        caption_embeds = self.word_embeddings(captions[:,:-1])
        features = torch.unsqueeze(features, 1)
        lstm_output, self.hidden = self.lstm(torch.cat((features, caption_embeds), 1))
        lstm_output = self.dropout(lstm_output)
        scores = self.fc(lstm_output)
        
        return scores
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        max_score_ind = -1
        list_of_inds = []
        while True:
        
            lstm_output, self.hidden = self.lstm(inputs)
            scores = self.fc(lstm_output)
            max_score_inds = torch.argmax(scores, 2)
            cur_word = max_score_inds[0, -1]
            list_of_inds.append(cur_word.item())
            cur_word_embed = self.word_embeddings(cur_word)
            inputs = torch.cat((inputs, cur_word_embed.unsqueeze(0).unsqueeze(0)), 1)
            
            if cur_word.item() == 1 or len(list_of_inds) >= max_len:
                break
        
        return list_of_inds
        