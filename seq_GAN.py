import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"


import tensorflow as tf
from tensorflow.io import parse_tensor

from tfrecord.torch.dataset import TFRecordDataset

import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from allennlp.commands.elmo import ElmoEmbedder

import numpy as np


from os import listdir
from os.path import isfile, join


embeddings_path = "./rhodopsins_embeddings"
serialized_tensors_seqvec = './rhodopsins_embeddings/'
onlyfiles = [f for f in listdir(embeddings_path) if isfile(join(embeddings_path, f))]


vocabulary = ["R", "H", "K", "D", "E", "C", "U", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W", "S", "T", "N", "Q"]

idx_to_char = {}
for idx, char in enumerate(vocabulary):
    idx_to_char[idx] = char


class Generator(nn.ModuleList):
    def __init__(self):
        super(Generator, self).__init__()

        self.batch_size = 1
        self.hidden_dim = 128 #dimension for noise
        self.input_size = len(vocabulary)
        self.num_classes = len(vocabulary)
        
        self.num_layers = 3
        
        # Dropout
        self.dropout = nn.Dropout(0.25)
        
        # Bi-LSTM
        # Forward and backward
        
        
        #self.lstm_cell_forward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        #self.lstm_cell_backward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        
        # LSTM layer
        self.lstm_cell = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers)
        
        # Linear layer
        self.linear = nn.Linear(self.hidden_dim, self.num_classes)
        
    def forward(self, noise, prev_state):
    
        # Bi-LSTM
        # hs = [batch_size x hidden_size]
        # cs = [batch_size x hidden_size]
        
        '''
        hs_forward = torch.zeros(x.size(0), self.hidden_dim)
        cs_forward = torch.zeros(x.size(0), self.hidden_dim)
        hs_backward = torch.zeros(x.size(0), self.hidden_dim)
        cs_backward = torch.zeros(x.size(0), self.hidden_dim)
        
        # LSTM
        # hs = [batch_size x (hidden_size * 2)]
        # cs = [batch_size x (hidden_size * 2)]
        hs_lstm = torch.zeros(x.size(0), self.hidden_dim * 2)
        cs_lstm = torch.zeros(x.size(0), self.hidden_dim * 2)
    
        # Weights initialization
        torch.nn.init.kaiming_normal_(hs_forward)
        torch.nn.init.kaiming_normal_(cs_forward)
        torch.nn.init.kaiming_normal_(hs_backward)
        torch.nn.init.kaiming_normal_(cs_backward)
        torch.nn.init.kaiming_normal_(hs_lstm)
        torch.nn.init.kaiming_normal_(cs_lstm)
        
        forward = []
        backward = []
        
        # Unfolding Bi-LSTM
        # Forward
        hs_forward, cs_forward = self.lstm_cell_forward(out[i], (hs_forward, cs_forward))
        forward.append(hs_forward)
             
        # Backward
        for i in reversed(range(self.sequence_len)):
            hs_backward, cs_backward = self.lstm_cell_backward(out[i], (hs_backward, cs_backward))
            backward.append(hs_backward)
        '''

        # LSTM
        #for fwd, bwd in zip(forward, backward):
            #input_tensor = torch.cat((fwd, bwd), 1)

        noise = noise.unsqueeze(0)

        output, state = self.lstm_cell(noise, prev_state)
        logits = self.linear(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_dim),
                torch.zeros(self.num_layers, sequence_length, self.hidden_dim))


gen = Generator()


def generator(model, noise_vector, idx_to_char, n_chars):
    
    softmax = nn.Softmax(dim=0)
    
    model.eval()
    full_prediction = []
    state_h, state_c = model.init_state(1)
    
      # number of characters
    for i in range(n_chars):
        
        y_pred, (state_h, state_c) = model(noise, (state_h, state_c))
        # It is applied the softmax function to the predicted tensor
        prediction = softmax(y_pred.view(-1))
        

        # The prediction tensor is transformed into a numpy array
        prediction = prediction.squeeze().detach().numpy()
        # It is taken the idx with the highest probability
        arg_max = np.argmax(prediction)

        # The full prediction is saved
        full_prediction = np.append(full_prediction, arg_max)
        
    string_prediction = ''.join([idx_to_char[value] for value in full_prediction])
    
    return string_prediction

    #print("Prediction: \n")
    #print(''.join([idx_to_char[value] for value in full_prediction]), "\"")


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, protein_vector):
        validity = self.model(protein_vector)
        return validity



adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
#generator = Generator()
discriminator = Discriminator()
  #  generator.cuda()
discriminator.cuda()
adversarial_loss.cuda()


# In[18]:


discriminator(protein_embd.cuda())


# # Training


def generate_noise(batch_size = 1, noise_dimension = 200, device=None):
    """ Generate noise for number_of_images images, with a specific noise_dimension """
    return torch.randn(batch_size, noise_dimension, device=device)



# Batch size
BATCH_SIZE = 1

# (Lenght of Protein) x (Dimension of SeqVec embeddings) 400x1024 
L = 400
embedding_dim = 1024
clades_num = 26


serialized_tensors = serialized_tensors_seqvec

filenames = [serialized_tensors + '/' + file for file in os.listdir(serialized_tensors) if file != ".ipynb_checkpoints"]
raw_dataset = tf.data.TFRecordDataset(filenames)

def parse_example(example):
    example_proto = tf.train.Example()
    example_proto.ParseFromString(example.numpy())
    features = example_proto.features.feature
    label = parse_tensor(features['label'].bytes_list.value[0], out_type = tf.float32)
    tensor = parse_tensor(features['tensor'].bytes_list.value[0], out_type = tf.float32)
    return tf.argmax(label), tf.reduce_mean(tensor, axis=0)

def parsed_dataset_generator():
    for i in raw_dataset:
        yield parse_example(i)

parsed_dataset = tf.data.Dataset.from_generator(
     parsed_dataset_generator,
     (tf.int64, tf.float32),
     (tf.TensorShape([]), tf.TensorShape([embedding_dim, ]))).batch(BATCH_SIZE)

tf.executing_eagerly()



class EmbeddingsDataset(torch.utils.data.Dataset):

    def __init__(self, parsed_dataset):
        self.embeddings = []
        count = 0
        for x, y in parsed_dataset:
            count += 1
            embedding = torch.as_tensor(y.numpy())
            self.embeddings.append(embedding)
        print(count)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        sample = self.embeddings[idx].reshape(-1)
        
        return sample



dataset = EmbeddingsDataset(parsed_dataset)


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4)



def generate_even_data(max_int: int, batch_size: int=16) -> Tuple[List[int], List[List[int]]]:
    # Get the number of binary places needed to represent the maximum number
    max_length = int(math.log(max_int, 2))

    # Sample batch_size number of integers in range 0-max_int
    sampled_integers = np.random.randint(0, int(max_int / 2), batch_size)

    # create a list of labels all ones because all numbers are even
    labels = [1] * batch_size

    # Generate a list of binary numbers for training.
    data = [create_binary_list_from_int(int(x * 2)) for x in sampled_integers]
    data = [([0] * (max_length - len(x))) + x for x in data]

    return labels, data


batch_size = 16
training_steps = 500
input_length = int(math.log(max_int, 2))

# Models
generator = Generator()
discriminator = Discriminator()

# Optimizers
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# loss
loss = nn.BCECrossEntropyLoss()

for i in range(training_steps):
    # zero the gradients on each iteration
    generator_optimizer.zero_grad()

    # Create noisy input for generator
    # Need float type instead of int
    noises = []
    for i in range(4):
        noises.append((torch.rand(1, 128) - 0.5) / 0.5)

    b = torch.Tensor(4, 128)
    noises = torch.cat(noises, out=b)


    softmax = nn.Softmax(dim=0)

    model.train()
    full_prediction = []
    state_h, state_c = model.init_state(1)

    # number of characters
    for i in range(n_chars):

        y_pred, (state_h, state_c) = model(noise, (state_h, state_c))
        # It is applied the softmax function to the predicted tensor
        prediction = softmax(y_pred.view(-1))


        # The prediction tensor is transformed into a numpy array
        prediction = prediction.squeeze().detach().numpy()
        # It is taken the idx with the highest probability
        arg_max = np.argmax(prediction)

        # The full prediction is saved
        full_prediction = np.append(full_prediction, arg_max)

    string_prediction = ''.join([idx_to_char[value] for value in full_prediction])


    generated_data = generator(noise)

    # Generate examples of even real data
    true_labels, true_data = generate_even_data(max_int, batch_size=batch_size)
    true_labels = torch.tensor(true_labels).float()
    true_data = torch.tensor(true_data).float()

    # Train the generator
    # We invert the labels here and don't train the discriminator because we want the generator
    # to make things the discriminator classifies as true.
    generator_discriminator_out = discriminator(generated_data)
    generator_loss = loss(generator_discriminator_out, true_labels)
    generator_loss.backward()
    generator_optimizer.step()

    # Train the discriminator on the true/generated data
    discriminator_optimizer.zero_grad()
    true_discriminator_out = discriminator(true_data)
    true_discriminator_loss = loss(true_discriminator_out, true_labels)

    # add .detach() here think about this
    generator_discriminator_out = discriminator(generated_data.detach())
    generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size))
    discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
    discriminator_loss.backward()
    discriminator_optimizer.step()
