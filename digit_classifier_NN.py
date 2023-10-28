
import torch as torch
torch.set_num_threads(2)
import torchvision
import numpy as np
import struct
import time
from PIL import Image
from torch import nn, optim
from multiprocessing import Process, Queue

fp_image_train = "data/archive/train-images-idx3-ubyte/train-images-idx3-ubyte"
fp_label_train = "data/archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
fp_image_test = "data/archive/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
fp_label_test = "data/archive/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"

MLP_model_fp = "models/mpl_model_fp"
MLP_model_fp_2 = "models/mpl_model_fp_2"
MLP_model_fp_3 = "models/mpl_model_fp_3"

relu = nn.ReLU()


def get_keys(L, input_nodes=True):
    keys = [f'layers.{2*k}.weight' for k in range(L-1)]
    if input_nodes:
        return keys 
    return keys[1:]

def get_L_from_state_dict(state_dict):
    L = 0
    keys = state_dict.keys()
    while f'layers.{2*L}.weight' in keys:
        L += 1
    return L + 1

def get_s_from_state_dict(state_dict):

    L = get_L_from_state_dict(state_dict)
    keys = get_keys(L)
    s = list(state_dict[keys[0]].shape[::-1])
    for l in range(1,L-1):
        s.append(state_dict[keys[l]].shape[0])
    return s

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class MLP(nn.Module):
    """
    Multi-Layer-Perceptron (MLP) model for classifying MNIST hand-drawn numbers.
    """
    def __init__(self, state_dict=None, s=None):
        super().__init__()

        if state_dict is not None:
            s = get_s_from_state_dict(state_dict)
        elif s is None:
            raise Exception("No network structure provided to MLP.__init__")
        layers = nn.Sequential()

        L = len(s)
        for l in range(L-2):
            layers.append(nn.Linear(s[l], s[l+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(s[L-2], s[L-1]))
        

        layers_1 = nn.Sequential(
            nn.Linear(28*28, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 10)
            )
        
        layers_2 = nn.Sequential(
            nn.Linear(28*28, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
            )
        
        self.layers = layers
    
    @staticmethod
    def flatten_x(x):
        return x.view(-1, 28*28)

    def forward(self, x):
        x = self.flatten_x(x)
        return self.layers(x)
    
    def predict(self, x):
        x = self.flatten_x(x)
        return nn.Softmax(dim=1)(self.layers(x))


    def save(self, fp=MLP_model_fp):
        print(f"Saving model to {fp}")
        torch.save(self.state_dict(), fp)

    @staticmethod
    def load(fp = MLP_model_fp):
        state_dict = torch.load(fp)
        model = MLP(state_dict)
        model.load_state_dict(state_dict)
        return model

    def get_node_vals(self, x, input_nodes):
        if x is None:
            return None
        x = self.flatten_x(image_preprocess(x))
        node_vals = [x] if input_nodes else []
        for k, layer in enumerate(self.layers):
            x = layer(x)
            if k%2==1:
                node_vals.append(np.array(x.detach()))
        node_vals.append(np.array(relu(x).detach()))
        return node_vals


class CNN(nn.Module):
    """
    Convolutional-Neural-Network (CNN) model for classifying MNIST hand-drawn numbers.
     """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=15)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=8)



def load_mnist_images(filename):
    with open(filename, "rb") as f:
        # Read the magic number and number of images
        magic, num_images = struct.unpack(">II", f.read(8))
        if magic != 2051:
            raise ValueError("Not a MNIST image file!")

        # Read image dimensions
        num_rows, num_cols = struct.unpack(">II", f.read(8))

        # Read image data
        data = f.read()
        images = np.frombuffer(data, dtype=np.uint8).reshape(num_images, num_rows, num_cols)
    return torch.tensor(images)

def image_preprocess(images):
    return (images/255).to(torch.float32)


def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        # Read the magic number and number of labels
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Not a MNIST label file!")

        # Read label data
        data = f.read()
        labels = np.frombuffer(data, dtype=np.uint8)
    return torch.tensor(labels)

train_images = load_mnist_images(fp_image_train)
train_labels = load_mnist_labels(fp_label_train)
test_images = load_mnist_images(fp_image_test)
test_labels = load_mnist_labels(fp_label_test)



def get_dataloader(images, labels, batch_size):

    dataset = CustomDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def visualise_image(image_arr):
    """
    Visualise 28x28 image represented by numpy array.
    """
    image = Image.fromarray(image_arr)
    return image

def check_accuracy(model, images, labels):
    correctly_classified = torch.max(model(images), axis=1)[1] == labels
    accuracy = torch.sum(correctly_classified.to(int)) / len(correctly_classified)
    print(f"Accuracy = {accuracy*100:.2f}%")
    return accuracy.item()

def train_NN(model, n_epochs=1, batch_size=8, lr=0.1, queue=None):
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(), lr=lr)

    train_images = image_preprocess(load_mnist_images(fp_image_train))
    train_labels = load_mnist_labels(fp_label_train)
    trainloader = get_dataloader(train_images, train_labels, batch_size)
    
    for epoch in range(n_epochs):
        running_loss = 0
        for batch_idx, (data, labels) in enumerate(trainloader):
            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # print(f"batch {batch_idx}: loss = {loss.item():.4f}", end="\r")
            if batch_idx%1==0:
                if queue is not None and queue.empty():
                    status = {
                        'epoch': epoch, 
                        'batch_idx': batch_idx, 
                        'loss': running_loss/len(trainloader), 
                        'accuracy': check_accuracy(model, train_images, train_labels),
                        'epoch_completed': batch_idx/len(trainloader)
                        }
                    queue.put(status)
        # print(f"Epoch {epoch}: running_loss = {running_loss/len(trainloader):.4f}. Accuracy = {100*accuracy:.2f}%")


    print(f"Accuracy on test set: {check_accuracy(model, image_preprocess(test_images), test_labels)*100}%")
    return


def train_NN_process(model, n_epochs=1, batch_size=8, lr=0.1):
    queue = Queue()
    args = (model, n_epochs, batch_size, lr, queue)
    process = Process(target = train_NN, args=args)
    process.start()
    return process, queue



def test_thread():

    model = MLP(s=[784, 16, 10])
    process, queue = train_NN_process(model, n_epochs=1, batch_size=64, lr=0.001)

    while process.is_alive():
        if not queue.empty():
            status = queue.get()
            print(status)
    process.join()


if __name__ == '__main__':
    # model = MLP(s=[784, 16, 16, 16, 10])
    # train_NN_thread(model, n_epochs=5, batch_size=100, lr=0.001)
    test_thread()


