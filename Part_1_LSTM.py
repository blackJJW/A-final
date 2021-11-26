import torch 
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import datetime
import matplotlib.pyplot as plt 
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mm = MinMaxScaler()
ss = StandardScaler()

class Scaler:
    def __init__(self, X, y):
        print("Part_1_LSTM - Scaler  Start")
        self.X = X
        self.y = y
        self.prob = 0.7
        
    def prep(self):
        print("Part_1_LSTM - Scaler - prep  Start")
        
        print("----- fit Start -----")
        self.X_ss = ss.fit_transform(self.X)
        self.y_mm = mm.fit_transform(self.y)
        print("----- fit Done -----")
        print("----- setting test, train data Start -----")
        #------Test Data------------------------------------------------
        self.X_train = self.X_ss[:int(len(self.X_ss) * self.prob), :]
        self.X_test = self.X_ss[int(len(self.X_ss) * self.prob):, :]
        self.y_train = self.y_mm[:int(len(self.y_mm) * self.prob), :]
        self.y_test = self.y_mm[int(len(self.y_mm) * self.prob):, :]
        #---------------------------------------------------------------
        print("----- setting test, train data Done -----")
        
        #numpy형태에서는 학습이 불가능하기 때문에 학습할 수 있는 형태로 변환하기 위해 Torch로 변환
        print("----- transforming to Torch Start -----")
        self.X_train_tensors = Variable(torch.Tensor(self.X_train)) 
        self.X_test_tensors = Variable(torch.Tensor(self.X_test)) 

        self.y_train_tensors = Variable(torch.Tensor(self.y_train)) 
        self.y_test_tensors = Variable(torch.Tensor(self.y_test)) 

        self.X_train_tensors_final = torch.reshape(self.X_train_tensors, 
                                                   (self.X_train_tensors.shape[0], 1, self.X_train_tensors.shape[1])) 
        self.X_test_tensors_final = torch.reshape(self.X_test_tensors, 
                                                  (self.X_test_tensors.shape[0], 1, self.X_test_tensors.shape[1]))
        print("----- transforming to Torch Done -----")
        print("Part_1_LSTM - Scaler - prep  Done")
        print("Part_1_LSTM - Scaler   Done")
        return self.X_train_tensors_final, self.X_test_tensors_final, self.y_train_tensors, self.y_test_tensors, int(len(self.X_ss) * self.prob)

#--------------< LSTM Model >---------------------------------------------      
class LSTM1(nn.Module): 
  def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length): 
    print("Part_1_LSTM - LSTM1  Start")
    super(LSTM1, self).__init__() 
    self.num_classes = num_classes #number of classes 
    self.num_layers = num_layers #number of layers 
    self.input_size = input_size #input size 
    self.hidden_size = hidden_size #hidden state 
    self.seq_length = seq_length #sequence length 
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #lstm 
    self.fc_1 = nn.Linear(hidden_size, 128) #fully connected 1 
    self.fc = nn.Linear(128, num_classes) #fully connected last layer 

    self.relu = nn.ReLU() 
    
  def forward(self,x): 
    print("Part_1_LSTM - LSTM1 - forward  Start")
    h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state 
    c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state 
    # Propagate input through LSTM 
    
    output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state 
    
    hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next 
    out = self.relu(hn) 
    out = self.fc_1(out) #first Dense 
    out = self.relu(out) #relu 
    out = self.fc(out) #Final Output 
    print("Part_1_LSTM - LSTM1 - forward  Done")
    print("Part_1_LSTM - LSTM1  Done")
    return out

class LSTM_predict:
    def __init__(self, lstm1, num_epochs, learning_rate, X_train, y_train, length, df):
        print("Part_1_LSTM - LSTM_predict  Start")
        self.lstm1 = lstm1
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.X_train = X_train
        self.y_train = y_train
        self.length = length
        self.df = df
        
        self.epochs()
        self.predict()
    
    def epochs(self):
        print("Part_1_LSTM - LSTM_predict - epochs  Start")
        loss_function = torch.nn.MSELoss()    # mean-squared error for regression
        optimizer = torch.optim.Adam(self.lstm1.parameters(), lr=self.learning_rate)  # adam optimizer
        print("----- epoch Start -----")
        for epoch in tqdm(range(self.num_epochs)): 
          outputs = self.lstm1.forward(self.X_train.to(device)) #forward pass 
          optimizer.zero_grad() #caluclate the gradient, manually setting to 0 

          # obtain the loss function 
          loss = loss_function(outputs, self.y_train.to(device)) 

          loss.backward() #calculates the loss of the loss function 

          optimizer.step() #improve from loss, i.e backprop 
          if epoch % 100 == 0: 
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        print("----- epoch Done -----")
        print("Part_1_LSTM - LSTM_predict - epochs  Done")
            
    def predict(self):
        print("Part_1_LSTM - LSTM_predict - predict  Start")
        df_X_ss = ss.transform(self.df.drop(columns='close').iloc[:-1])
        df_y_mm = mm.transform(self.df.iloc[1:, 2:3])

        df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
        df_y_mm = Variable(torch.Tensor(df_y_mm))
        print("----- reshaping dataset Start -----")
        #reshaping the dataset
        df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))
        train_predict = self.lstm1(df_X_ss.to(device))#forward pass
        data_predict = train_predict.data.detach().cpu().numpy() #numpy conversion
        dataY_plot = df_y_mm.data.numpy()

        self.data_predict = mm.inverse_transform(data_predict) #reverse transformation
        self.dataY_plot = mm.inverse_transform(dataY_plot)
        print("----- reshaping dataset Done -----")
        
        print("----- drawing plt Start -----")
        plt.figure(figsize=(20,10)) #plotting
        plt.axvline(x= self.length, c='r', linestyle='--') #size of the training set

        plt.plot(dataY_plot, label='Actual Data') #actual plot
        plt.plot(data_predict, label='Predicted Data') #predicted plot
        plt.title('LSTM Time-Series Prediction')
        plt.legend()
        plt.show() 
        print("----- drawing plt Done -----")
        print("Part_1_LSTM - LSTM_predict - predict  Done")
        print("Part_1_LSTM - LSTM_predict  Done")
        