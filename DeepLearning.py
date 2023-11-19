import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import datetime
    
class LSTMModel(nn.Module):
    def __init__(self, previousSamples, futureSamples = 1, nodesPerLayer = 50, layers = 3, dropout_prob = 0.5):
        super(LSTMModel, self).__init__()
        self.layers = layers
        self.nodes = nodesPerLayer
        self.future = futureSamples
        self.prev = previousSamples
        self.lstm = nn.LSTM(1, nodesPerLayer, layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(nodesPerLayer, futureSamples)

    def forward(self, x):
        x = x.reshape(len(x), self.prev, 1)
        h0 = torch.zeros(self.layers, x.size(0), self.nodes,device=x.device).requires_grad_()
        c0 = torch.zeros(self.layers, x.size(0), self.nodes,device=x.device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = out[:, -1, :]
        out = self.fc(out)
        return out.reshape(len(out),1,self.future)

class GRUModel(nn.Module):
    def __init__(self, previousSamples, futureSamples = 1, nodesPerLayer = 20, layers = 2, dropout_prob = 0.3):
        super(GRUModel, self).__init__()
        self.layers = layers
        self.nodes = nodesPerLayer
        self.prev = previousSamples
        self.future = futureSamples
        self.gru = nn.GRU(1, nodesPerLayer, layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(nodesPerLayer, futureSamples)

    def forward(self, x):
        x = x.reshape(len(x), self.prev, 1)
        h0 = torch.zeros(self.layers, x.size(0), self.nodes,device=x.device).requires_grad_()
        out, _ = self.gru(x, h0.detach())
        out = out[:, -1, :]
        out = self.fc(out)
        return out.reshape(len(out),1,self.future)
    
class CNNModel(nn.Module):
    def __init__(self, previousSamples, futureSamples = 1, kernel = 2):
        super(CNNModel, self).__init__()
        self.future = futureSamples
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=2),  # 1D convolutional layer
            nn.ReLU(),
            nn.BatchNorm1d(3),
            nn.Conv1d(in_channels=3, out_channels=2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(2),
            nn.Flatten(),  # Flatten the output for fully connected layers
            nn.Linear(2 * (previousSamples - 1), 50),  # Adjust the input size based on your sequence length
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, futureSamples)
        )
    def forward(self, x):
        x = self.cnn(x)
        return x.reshape(len(x),1,self.future)
    
    
class LinearModel(nn.Module):
    def __init__(self, previousSamples, futureSamples = 1):
        super(LinearModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(previousSamples, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, futureSamples)
        )
    def forward(self, x):
        x = self.linear(x)
        return x


def readLatency(folderName, multipleSequence = True):
    #returns sampling frequency, scales latency to ms and converts time stamps to a difference from first time stamp that is HH:MM:SS = 00:00:00.
    #puts the resulting timeseries in a pandas dataframe, to be used by prophet
    timeseries = list()
    folderPath = "Data/" + folderName + "/"
    sequence = 0
    for parquetFile in os.listdir(folderPath):
        fileName = os.fsdecode(parquetFile)
        dataFrame = pd.read_parquet(folderPath + fileName)
        Fs = 1/(dataFrame["packet_interval"][0]*10**(-9)) #Sampling frequency, 1/(sampling period) where sampling period is given in ns
        dataFrame = dataFrame[["timestamps.client.send.wall", "timestamps.server.receive.wall"]]
        y = (dataFrame["timestamps.server.receive.wall"] - dataFrame["timestamps.client.send.wall"])*10**(-9)*10**3 #measured in ns, converted to ms
        y = y.to_numpy()
        if not multipleSequence:
            return y[1000:], Fs #Drop first 1000 samples to remove transients at measurement start
        else:
            timeseries.append(y.to_numpy())
    return np.concatenate(timeseries), Fs #concatenate all sequences into one

def subSample(timeseries, windowLength):
    means = list()
    N = len(timeseries)
    windows = N//windowLength
    windowStartIdx = 0
    for windowIdx in range(windows):
        window = timeseries[windowStartIdx:windowStartIdx+windowLength]
        means.append(window.mean())
        windowStartIdx += windowLength
    return np.array(means)

def plotTimeseries(timeseries):
    plt.plot(timeseries)
    plt.xlabel('Index')
    plt.ylabel('Latency [ms]')
    plt.grid(True)
    plt.show()

def readCSV(fileName):
    timeseries = list()
    dataFrame = pd.read_csv("Data/" + fileName)
    y = dataFrame["0"]
    return y.to_numpy()

def splitTimeseries(timeseries, partition):
    timeseries1 = timeseries[:int(len(timeseries)*partition)]
    timeseries2 = timeseries[int(len(timeseries)*partition):]
    return timeseries1, timeseries2

def preProcess(timeseries, normalize = False):
    #timeseries = timeseries[timeseries < 20] #Remove outliers
    trainTempTimeseries, testTimeseries = splitTimeseries(timeseries, 0.8)
    trainTimeseries, validationTimeseries = splitTimeseries(trainTempTimeseries, 0.8) 

    if normalize:
        trainMean = trainTimeseries.mean()
        trainStd = trainTimeseries.std()
        #Normalize with training (!) statistics
        trainTimeseries = (trainTimeseries - trainMean)/trainStd
        validationTimeseries = (validationTimeseries - trainMean)/trainStd
        testTimeseries = (testTimeseries - trainMean)/trainStd

    return trainTimeseries, validationTimeseries, testTimeseries

def createDataset(timeseries, previousSamples, futureSamples):
    X, y = [], []
    for i in range(len(timeseries)-previousSamples-futureSamples):
        feature = timeseries[i:i+previousSamples]
        target = timeseries[i+previousSamples:i+previousSamples+futureSamples]
        X.append(feature)
        y.append(target)
    X = torch.tensor(np.array(X), dtype=torch.float)
    y = torch.tensor(np.array(y), dtype=torch.float)
    return X[:, None, :], y[:, None, :] 

def calculateTime(start, stop):
    totalTimeDiff = stop - start
    totalDiffSeconds = int(totalTimeDiff.total_seconds())
    TotalDiffMinuteSeconds, seconds = divmod(totalDiffSeconds, 60)
    hours, minutes = divmod(TotalDiffMinuteSeconds, 60)
    return f"{hours}h {minutes}m {seconds}s"

def createAndTrainModel(XTrain, yTrain, XVal, yVal, previousSamples, futureSamples):
    startTime = datetime.datetime.now()
    #model = CNNModel(previousSamples, futureSamples)
    #model = LinearModel(previousSamples, futureSamples)
    #model = GRUModel(previousSamples, futureSamples)
    model = LSTMModel(previousSamples, futureSamples, nodesPerLayer = 50, layers = 3, dropout_prob = 0.5)
    optimizer = optim.Adam(model.parameters())
    lossFunction = nn.MSELoss()
    dataLoader = data.DataLoader(data.TensorDataset(XTrain, yTrain), shuffle=False, batch_size=16)
    trainingEpochs = 15
    bestValRMSE = 500
    for epoch in range(trainingEpochs):
        model.train()
        for XBatch, yBatch in dataLoader:
            yHat = model(XBatch)
            loss = lossFunction(yHat, yBatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0: #Can be adjusted to train more or less
            model.eval()
            with torch.no_grad():
                yHat = model(XTrain)
                trainRMSE = np.sqrt(lossFunction(yHat, yTrain))
                yHat = model(XVal)
                valRMSE = np.sqrt(lossFunction(yHat, yVal))
                if valRMSE < bestValRMSE:
                    bestModelTime = datetime.datetime.now()
                    bestValRMSE = valRMSE
                    torch.save(model, "LSTM.pth")
            print("Training Epoch: ", epoch, "    train RMSE: ", trainRMSE.item(), "    validation RMSE: ", valRMSE.item()) 
    stopTime = datetime.datetime.now()
    print("Total training time: ", calculateTime(startTime, stopTime))
    print("Time to find model with lowest validation loss: ", calculateTime(startTime, bestModelTime))
    return model

def DisplayPerformance(model, XTest, yTest):
    if model == None:
        model = torch.load("LSTM.pth")
    model.eval()
    lossFunction = nn.MSELoss()
    with torch.no_grad():
        yHat = model(XTest)
        MAPE = torch.sum((yHat-yTest/(yTest)).abs())/(yHat.shape[0]*yHat.shape[2])
        #print("Test RMSE: ", np.sqrt(lossFunction(yHat, yTest)).item()) #LSTM 2.209141492843628 (1.76 for one sample), GRU 2.2371926307678223 (Also much more computationally expensive)
        print("Test MAPE all prediction: ", MAPE.item())
    plt.plot(yTest[:, : , 0].reshape(-1)[:200], label= "y")
    plt.plot(yHat[:, : , 0].reshape(-1)[:200], label = "yHat")
    plt.xlabel('Test sample')
    plt.ylabel('Latency [ms]')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    folderNames = ["session13-UL"] #T = 5.5 ms -> 1 second is about 182 samples
    timeseries, Fs = readLatency(folderNames[0], False)
    #timeseries = subSample(timeseries, 182) #each sample now is average latency for a second of measurements
    #plotTimeseries(timeseries)
    #timeseries = readCSV("denoised.csv")
    trainTimeseries, validationTimeseries, testTimeseries = preProcess(timeseries)
    previousSamples = 10
    futureSamples = 1
    XTrain, yTrain = createDataset(trainTimeseries, previousSamples, futureSamples)
    XVal, yVal = createDataset(validationTimeseries, previousSamples, futureSamples)
    XTest, yTest = createDataset(testTimeseries, previousSamples, futureSamples)
    model = createAndTrainModel(XTrain, yTrain, XVal, yVal, previousSamples, futureSamples)
    DisplayPerformance(None, XTest, yTest)
    