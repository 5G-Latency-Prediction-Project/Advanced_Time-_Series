import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet, diagnostics
import itertools
import os
from prophet.serialize import model_to_json, model_from_json

def readAndFormatLatency(folderName, multipleSequence = True):
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
        dataFrame.drop(index=dataFrame.index[1:1000], axis=0, inplace=True)#Drop first 1000 measurments to discard transients at start of measurement
        if sequence == 0:
            ds = (dataFrame["timestamps.client.send.wall"] - dataFrame["timestamps.client.send.wall"][0]).apply(pd.Timestamp) #subtract first timestamp to get timestamps relative to time 0 (in ns), convert to pandas timestamp and only keep time (not date)
            nextSequenceStart = ds.iloc[-1] - ds.iloc[0] + pd.Timedelta(10, unit="milliseconds")
            sequence += 1
        else:
            ds = (dataFrame["timestamps.client.send.wall"] - dataFrame["timestamps.client.send.wall"][0]).apply(pd.Timestamp) + nextSequenceStart #subtract first timestamp to get timestamps relative to time 0 (in ns), convert to pandas timestamp and only keep time (not date)
            nextSequenceStart += ds.iloc[-1] - ds.iloc[0] + pd.Timedelta(10, unit="milliseconds")
        y = (dataFrame["timestamps.server.receive.wall"] - dataFrame["timestamps.client.send.wall"])*10**(-9)*10**3 #measured in ns, converted to ms
        if not multipleSequence:
            return pd.DataFrame(data = {"ds" : ds,  "y" : y}), Fs
        else:
            timeseries.append(pd.DataFrame(data = {"ds" : ds,  "y" : y}))
    return pd.concat(timeseries), Fs #concatenate all sequences into one

def plotTimeseries(timeseries):
    #timeseries["ds"] = timeseries["ds"].dt.strftime("%H:%M")
    #ax = timeseries.set_index("ds").plot(legend= None)
    #ax.set_ylabel("Latency [ms]")
    #ax.set_xlabel("Time (HH:MM)") #from measurement start
    #plt.show()
    plt.plot(timeseries["y"])
    plt.title('Original Data')
    plt.xlabel('Index')
    plt.ylabel('Latency (ms)')
    plt.grid(True)
    plt.show()

def splitTimeseries(timeseries, partition):
    timeseries1 = timeseries.iloc[:int(len(timeseries)*partition),:]
    timeseries2 = timeseries.iloc[int(len(timeseries)*partition):,:]
    return timeseries1, timeseries2

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

def getModel():
    with open('Prophet.json', 'r') as fin:
        model = model_from_json(fin.read())  # Load model
    return model

def saveModel(model, name):
    with open(name + ".json", "w") as fout:
        fout.write(model_to_json(model))  # Save model

def trainAndEvaluateModel(timeseries):
    trainTimeseries, testTimeseries = splitTimeseries(timeseries, 0.8)
    #model = Prophet(interval_width=0.95) #Larger internal width -> larger allowed variance in estimates. Parameter is prediction interval, for example 95% next data point sampled within the interval (assume gaussian dist)
    #model.fit(trainTimeseries) #takes 15 seconds for one sequence, 5 minutes for 6 sequences
    #saveModel(model, "Prophet")
    model = getModel()
    yHat = model.predict(testTimeseries)
    testTimeseries.loc[:, "yHat"] = yHat["yhat"].values
    testTimeseries.loc[:, "yHat lower"] = yHat["yhat_lower"].values
    testTimeseries.loc[:, "yHat upper"] = yHat["yhat_upper"].values
    timeseries = pd.concat([trainTimeseries, testTimeseries])
    print("Test RMSE:", ((testTimeseries.y - yHat.loc[:, "yhat"].values)**2).mean()**.5) 
    #print("Test MAPE: ", (yHat.loc[:, "yhat"].values-testTimeseries.y/(testTimeseries.y)).abs().sum())
    yHatnp = yHat.loc[:, "yhat"].to_numpy()
    ynp = testTimeseries.y.to_numpy()
    MAPE = np.sum(np.abs((yHatnp-ynp)/ynp))/len(ynp)
    print("Test MAPE: ", MAPE)
    #timeseries["ds"] = timeseries["ds"].dt.strftime("%H:%M")
    #ax = timeseries.set_index("ds").plot()
    #ax.set_ylabel("Latency [ms]")
    #ax.set_xlabel("Time (HH:MM)") 
    #plt.show()

    #fig1 = model.plot(yHat) #The interval looks alright
    #fig2 = model.plot_components(yHat)
    #plt.show()

    timeseries["ds"] = timeseries["ds"].dt.strftime("%H:%M:%S.%f")
    testIdx = int(len(timeseries)*0.8)
    timeseriesSubset = timeseries.iloc[testIdx:testIdx+200]
    #timeseriesSubset = timeseries.tail(200)
    ax = timeseriesSubset.set_index("ds").plot()
    ax.set_ylabel("Latency [ms]")
    ax.set_xlabel("Time (HH:MM:SS)") 
    plt.show() #Looks much better when zoomed in compared to viewing all data, the outliers become more visually overwhelming for all data

def finetuneAndTrainModel(timeseries):
    #Split timeseries into train(0.8*0.8), validation(0.8*0.2) and test (0.2)
    trainTimeseries, testTimeseries = splitTimeseries(timeseries, 0.8)
    trainVTimeseries, validationTimeseries = splitTimeseries(trainTimeseries, 0.8) #Use this for training and then validating on test set
    #After best parameters have been found, train a model using all training data (trainV + validation)
    
    #Hyperparameters that can be tuned, granularity can be increased
    param_grid = {"changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5], "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0], "holidays_prior_scale" : [0.01, 0.1, 1.0, 10.0],"seasonality_mode" : ["additive", "multiplicative"]}
    params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())] # Generate all combinations of parameters
    RMSEs = []  # Store the RMSEs for each params here
    for param in params:
        model = Prophet(**param) 
        model.fit(trainVTimeseries)
        yHat = model.predict(validationTimeseries)
        #validationTimeseries.loc[:, "yHat"] = yHat.loc[:,"yhat"].values
        RMSEs.append(((validationTimeseries.y - yHat.loc[:, "yhat"].values)**2).mean()**.5)
    
    # Display results
    Results = pd.DataFrame(params)
    Results['root-mean-square-error'] = RMSEs
    print(Results)

    #Find the best parameters
    optimalParams = params[np.argmin(RMSEs)]
    print("Default parameters: ", "{'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10, 'holidays_prior_scale': 10, 'seasonality_mode': 'additive'}")
    print("Best found parameters: ", optimalParams)
    model = Prophet(**optimalParams) 
    model.fit(trainTimeseries)
    saveModel(model, "OptimalProphet")

    
if __name__ == "__main__":
    folderNames = ["session13-UL"] #T = 5.5 ms -> 1 second is about 182 samples
    timeseries, Fs = readAndFormatLatency(folderNames[0], False)
    #plotTimeseries(timeseries)
    trainAndEvaluateModel(timeseries)
    #finetuneAndTrainModel(timeseries)
    