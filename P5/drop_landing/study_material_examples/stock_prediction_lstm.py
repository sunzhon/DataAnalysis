import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import quandl 
import numpy as np
from datetime import date

def prdict_stock():
    start_date=date(2000,10,12)
    end_date= date.today()
    google_stock = pd.DataFrame(quandl.get("WIKI/GOOGL", start_date=start_date,end_date=end_date))
    print(google_stock.shape)
    print(google_stock.tail())
    print(google_stock.head())
    
    plt.figure(figsize=(16,8))
    plt.plot(google_stock['Close'])
    #plt.show()
    
    
    time_stamp= 50
    
    google_stock=google_stock[['Open','High', 'Low', 'Close', 'Volume']]
    
    train = google_stock[0:2800 + time_stamp]
    valid = google_stock[2800-time_stamp:]
    
    
    #规一化
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(train)
    x_train, y_train =[], []
    
    
    #训练集
    for i in range(time_stamp, len(train)):
        x_train.append(scaled_data[i-time_stamp:i])
        y_train.append(scaled_data[i,3])
    x_train, y_train=np.array(x_train), np.array(y_train)
    
    #Testset
    scaled_data=scaler.fit_transform(valid)
    
    x_valid, y_valid=[],[]
    for i in range(time_stamp,len(valid)):
        x_valid.append(scaled_data[i-time_stamp:i])
        y_valid.append(scaled_data[i,3])
    
    x_valid, y_valid= np.array(x_valid), np.array(y_valid)
    
    print(x_train.shape)
    print(train.head())
    # Modling
    ##super parameters
    epochs = 3
    batch_size = 16
    model=Sequential()
    model.add(LSTM(units=100,return_sequences=True, input_dim=x_train.shape[-1],input_length=x_train.shape[1]))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=1)
    
    
    closing_price=model.predict(x_valid)
    scaler.fit_transform(pd.DataFrame(valid['Close'].values))
    closing_price=scaler.inverse_transform(closing_price)
    y_valid=scaler.inverse_transform([y_valid])
    
    rms=np.sqrt(np.mean(np.power((y_valid-closing_price),2)))
    print(rms)
    
    
    plt.figure(figsize=[16,8])
    dict_data={
            'Prediction':closing_price.reshape(1,-1)[0],
            'Close':y_valid[0]
            }
    
    data_pd=pd.DataFrame(dict_data)
    
    plt.plot(data_pd[['Close','Prediction']])
    plt.show()
    





