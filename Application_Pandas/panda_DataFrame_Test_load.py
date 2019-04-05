import numpy as np
import pandas as pd



# test the first data frame
df=pd.read_csv('test_1.csv')
print(df)

# general information of the data frame
print('Total number of data entries in the data frame is {}'.format(df.size))
print('Dimension of data entries in the data frame is {} by {}'.format(df.shape[0], df.shape[1]))

# get entry by location
print('Second column of the data frame')
print(df.iloc[:,1])
print('Second to third column of the data frame')
print(df.iloc[:,1:2])
print('Second to third row of the data frame')
print(df.iloc[1:2,:])

# get entry by key
print('The column that key equals to "a" is:')
print(df['a'])

