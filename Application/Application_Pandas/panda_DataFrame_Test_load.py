import numpy as np
import pandas as pd



# test the first data frame
df=pd.read_csv('test_2.csv')
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
print('The column that key equals to 0 is:')
# problem using csv is that the numerical key 0 after reading needs '0', that is, print(df[0]) give error
# So it is better to use excel file
print(df['0'])

# read excel file
df_1 = pd.read_excel('test_2.xls')
print('The column that key equals to 0 is:')
print(df_1[0])

# convert to dict
dc = df_1.to_dict()
print(type(dc))