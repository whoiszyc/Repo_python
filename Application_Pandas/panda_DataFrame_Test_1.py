import numpy as np
import pandas as pd

# use string as key
mydict0 = [{'a': 10, 'b': 20, 'c': 30, 'd': 40},{'a': 100, 'b': 200, 'c': 300, 'd': 400},{'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]

# use int number as key
mydict1 = [{0: 10, 1: 20, 2: 30, 3: 40},{0: 100, 1: 200, 2: 300, 3: 400},{0: 1000, 1: 2000, 2: 3000, 3: 4000 }]



# test the first data frame
df=pd.DataFrame(mydict0)
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

# save data frame to csv
df.to_csv('test_1.csv')


# test the second data frame
# get entry by key
df=pd.DataFrame(mydict1)
print(df)
print('The column that key equals to 0 is:')
print(df[0])

# save data frame to csv
df.to_csv('test_2.csv', encoding='utf-8')