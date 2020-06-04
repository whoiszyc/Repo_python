import pandas as pd



# List of Tuples
students = [('jack', 34, 'Sydeny', 'Australia'),
            ('Riti', 30, 'Delhi', 'India'),
            ('Vikas', 31, 'Mumbai', 'India'),
            ('Neelu', 32, 'Bangalore', 'India'),
            ('John', 16, 'New York', 'US'),
            ('Mike', 17, 'las vegas', 'US')]

# Create a DataFrame object
dfObj = pd.DataFrame(students, columns=['Name', 'Age', 'City', 'Country'], index=['a', 'b', 'c', 'd', 'e', 'f'])

print("Original Dataframe", dfObj, sep='\n')

print("*****Add row in the dataframe using dataframe.append() ****")

# Pass the row elements as key value pairs to append() function
modDfObj = dfObj.append({'Name': 'Sahil', 'Age': 22}, ignore_index=True)

print("Updated Dataframe", modDfObj, sep='\n')

# Pass a series in append() to append a row in dataframe
modDfObj = dfObj.append(pd.Series(['Raju', 21, 'Bangalore', 'India'], index=dfObj.columns), ignore_index=True)

print("Updated Dataframe", modDfObj, sep='\n')

print("**** Add multiple rows in the dataframe using dataframe.append() and Series ****")

# List of series
listOfSeries = [pd.Series(['Raju', 21, 'Bangalore', 'India'], index=dfObj.columns),
                pd.Series(['Sam', 22, 'Tokyo', 'Japan'], index=dfObj.columns),
                pd.Series(['Rocky', 23, 'Las Vegas', 'US'], index=dfObj.columns)]

# Pass a list of series to the append() to add multiple rows
modDfObj = dfObj.append(listOfSeries, ignore_index=True)

print("Updated Dataframe", modDfObj, sep='\n')

print("*****Add a row from one dataframe to other dataframe ****")

# Create an another dataframe
# List of Tuples
students = [('Rahul', 22, 'Sydeny', 'Australia'),
            ('Parul', 23, 'Pune', 'India')]

# Create a DataFrame object
dfObj2 = pd.DataFrame(students, columns=['Name', 'Age', 'City', 'Country'], index=['a', 'b'])

print("Another Dataframe", dfObj2, sep='\n')

# add row at index b from dataframe dfObj2 to dataframe dfObj
modDfObj = dfObj.append(dfObj2.loc['b'], ignore_index=True)

print("Updated Dataframe", modDfObj, sep='\n')

print("*****Add a row in the dataframe using loc[] ****")

# Add a new row at index k with values provided in list
dfObj.loc['k'] = ['Smriti', 26, 'Bangalore', 'India']

print("Updated Dataframe", dfObj, sep='\n')

print("*****Add a row in the dataframe at index position using iloc[] ****")

# Add a new row at index position 2 with values provided in list
dfObj.iloc[2] = ['Smriti', 26, 'Bangalore', 'India']

print("Updated Dataframe", dfObj, sep='\n')
