import pandas as pd

data = {'name': ['Amol', 'Lini'],
	'physics': [77, 78],
	'chemistry': [73, 85]}

#create dataframe
df_marks = pd.DataFrame(data)
print('Original DataFrame\n------------------')
print(df_marks)

data_add = {}
data_add['name'] = 'Geo'
data_add['physics'] = 87
data_add['chemistry'] = 92

new_row = pd.Series(data=data_add, name='x')
#append row to the dataframe
df_marks = df_marks.append(new_row, ignore_index=True)

print('\n\nNew row added to DataFrame\n--------------------------')
print(df_marks)