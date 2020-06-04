import pandas as pd

col_name = ['name', 'physics', 'chemistry']
df_marks = pd.DataFrame(columns=col_name)


data_add = {}
data_add['name'] = 'Geo'
data_add['physics'] = 87
data_add['chemistry'] = 92

new_row = pd.Series(data=data_add, name='x')
#append row to the dataframe
df_marks = df_marks.append(new_row, ignore_index=True)

print('\n\nNew row added to DataFrame\n--------------------------')
print(df_marks)