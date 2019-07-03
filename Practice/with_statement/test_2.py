
# Source: https://www.geeksforgeeks.org/with-statement-in-python/

# file handling

# 1) without using with statement
file = open('file_path_1', 'w')
file.write('hello world !')
file.close()

# 2) without using with statement
file = open('file_path_2', 'w')
try:
    file.write('hello world')
finally:
    file.close()


# using with statement
with open('file_path_with', 'w') as file:
    file.write('hello world ! (with statement)')



