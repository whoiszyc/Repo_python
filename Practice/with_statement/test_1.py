
# https://www.pythonforbeginners.com/files/with-statement-in-python

# In Python you need to give access to a file by opening it. You can do it by using
# the open() function. Open returns a file object, which has methods and attributes
# for getting information about and manipulating the opened file.

# With the "With" statement, you get better syntax and exceptions handling.
# "The with statement simplifies exception handling by encapsulating common
# preparation and cleanup tasks."
# In addition, it will automatically close the file. The with statement provides
# a way for ensuring that a clean-up is always used.



# Without the with statement, we would write something like this:
file = open("welcome.txt")
data = file.read()
print(data)
file.close()  # It's important to close the file when you're done with it



# With Statement Usage
with open("welcome.txt") as file:
    data = file.read()
    print(data)

