# Test sequence unpacking operator (*)

import math

# First case
# Keywords are defined in the function
# The sequence unpacking operator is used when calling the function
def heron(a, b, c):
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))

L=[3,4,5]
print(heron(*L))


# Second case
# Keywords are not defined in the function
# *args means args will be a tuple with its items set to however many positional arguments are given
def product(*args):
    result = 1
    for arg in args:
        result *= arg  # result = result * arg
    return result

print(product(1, 2, 3, 4))  # args == (1, 2, 3, 4); returns: 24
print(product(5,3,8))       # args == (5, 3, 8); returns: 120
print(product(11))          # args == (11,); returns: 11




