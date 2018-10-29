
# Exception handling allows parts of a python program contain error under certain cases.


d=["a", "zyc", 33, 55]


try:
    x=d[5]
except IndexError:
    print("Print error using exception handling: list index out of range")


# So when we use multiple except blocks, we must always order them from most specific (lowest in the hierarchy) to least
# specific (highest in the hierarchy).
try:
    x = d[5]
except LookupError:
    print("Print error using exception handling: Lookup error occurred")
except IndexError:
    print("Print error using exception handling: list index out of range")
except KeyError:
    print("rint error using exception handling: Invalid key used")


