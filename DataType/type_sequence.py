# Python provides five built-in sequence types: bytearray, bytes, list, str, and tuple
# strings and tuples are mutable
# list is mutable

# Define string
str1="abcd"
print(str1[1])



# Define tuples
tup1=(1, 3, "hehe", ("chenmoshijin", "anbenxinzuoshi"))
print(tup1[3][0][1])

hair = "black", "brown", "blonde", "red"
print(hair[0:2])

MANUFACTURER, MODEL, SEATING = (0, 1, 2)
MINIMUM, MAXIMUM = 0, 1
aircraft = ("Airbus", "A320-200", (100, 220))
aircraft[SEATING][MAXIMUM]

for x, y in ((-3, 4), (5, 12), (28, -45)):
    print(x,y)



# Define lists
list1=[1, 3, 4, 5]

list2 = [-17.5, "kilo", 49, "V", ["ram", 5, "echo"], 7]




