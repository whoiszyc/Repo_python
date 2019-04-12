# sequence unpacking operator,an asterisk or star (*)
first, *rest = [9, 2, -4, 8, 7]  # not supported by python 2

def product(a, b, c):
    return a * b * c # here, * is the multiplication operator

L = [2, 3, 5]
print(*L)

# Pass parameter to function using unpacking operator
product(*L)





