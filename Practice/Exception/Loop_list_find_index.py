# vowels list
vowels = ['a', 'e', 'i', 'o', 'i', 'u']

# element 'e' is searched
index = vowels.index('e')
# index of 'e' is printed
print('The index of e:', index)

# element 'i' is searched
index = vowels.index('i')
# only the first index of the element is printed
print('The index of i:', index)

# element 'k' is searched
try:
    index = vowels.index('k')
except:
    print('Searched element is not in the list')