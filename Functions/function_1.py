import math

def heron(a, b, c):
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))

print(heron(3,4,5))


# Because both length and indicator have default values, either or both can be omitted entirely, in which case the default is used
# A parameter with a default is optional, while a parameter with no default is mandatory
def shorten(text, length=25, indicator="..."):
    if len(text) > length:
        text = text[:length - len(indicator)] + indicator
    return text

shorten("The Silkie") # returns: 'The Silkie'
shorten(length=7, text="The Silkie") # returns: 'The ...'
shorten("The Silkie", indicator="&", length=7) # returns: 'The Si&'
shorten("The Silkie", 7, "&") # returns: 'The Si&'



def append_if_even(x, lst=None):
    if lst is None:
        lst = []
    if x % 2 == 0:
        lst.append(x)
    return lst


