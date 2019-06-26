# In Python, super() built-in has two major use cases:
#   Allows us to avoid using base class explicitly
#   Working with Multiple Inheritance


# In case of single inheritance, it allows us to refer base class by super().
class Mammal(object):
    def __init__(self, mammalName):
        print(mammalName, 'is a warm-blooded animal.')


# without super
# in the initialization of child class, we need to call the initialization of the parent class name
class Dog(Mammal):
    def __init__(self):
        Mammal.__init__(self, 'Dog')
        print('Dog has four legs.')

d1 = Dog()
print(type(d1))


# using super
# in the initialization of child class, we do not need to call the initialization of the parent class
class Dog1(Mammal):
    def __init__(self):
        super().__init__('Dog')
        print('Dog has four legs.')

d2 = Dog1()
print(type(d2))

