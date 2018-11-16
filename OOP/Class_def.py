
class MyClass:
    """A simple example class"""
    i = 12345

    def f(self):
        return 'hello world'


# Function defined outside the class
def f1(self, x, y):
    return min(x, x+y)

class C:
    f = f1

    def g(self):
        return 'hello world'

    h = g