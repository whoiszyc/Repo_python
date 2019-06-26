# Example 2: super() with Multiple Inheritance

class Animal:
    def __init__(self, animalName):
        print(animalName, 'is an animal.')


class Mammal(Animal):
    def __init__(self, mammalName):
        print(mammalName, 'is a warm-blooded animal.')
        super().__init__(mammalName)


class NonWingedMammal(Mammal):
    def __init__(self, NonWingedMammalName):
        print(NonWingedMammalName, "can't fly.")
        super().__init__(NonWingedMammalName)


class NonMarineMammal(Mammal):
    def __init__(self, NonMarineMammalName):
        print(NonMarineMammalName, "can't swim.")
        super().__init__(NonMarineMammalName)


class Dog(NonMarineMammal, NonWingedMammal):
    def __init__(self):
        print('Dog has 4 legs.')

        # initialization 1 (correct)
        # super().__init__('Dog')

        # initialization 2 (not correct)
        NonMarineMammal.__init__(self, 'Dog')
        NonWingedMammal.__init__(self, 'Dog')


d = Dog()
print('')
bat = NonMarineMammal('Bat')