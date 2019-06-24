
# This case demonstrates if the package is initialized by using __init__.py file
# Then the package "Animals" can be imported using the package name


# Import classes from your brand new package
from Animals import Mammals
from Animals import Birds

# Create an object of Mammals class & call a method of it
myMammal = Mammals()
myMammal.printMembers()

# Create an object of Birds class & call a method of it
myBird = Birds()
myBird.printMembers()