
# There are two ways to initialize an instance

# First method is to pass necessary parameters as function inputs when the instance is first defined
# Disadvantage is that the parameters have to be pass in order

# Second method is to pass necessary parameters as attribute values of the instance
# Disadvantage is that the attribute names have to be used

# A mixed strategy can be used. Parameters are defined as function inputs with default value
# Then use the second method mainly to pass the parameters

# Method 1:
class Car(object):
	"""
		blueprint for car
	"""

	def __init__(self, model, color, made, speed_limit):
		self.color = color
		self.company = made
		self.speed_limit = speed_limit
		self.model = model

	def start(self):
		print("started")

	def stop(self):
		print("stopped")

	def accelarate(self):
		print("accelarating...")
		"accelarator functionality here"

	def change_gear(self, gear_type):
		print("gear changed")
		" gear related functionality here"


maruthi_suzuki = Car("ertiga", "black", "suzuki", 60)
audi = Car("A6", "red", "audi", 80)
print()
print()

# Method 2
class Rectangle:
	def __init__(self):
		self.length = 100 # default value
		self.breadth = 80 # default value
		self.unit_cost = 0

	def get_perimeter(self):
		return 2 * (self.length + self.breadth)

	def get_area(self):
		return self.length * self.breadth

	def calculate_cost(self):
		area = self.get_area()
		return area * self.unit_cost


# breadth = 120 cm, length = 160 cm, 1 cm^2 = Rs 2000
r = Rectangle()
r.length = 160
r.breadth = 120
r.unit_cost = 10
print("Area of Rectangle: %s cm^2" % (r.get_area()))
print("Cost of rectangular field: Rs. %s " % (r.calculate_cost()))
print()
print()




# Mixed strategy
class Rectangle:
	def __init__(self, length = 100, breadth = 80, unit_cost = 0):
		self.length = length
		self.breadth = breadth
		self.unit_cost = unit_cost

	def get_perimeter(self):
		return 2 * (self.length + self.breadth)

	def get_area(self):
		return self.length * self.breadth

	def calculate_cost(self):
		area = self.get_area()
		return area * self.unit_cost


# breadth = 120 cm, length = 160 cm, 1 cm^2 = Rs 2000
r = Rectangle()
r.length = 160
r.breadth = 120
r.unit_cost = 10
print("Area of Rectangle: %s cm^2" % (r.get_area()))
print("Cost of rectangular field: Rs. %s " % (r.calculate_cost()))