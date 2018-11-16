import Shape

# use intermediate variables
a = Shape.Point()
repr(a)

b = Shape.Point(3, 4)
str(b)
b.distance_from_origin()

b.x = -19
str(b)

# no intermediate variables
distance=Shape.Point(10,20).distance_from_origin()
print(distance)


# object itself does not and cannot be claimed in the method
print( a.eq(b) )



