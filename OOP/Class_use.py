import Class_def as test


print(test.MyClass.i)

# When the error says XXXX is not callable, it means when it is used, there is no need of ()
print(test.MyClass.__doc__)


# Instantiation
# Note that to do instantiation, () is needed
x = test.MyClass()

# the call x.f() is exactly equivalent to MyClass.f(x)
aa=x.f()

# However, test.MyClass.f() will not work, instead
aaa=test.MyClass.f(test.MyClass)


# Note the difference between x.f and test.MyClass.f
kk=x.f
kkk=test.MyClass.f


# Test of Function defined outside the class
y=test.C()
print(y.f(5,2))
