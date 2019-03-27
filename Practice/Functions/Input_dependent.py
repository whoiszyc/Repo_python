

def cal_test(a,*b):
    print(type(a))
    print(type(b))
    k=sum(i for i in b)
    return a+k

cal_test(1,2,3,5)

def test_optional_input(a,b=None):
    print(type(a))
    print(type(b))

    if b == None:
        return a
    else:
        return a+b


test_optional_input(1,2)

