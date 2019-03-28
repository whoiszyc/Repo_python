

def cal_test(a,*b):
    print(type(a))
    print(type(b))
    print(len(b))

    for i in b:
        print(i)

cal_test(1,[1,2,3],[5,4,5])



