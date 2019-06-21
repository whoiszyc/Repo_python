# import module sys to get the type of exception
import sys

randomList = ['a', 0, 2]

## With exception
print('Run with exception handling')
for entry in randomList:
    try:
        print("The entry is", entry)
        r = 1/int(entry)
        break
    except:
        print("Oops!",sys.exc_info()[0],"occured.")
        print("Next entry.")
        print()


## Without exception. the code cannot proceed
print()
print()
print('Run without exception handling')
for entry in randomList:
    print("The entry is", entry)
    r = 1/int(entry)


