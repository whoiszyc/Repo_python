# Custom exceptions are custom data types (classes)

# not running yet
found = False
for row, record in enumerate(table):
    for column, field in enumerate(record):
        for index, item in enumerate(field):
            if item == target:
                found = True
                break
            if found:
                break
        if found:
            break
    if found:
        print("found at ({0}, {1}, {2})".format(row, column, index))
    else:
        print("not found")