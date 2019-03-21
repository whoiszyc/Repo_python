# Mappings are collections of key–value items and provide methods for accessing items and their keys and values.

# Define a Dictionary, values are integers
d1 = dict({"id": 1948, "name": "Washer", "size": 3})
d2 = dict(id=1948, name="Washer", size=3)
d3 = dict([("id", 1948), ("name", "Washer"), ("size", 3)])
d4 = dict(zip(("id", "name", "size"), (1948, "Washer", 3)))
d5 = {"id": 1948, "name": "Washer", "size": 3}


for a, b in d1.items():
    print(a,b)


for key in d1.keys():
    print(key)


# Define a Dictionary, values are lists
d = {"id": [1948, 1950, 1960], "name": ["Washer", "Yichen", "hehe"], "size": [3, 5, 7]}
print(d["id"][0])



