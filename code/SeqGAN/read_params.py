import pickle

with open("target_params.pkl", "rb") as f:
    data = pickle.load(f)

print type(data)
print len(data)
print data[0].shape
