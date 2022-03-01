import pickle
file = open("pretrained_models/places-googlenet.pickle", "rb")
weights = pickle.load(file, encoding="bytes")
file.close()
