import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Example input: [square footage, bedrooms, bathrooms]
sample = np.array([[2000, 3, 2]])

prediction = model.predict(sample)
print("Predicted Price:", prediction[0])
