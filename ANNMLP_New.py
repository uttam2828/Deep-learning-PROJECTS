# Importing necessary libraries for building the MLP (Multi-Layer Perceptron) model
# Import essential libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations and array operations
from keras.models import Sequential  # For building a sequential neural network model
from keras.layers import Dense  # For adding dense (fully connected) layers to the model
from tensorflow.keras.utils import to_categorical  # For pre-processing categorical data

#Load model architecture from JSON
from tensorflow.keras.models import model_from_json

# Connect to the MySQL database
from sqlalchemy import create_engine
from urllib.parse import quote
# Specify database credentials for connection
user = 'user1'  # Database username
pw = quote('amer@mysql')   # Database password
db = 'titan' # Database name

# Create a database engine using credentials
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Loading the data set using pandas as dataframe format 
train_nn = pd.read_csv(r"D:/New materials/DS/ANN/train_sample.csv")
test_nn = pd.read_csv(r"D:/New materials/DS/ANN/test_sample.csv")

# save dataframe into sql database
train_nn.to_sql('train_nn', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
test_nn.to_sql('test_nn', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# Define SQL queries to retrieve data from specified tables
sql1 = 'select * from train_nn;'  # Fetch all data from the 'train_nn' table
sql2 = 'select * from test_nn;'   # Fetch all data from the 'test_nn' table

# Retrieving train and test data from the database
# Read data from database into DataFrames
train = pd.read_sql_query(sql1, engine)  # Load data from 'train_nn' table
test = pd.read_sql_query(sql2, engine)   # Load data from 'test_nn' table


# Separate features (input data) and labels
x_train = train.iloc[:, 1:].values.astype("float32")  # Features from training set (excluding first column for labels)
x_test = test.iloc[:, 1:].values.astype("float32")    # Features from testing set (excluding first column)
y_train = train.label.values.astype("float32")       # Labels from training set
y_test = test.label.values.astype("float32")         # Labels from testing set

# Normalize pixel values between 0 and 1
x_train = x_train / 255  # Rescale pixel values for better model performance
x_test = x_test / 255

# One-hot encode labels for categorical representation
y_train = to_categorical(y_train)  # Convert labels to one-hot encoded format
y_test = to_categorical(y_test)

# Determine the number of classes (unique labels)
num_of_classes = y_test.shape[1]  # Get the number of unique label categories

# Create a sequential neural network model
model = Sequential()  # Initialize an empty sequential model


# Define the model architecture
model.add(Dense(150, input_dim=784, activation="relu"))  # First hidden layer with 150 neurons and ReLU activation
model.add(Dense(200, activation="tanh"))              # Second hidden layer with 200 neurons and tanh activation
model.add(Dense(100, activation="tanh"))              # Third hidden layer with 100 neurons and tanh activation
model.add(Dense(500, activation="tanh"))              # Fourth hidden layer with 500 neurons and tanh activation
model.add(Dense(num_of_classes, activation="softmax"))  # Output layer with number of classes and softmax activation for probabilities

# Configure model learning process
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  # - Loss function: categorical_crossentropy for multi-class classification
  # - Optimizer: adam (efficient gradient descent algorithm)
  # - Metrics: Monitor accuracy during training

# Visualize model architecture
model.summary()  # Display a summary of the model's layers and parameters

# Train the model
model.fit(x=x_train, y=y_train, batch_size=1000, epochs=20, verbose=1, validation_data=(x_test, y_test))
  # - Train on training data (x_train, y_train)
  # - Batch size: 1000 samples per training update
  # - Epochs: 20 iterations over the entire training set
  # - Verbose: 1 for progress bar
  # - Validation data: Evaluate model on test data during training

# Generate predictions on test data
predict = model.predict(x_test, verbose=1)  # Predict probabilities for each class on test data

# Convert predictions to class labels
results = pd.DataFrame(np.argmax(predict, axis=1), columns=['Label'])  # Extract the most probable class for each sample

# Evaluate model performance on test data
eval_score_test = model.evaluate(x_test, y_test, verbose=1)
print("Accuracy on test data: %.2f%%" % (eval_score_test[1] * 100))
# - Calculates metrics like loss and accuracy on the testing set.
# - Verbose: 1 for detailed output during evaluation.
# - Prints the test accuracy rounded to two decimal places.

# Evaluate model performance on training data
eval_score_train = model.evaluate(x_train, y_train, verbose=0)
print("Accuracy on train data: %.2f%%" % (eval_score_train[1] * 100))
# - Similar to above, but on the training set (usually higher accuracy).
# - Verbose: 0 to suppress output during evaluation.
# - Prints the training accuracy rounded to two decimal places.

# Save the trained model architecture (JSON format)
model_json = model.to_json()
with open("model.json", "w") as json_file:
  json_file.write(model_json)
# - Converts the model architecture to a JSON string.
# - Saves the JSON string to a file named "model.json".

# Save the trained model weights (HDF5 format)
model.save_weights("model.weights.h5")
# - Saves the model's learned weights (parameters) to an HDF5 file named "model.h5".

# **Testing the Saved Model**

# Load the model architecture from JSON file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# - Opens the "model.json" file for reading.
# - Reads the JSON content containing the model architecture.
# - Closes the file.

# Reconstruct the model from the loaded JSON
loaded_model = model_from_json(loaded_model_json)
# - Creates a new model based on the loaded JSON architecture.

# Load weights into the reconstructed model
loaded_model.load_weights("model.weights.h5")
print("Loaded Model from disk")
# - Loads the weights from the "model.h5" file into the reconstructed model.
# - Prints a message indicating successful loading.

# Compile the loaded model (for compatibility with evaluation)
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# - Compiles the loaded model (might be necessary for evaluation).
# - Uses the same loss function, optimizer, and metrics as the original model.

# Load test data for predictions (assuming CSV format)
test_data = pd.read_csv(r"test.csv")
test_pred = test_data.values.astype('float32') / 255
# - Reads the test data from a CSV file (path needs adjustment).
# - Converts the data to NumPy arrays and normalizes pixel values.

# Make predictions using the loaded model
predictions = loaded_model.predict(test_pred)
# - Generates predictions (probabilities for each class) on the test data.

# Convert predictions to class labels
result = pd.DataFrame(np.argmax(predictions, axis=1), columns=['Label'])
print(result)  # Display the DataFrame containing predicted labels
