
import numpy as np  # Importing the NumPy library for numerical operations
import pandas as pd  # Importing the Pandas library for data manipulation
import streamlit as st  # Importing the Streamlit library for building web applications

# Importing model_from_json from the TensorFlow library to load the trained model
from tensorflow.keras.models import model_from_json
from sqlalchemy import create_engine  # Importing create_engine from the SQLAlchemy library for database operations

# Opening and reading the serialized model architecture from a JSON file
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

# Using Keras model_from_json to create a model architecture from the loaded JSON
loaded_model = model_from_json(loaded_model_json)

# Loading the trained model weights
loaded_model.load_weights("C:/Users/uttam/Desktop/Al/6.d.Deep Learning & AI-ANN/model.weights.h5")
print("Loaded Model from disk")

# Compiling the loaded model with specified loss function, optimizer, and metrics
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Function to make predictions using the loaded model and store the results in a database
def predict(data, user, pw, db):
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")  # Creating a database engine

    # Preprocessing data for prediction by converting values to float32 and normalizing
    data_pred = data.values.astype('float32') / 255
    
    # Making predictions using the loaded model
    prediction = pd.DataFrame(np.argmax(loaded_model.predict(data_pred), axis=1), columns=['Label'])
    
    # Combining predictions with the original data
    final = pd.concat([prediction, data], axis=1)
    
    # Storing the results in a database table named 'annmlp_test'
    final.to_sql('annmlp_test', con=engine, if_exists='replace', chunksize=1000, index=False)

    return final

# Main function to build the web application interface
def main():
    st.title("Neural_Network")  # Setting title for the web app
    st.sidebar.title("Neural_Network")  # Setting title for the sidebar
    
    # HTML template for styling
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Neural_Network </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)  # Rendering HTML template
    
    st.text("")  # Adding space
    
    # Uploading file through sidebar
    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'], accept_multiple_files=False, key="fileUploader")
    
    # Checking if file is uploaded
    if uploadedFile is not None:
        try:
            data = pd.read_csv(uploadedFile)  # Attempting to read CSV file
        except:
            try:
                data = pd.read_excel(uploadedFile)  # Attempting to read Excel file
            except:
                data = pd.DataFrame(uploadedFile)  # Creating DataFrame from uploaded file
        
    else:
        st.sidebar.warning("You need to upload a CSV or Excel file.")  # Warning message if no file is uploaded
    
    # HTML template for database credentials
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
   # Rendering HTML template for database credentials on the sidebar
    # Rendering HTML template for database credentials on the sidebar
    st.sidebar.markdown(html_temp, unsafe_allow_html=True)
    
    # Text input field for entering the database username on the sidebar
    user = st.sidebar.text_input("user", "Type Here")
    
    # Text input field for entering the database password on the sidebar
    pw = st.sidebar.text_input("password", "Type Here")
    
    # Text input field for entering the database name on the sidebar
    db = st.sidebar.text_input("database", "Type Here")

    result = ""  # Initializing result variable
    
    # Button to trigger prediction
    if st.button("Predict"):
        result = predict(data, user, pw, db)  # Calling the predict function
    
        # Displaying results in a table with background gradient
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap=True)
        st.table(result.style.background_gradient(cmap=cm).set_precision(2))

# Running main function if the script is executed directly
if __name__ == '__main__':
    main()
