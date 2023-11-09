import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('trained_IRIS_classification_model.sav', 'rb'))

# Function to predict the flower species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
    prediction = loaded_model.predict(features)
    prediction_proba = loaded_model.predict_proba(features)
    return prediction[0], np.max(prediction_proba) * 100

# Streamlit UI
def main():
    st.title('🌸 Iris Flower Species Classifier 🌸')
    st.image('flower1.png', use_column_width=True)  # You can replace 'flower.jpg' with the path to your image file

    st.write(
        "Welcome to the Iris Flower Species Classifier! This app predicts the species of iris flowers "
        "based on their sepal and petal dimensions. Different species require different treatments and care."
        "\n\nPlease enter the flower's dimensions and click on 'Classify'."
    )

    # Input boxes for flower dimensions with arrows and 'cms' text
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sepal_length = st.number_input("Sepal Length (cms)", format="%.2f", step=0.01)
        
    with col2:
        sepal_width = st.number_input("Sepal Width (cms)", format="%.2f", step=0.01)
        
    with col3:
        petal_length = st.number_input("Petal Length (cms)", format="%.2f", step=0.01)
        
    with col4:
        petal_width = st.number_input("Petal Width (cms)", format="%.2f", step=0.01)
        

    if st.button("Classify"):
        try:
            species, probability = predict_species(sepal_length, sepal_width, petal_length, petal_width)
            st.write(f"🔍 Predicted Species: **{species}**")
            st.write(f"🎯 Probability: **{probability:.2f}%**")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.write("\n\n---\n\n")
    st.write("ℹ️ **Note**: This model was trained on the Iris dataset, a well-known dataset in machine learning. "
             "It contains 150 samples of iris flowers, each belonging to one of three species: Setosa, Versicolor, "
             "or Virginica. The model uses sepal and petal dimensions to classify the species. The predictions are "
             "for demonstration purposes only.")
            
# Run the app
if __name__ == '__main__':
    main()
