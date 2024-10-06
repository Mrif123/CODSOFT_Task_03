import streamlit as st
import numpy as np
import joblib

try:
  model=joblib.load('C:/Users/MOHAMMED RIFAIZ/OneDrive/Desktop/Codsoft/iris_svm_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")


def main():
    st.title("Iris Species Classification App")
    st.write("Enter the measurements of the iris flower to predict its species.")

    sepal_length=st.number_input("Sepal Length(cm)",min_value=0.0,max_value=10.0,value=5.0,step=0.1)
    sepal_width=st.number_input("Sepal Width(cm)",min_value=0.0,max_value=10.0,value=3.0,step=0.1)
    petal_length=st.number_input("Petal Length(cm)",min_value=0.0,max_value=10.0,value=1.5,step=0.1)
    petal_width=st.number_input("Petal Width(cm)",min_value=0.0,max_value=10.0,value=0.3,step=0.1)

    if st.button('Predict'):
        input_data=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
        
        try:
         prediction=model.predict(input_data)
         st.success(f"The predicted species is: {prediction[0]}")
        except ValueError as e:
            st.error(f"Prediction error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
if __name__ == '__main__':
    main()