import streamlit as st
import pandas as pd
import pickle
st.write("""
# Heart disease Prediction App

This app predicts If a patient has a heart disease

Data obtained from Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset.
""")

st.sidebar.header('User Input Features')



# Collects user input features into dataframe

def user_input_features():
    age = st.sidebar.number_input('Enter your age:  Between 29 to 71 ')

    sex  = st.sidebar.selectbox('Sex 0-Female 1-Male',(0,1))
    cp = st.sidebar.selectbox('Chest pain type',(0,1,2,3))
    tres = st.sidebar.number_input('Resting blood pressure: Between 94 to 200')
    chol = st.sidebar.number_input('Serum cholestoral in mg/dl: Between 126 to 564 ')
    fbs = st.sidebar.selectbox('Fasting blood sugar Enter 0 or 1',(0,1))
    res = st.sidebar.number_input('Resting electrocardiographic results: Enter 0 ,1 or 2')
    tha = st.sidebar.number_input('Maximum heart rate achieved: Between 71 to 202 ')
    exa = st.sidebar.selectbox('Exercise induced angina: ',(0,1))
    old = st.sidebar.number_input('oldpeak Between 0 t0 6.2  ')
    slope = st.sidebar.number_input('the slope of the peak exercise ST segmen: Enter 1,2 or 3')
    ca = st.sidebar.selectbox('number of major vessels Enter 0,1,2 or 3 ',(0,1,2,3))
    thal = st.sidebar.selectbox('thal Enter 0,1,2 or 3',(0,1,2))

    data = {'age': age,
            'sex': sex, 
            'cp': cp,
            'trestbps':tres,
            'chol': chol,
            'fbs': fbs,
            'restecg': res,
            'thalach':tha,
            'exang':exa,
            'oldpeak':old,
            'slope':slope,
            'ca':ca,
            'thal':thal
                }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire dataset
# This will be useful for the encoding phase
heart_dataset = pd.read_csv('heart.csv')
heart_dataset = heart_dataset.drop(columns=['target'])

df = pd.concat([input_df,heart_dataset],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

df = df[:1] # Selects only the first row (the user input data)

st.write(input_df)
# Reads in saved classification model
load_clf = pickle.load(open('Random_forest_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)