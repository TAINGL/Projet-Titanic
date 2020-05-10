# Consigne pour une app avec streamlit : 
# Faire ça mais avec le titanic : https://twitter.com/AlexVianaPro/status/1181223836727029760/photo/1
# à gauche : possibilité de choisir les valeurs des différentes features. 
# à droite, le pourcentage de probabilité que la personne survive.
# Bonus : laisser le choix du model de prédiction  à l'utilisateur 

import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Add a title
st.title("Who Survived the Titanic?")
st.subheader("Predicting Who's going to Survive on Titanic Dataset")

# Image Manipulation
st.image('titanic.jpg', width= 800)

# Introduction
"""
**The sinking of the Titanic is one of the most infamous shipwrecks in history.**

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc). 

[Source link - Kaggle: Titanic](https://www.kaggle.com/c/titanic)
"""

# EDA
DATA_URL = "train.csv"

def age_missing_replace(means, dframe, title_list):
    for title in title_list:
        temp = dframe['Title'] == title 
        dframe.loc[temp, 'Age'] = dframe.loc[temp, 'Age'].fillna(means[title]) 


# To Improve speed and cache data
@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv(os.path.join(DATA_URL))
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Dr', 'Rev', 'Col','Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')

    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Fare'] = data['Fare'].round(0) 
    
    # Replace place where age is missing with mean age for each title
    means = round(data.groupby('Title')['Age'].mean(),0)
    title_list = ['Master','Miss','Mr','Mrs','Others']
    age_missing_replace(means, data, title_list)

    # Create new columns for Family Size = SibSp + Parch + 1 (for yourself)
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    # LabelEcoder for categorical variable
    lb_make = LabelEncoder()
    data['Embarked'] = lb_make.fit_transform(data['Embarked']) # 0:C, 1:Q , 2:S
    data['Title'] = lb_make.fit_transform(data['Title']) # 0:Master, 1:Miss, 2:Mr , 3:Mrs, 4:Others
    data['Sex'] = lb_make.fit_transform(data['Sex']) # 0: M, 1:F
    data = data.drop(['Ticket','PassengerId','Cabin','Name'],axis=1)


    return data

# Control load data
data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text('Loading data...done!')

################################################################
st.markdown("## 1. Dataset description")
# Show Dataset
if st.checkbox("Preview DataFrame"):
    data = load_data()
    if st.button("Head"):
        st.write(data.head())
    if st.button("Tail"):
        st.write(data.tail())
    else:
        st.write(data.head(2))

# Show Entire Dataframe
if st.checkbox("Show All DataFrame"):
    data = load_data()
    st.dataframe(data)

# Show Description
if st.checkbox("Show All Column Name"):
    data = load_data()
    st.text("Columns:")
    st.write(data.columns)

# Dimensions
data_dim = st.radio('What Dimension Do You Want to Show',('Rows','Columns'))
if data_dim == 'Rows':
    data = load_data()
    st.text("Showing Length of Rows")
    st.write(len(data))
if data_dim == 'Columns':
    data = load_data()
    st.text("Showing Length of Columns")
    st.write(data.shape[1])

if st.checkbox("Show Summary of Dataset"):
    data = load_data()
    st.write(data.describe())

################################################################
st.markdown("## 2. Description of variables")
"""
**Variable Notes**

**pclass**: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

**age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

**sibsp**: The dataset defines family relations in this way...
- *Sibling* = brother, sister, stepbrother, stepsister
- *Spouse* = husband, wife (mistresses and fiancés were ignored)

**parch**: The dataset defines family relations in this way...
- *Parent* = mother, father
- *Child* = daughter, son, stepdaughter, stepson

Some children travelled only with a nanny, therefore parch=0 for them.
"""

# Selection
species_option = st.selectbox('Select Columns',('Embarked','Title','Genre','Pclass','Age','SibSp', 'Parch', 'Fare', 'FamilySize'))
data = load_data()
if species_option == 'Embarked':
    st.write(data['Embarked'])
elif species_option == 'Title':
    st.write(data['Title'])
elif species_option == 'Genre':
    st.write(data['Sex'])
elif species_option == 'Pclass':
    st.write(data['Pclass'])
elif species_option == 'Age':
    st.write(data['Age'])
elif species_option == 'SibSp':
    st.write(data['SibSp'])
elif species_option == 'Parch':
    st.write(data['Parch'])
elif species_option == 'Fare':
    st.write(data['Fare'])
elif species_option == 'FamilySize':
    st.write(data['FamilySize'])
else:
    st.write("Select A Column")


################################################################
st.markdown("## 3. Graphics")

COLUMN = 'Sex'

st.subheader('Probability of survival (Age and Sex)')
# Choose between woman and man
genre = st.radio(
        "Woman or Man",
        ('Woman', 'Man'))

filtered_data = data[data[COLUMN] == genre]

survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(10, 4))
ax = sns.distplot(data[data['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes, kde =False)
ax = sns.distplot(data[data['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes, kde =False)
ax.legend()
ax.set_title(genre)
st.pyplot()

################################################################
st.markdown("## 4. Resultat of Prediction Titanic Games")
st.subheader("Prediction")

# Prediction Titanic Games

# Choose your ML model
def classifiers(classifier):
    classifiers = {
        'Random Forest': RandomForestClassifier(),
        'Regression logistique': LogisticRegression(),
        'KNN': KNeighborsClassifier(),
        'LinearSVC': LinearSVC(), 
        'SVC': SVC(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'LinearRegression': LinearRegression(), 
        'LogisticRegression': LogisticRegression(),
        'GaussianNB': GaussianNB(), 
        'MultinomialNB': MultinomialNB(), 
        'ComplementNB': ComplementNB(), 
        'BernoulliNB': BernoulliNB(),      
    }

    return classifiers[classifier]

# Give the input data, add a selector for the app mode on the sidebar.
    
def input_data():
    st.sidebar.subheader("✔️ Prediction Titanic Games")

    st.sidebar.title("Sexe")
    Sexe = st.sidebar.selectbox("Choose your genre: ", ('M', 'F')) # 0: M, 1:F

    st.sidebar.title("Class")
    Class = st.sidebar.selectbox("Choose your class: ", data['Pclass'].unique())

    st.sidebar.title("Age")
    min_age = min(data['Age'])
    max_age = max(data['Age'])
    Age = st.sidebar.slider("Choose your age: ", min_value=round(min_age), max_value=round(max_age), value=20, step=1)

    st.sidebar.title("Ticket Price")
    min_ticket = min(data['Fare'])
    max_ticket = max(data['Fare'])
    Fare = st.sidebar.slider("Choose your ticket price: ", min_value=round(min_ticket), max_value=round(max_ticket), value=100, step=1)

    st.sidebar.title("Family Size")
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    min_fsize = min(data['FamilySize'])
    max_fsize = max(data['FamilySize'])
    FamilySize = st.sidebar.slider("What is your family size: ", min_value=round(min_fsize), max_value=round(max_fsize), value=1, step=1)

    st.sidebar.title("Embarked")
    Embarked = st.sidebar.selectbox("Choose your embarked: ", ('Cherbourg', 'Queenstown','Southampton')) # 0:C, 1:Q , 2:S
    titanic_gate = {'Cherbourg':0, 'Queenstown':1, 'Southampton':2}

    st.sidebar.title("ML Model")
    Model = st.sidebar.selectbox("Modele", ('Random Forest', 'Regression logistique', 'KNN', 'LinearSVC', 'SVC', 
    'LinearRegression', 'LogisticRegression', 'GaussianNB', 'MultinomialNB', 'ComplementNB', 'BernoulliNB'))

    if Sexe == "M":
        Sexe_id = 0
    else:
        Sexe_id = 1


    #dataframe user
    input_user = np.array([Sexe_id, Class, Age, FamilySize, Fare, titanic_gate[Embarked]]).reshape(1,-1)
    df_user = pd.DataFrame(input_user, columns = ['Sex','Class','Age','FamilySize','Fare','Embarked']) 

    #classifier
    classifier=classifiers(Model)

    return df_user, classifier

# Application with model
def modele():

    df_user, classifier=input_data()

    # on crée un pipeline de traitement intégrant la préparation
    pipeline_ml = classifier

    # on sépare la cible du reste des données
    data = load_data()
    X = data.drop(['Survived','Title','SibSp', 'Parch'],axis=1)
    y = data['Survived']

    # on construit les échantillons d'apprentissage et de validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # on ajuste le modèle en utilisant les données d'apprentissage
    # le modèle comporte la préparation et le modèle logistique
    pipeline_ml.fit(X_train, y_train)
    y_pred=pipeline_ml.predict(X_test)
    
	# metrics
    score =accuracy_score(y_test, y_pred)
    report= confusion_matrix(y_test, y_pred)

    # Probabilité de survie calculée par le modèle
    user_prediction = pipeline_ml.predict_proba(df_user)

    return user_prediction, score, report 

# Prediction
user_prediction, score, report = modele()

st.subheader('Model Metrics')
st.text("Accuracy")
st.write(round(score,2),"%")
st.text("Report: ")
st.write(report)  
    
if survived[0] == 1:
	st.success("The person has  {}% of being alive.".format(round(user_prediction[0,1]*100,2)))
else:
	st.error("The person has  {}% of being dead.".format(round(user_prediction[0,0]*100,2)))

# About

st.subheader("About App")
st.text("Titanic Dataset EDA App")
st.text("Built with Streamlit")

st.text("by SIMPLON")


