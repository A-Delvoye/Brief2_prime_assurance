"""import streamlit as st
import time

st.title("This is the app title")
st.header("This is the header")
st.markdown("This is the markdown")
st.subheader("This is the subheader")
st.caption("This is the caption")
st.code("x = 2021")
st.latex(r''' a+a r^1+a r^2+a r^3 ''')

st.header("Image name:")
st.image("icone.jpg", caption="A kid playing")
#st.audio("audio.mp3")
#st.video("video.mp4")

st.checkbox('Yes')
st.button('Click Me')
st.radio('Pick your gender', ['Male', 'Female'])
st.selectbox('Pick a fruit', ['Apple', 'Banana', 'Orange'])
st.multiselect('Choose a planet', ['Jupiter', 'Mars', 'Neptune'])
st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
st.slider('Pick a number', 0, 50)

st.number_input('Pick a number', 0, 10)
st.text_input('Email address')
st.date_input('Traveling date')
st.time_input('School time')
st.text_area('Description')
st.file_uploader('Upload a photo')
st.color_picker('Choose your favorite color')

st.balloons()  # Celebration balloons
st.subheader("Progess bar")
st.progress(10)  # Progress barwith

st.subheader("Wait the execution") 
with st.spinner('Wait for it...'): 
    time.sleep(10)
# Simulating a process delay

st.success("You did itt!")
st.error("Error occurred")
st.warning("This is a warning")
st.info("It's easy to build a Streamlit app")
#st.exception(RuntimeError("RuntimeError exception"))

st.sidebar.title("Sidebar Title")
st.sidebar.button("Click")
#st.sidebar.radio("Pick your gender",["Male","Female"])
st.sidebar.markdown("This is the sidebar content")

with st.container():    
    st.write("This is inside the container")

container = st.container()
container.write("This is inside")
st.write("This is outside")

import matplotlib.pyplot as plt
import numpy as np

rand = np.random.normal(1, 2, size=20)
fig, ax = plt.subplots()
ax.hist(rand, bins=15)#,color="pink"
st.pyplot(fig)

import pandas as pd
df = pd.DataFrame(np.random.randn(10, 2), columns=['x', 'y'])
st.line_chart(df)


df2 = pd.DataFrame(np.random.randn(10, 2), columns=['x', 'y'])
st.bar_chart(df2)

df3 = pd.DataFrame(np.random.randn(10, 2), columns=['x', 'y'])
st.area_chart(df3)

#import altair as alt
#df4 = pd.DataFrame(np.random.randn(500, 3), columns=['x', 'y', 'z'])
#chart = alt.Chart(df).mark_circle().encode(x='x', y='y', size='z', color='z', tooltip=['x', 'y', 'z'])
#st.altair_chart(chart, use_container_width=True)

import graphviz as graphviz
st.graphviz_chart('''    digraph {
                          Big_shark -> Tuna
                          Tuna -> Mackerel
                          Mackerel -> Small_fishes
                          Small_fishes -> Shrimp    }
                  ''')

import pickle  # to load a saved model
import base64  # to handle gif encoding

@st.cache_data 
def get_fvalue(val):    
    feature_dict = {"No": 1, "Yes": 2}    
    return feature_dict[val]

def get_value(val, my_dict):    
    return my_dict[val]

if app_mode == 'Home':    
    st.title('Loan Prediction')    
    st.image('loan_image.jpg')    
    st.markdown('Dataset:')    
    data = pd.read_csv('loan_dataset.csv')    
    st.write(data.head())    
    st.bar_chart(data[['ApplicantIncome', 'LoanAmount']].head(20))


        st.title('Prediction')
    st.write('App')
    st.header('Image name:')
    st.image('icone.jpg', caption="A kid playing")
    st.markdown('Dataset :')
    data = pd.read_csv("data_.csv")
    st.write(data.head())
    st.markdown('Application Assurance')
    st.bar_chart(data[["age","sex"]].head(20))
    """
import streamlit as st
import time
import pandas as pd
import numpy as np
import pickle
import base64

@st.cache_data() 
def get_fvalue(val):    
    feature_dict = {"No": 1, "Yes": 2}
    for key, value in feature_dict.items():    
        if val == key:
            return value
        
def get_value(val, my_dict):
    for key, value in my_dict.items():    
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page', ['Presentation', 'Prediction'])
if app_mode == 'Presentation':

    # Charger le modèle
    
    with open("linear_regression_model_1.pkl", "rb") as file:
        model_pipeline = pickle.load(file)

    # Titre de l'application
    st.title("Affichage du Modèle et de ses Coefficients")

    # Vérifier si le modèle contient une étape de régression linéaire
    try:
        # Accès au modèle final (dans le pipeline)
        linear_model = model_pipeline.named_steps['linearregression']  # Adapter en fonction du nom de l'étape du modèle
    except AttributeError:
        st.error("Impossible de trouver l'étape de régression dans le pipeline. Vérifiez le fichier chargé.")
        linear_model = None

    # Si le modèle est un modèle de régression linéaire, afficher les coefficients
    if linear_model:
        st.header("Modèle et Coefficients")
        
        # Afficher le type du modèle
        st.write("**Modèle chargé :*****", type(linear_model).__name__)
        
        # Récupérer les coefficients
        if hasattr(linear_model, 'coef_') and hasattr(linear_model, 'intercept_'):
            coefficients = linear_model.coef_
            intercept = linear_model.intercept_
            st.write("Hello",model_pipeline.named_steps)
            # Afficher les coefficients
            st.write("### Coefficients du modèle")
            coef_df = pd.DataFrame({
                'Feature': model_pipeline.named_steps['polynomialfeatures'].get_feature_names_out(),
                'Coefficient': coefficients
            })
            st.dataframe(coef_df)

            # Afficher l'intercept
            st.write("### Intercept")
            st.write(intercept)
        else:
            st.error("Le modèle chargé n'a pas de coefficients (ce n'est peut-être pas un modèle de régression linéaire).")


if app_mode == 'Prediction':    
    # Charger le modèle
    with open('linear_regression_model_1.pkl', 'rb') as file:
        model = pickle.load(file)

        st.write("Étapes du pipeline :", model.named_steps)

        st.title("Prédiction avec un Modèle de Régression Linéaire")
        # Entrée utilisateur
        st.header("Entrer les caractéristiques")
        # Variables numériques
        age = st.number_input("Âge:", min_value=0, max_value=100, step=1)
        bmi = st.number_input("Indice de Masse Corporelle (BMI):", min_value=10.0, max_value=50.0, step=0.1)
        children = st.number_input("Nombre d'enfants:", min_value=0, max_value=10, step=1)

        # Variables catégoriques
        sex = st.selectbox("Sexe:", options=["male", "female"])
        smoker = st.selectbox("Fumeur:", options=["yes", "no"])

        # Variables ordinales
        region = st.selectbox(
            "Région:", 
            options=["southeast", "northwest", "northeast", "southwest"]
        )

        # Convertir les variables catégoriques et ordinales en format numérique
        # Remplacez ces mappings par ceux utilisés dans votre prétraitement
        sex_mapping = {"male": "male", "female":"female"}
        smoker_mapping = {"yes": "yes", "no": "no"}
        region_mapping = {"southeast":"southeast", "northwest":"northwest", "northeast":"northeast", "southwest":"southwest"}

        sex = sex_mapping[sex]
        smoker = smoker_mapping[smoker]
        region = region_mapping[region]

        input_data = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }

        # Ajouter un bouton pour effectuer une prédiction
        if st.button("Prédire"):
            input_features = pd.DataFrame([input_data])
            st.write(input_features.head())

            st.write("Données d'entrée :")
            st.write(input_features)

            # Combiner toutes les caractéristiques en un tableau
            #input_features = pd.DataFrame([age, sex, bmi, children, smoker, region])
            input_features = pd.DataFrame([input_data])
            # Faire une prédiction
            prediction = model.predict(input_features)
            
            # Afficher le résultat
            st.success(f"Prédiction : {prediction[0]:.2f}")