import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_option_menu import option_menu

import shap
import pickle

from Sinfonia_preprocessing import Sinfonia
from Sinfonia_ML import ML_LogisticRegression

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

data = pd.read_csv("DATA/DATA.csv")

preprocess = Sinfonia(df = data)

preprocess.Cleaning()
data_encoding = preprocess.FeatureEngineering()
data_final = preprocess.Encoding()

##########################################################
# Para el dashboard deberia modelar o cargar los modelos?
##########################################################

model = ML_LogisticRegression(df = data_final)
(X_train, X_test, y_predict, logistic_accuracy, shap_values_log) = model.Prediction()

####################################################################################################################
###################################### Gráficos: Desarrollo de Gráficos   ##########################################
####################################################################################################################

# -- Figura 1: Cantidad de beneficiarios por Género

df1 = data_encoding[data_encoding['Estado del beneficiarios'] == 'ACTIVO'][['Estado del beneficiarios','Sexo']]
df_plot1 = df1.groupby(by='Sexo').count().reset_index()
df_plot1.columns = ['Género', 'Cantidad de beneficiarios']

cantidad_beneficiarios = df1['Estado del beneficiarios'].count()

fig1 = go.Figure(data=[go.Bar( 
                    x=df_plot1['Género'], 
                    y=df_plot1['Cantidad de beneficiarios'],
                    text = df_plot1['Cantidad de beneficiarios'],
                    textposition='auto')
])
fig1.update_traces(marker_color=['#D90B1C', '#1EA4D9'])
#fig1.update_layout(title_text='Cantidad de beneficiarios por Género', barmode='group')
#fig1.show()

# -- Figura 2: Cantidad de beneficiarios por Edad

df2 = data_encoding[data_encoding['Estado del beneficiarios'] == 'ACTIVO'][['Estado del beneficiarios','Edad']]
df2['Tag'] = np.where(df2['Edad'] < 9, 'Menor a 9 años', np.where(df2['Edad'] > 15,'Mayor a 15 años','Entre 9 y 15 años'))

df_plot2 = df2[['Tag','Estado del beneficiarios']].groupby(by='Tag').count().reset_index()
df_plot2.columns = ['Tag','Cantidad de beneficiarios']

fig2 = go.Figure(data=[go.Pie(
                        labels=list(df_plot2['Tag']), 
                        values=list(df_plot2['Cantidad de beneficiarios']), 
                        hole=.5)
])
fig2.update_traces(marker=dict(colors=['#1EA4D9', '#8C1F85', '#8FBF26']))
#fig2.update_layout(title_text='Cantidad de beneficiarios por Edad')
#fig2.show()

# -- Figura 3: Cantidad de Ingresos/Salidas durante todos los meses

df3 = data_encoding[['Año Ingreso','Mes Ingreso','Mes Ingreso Num','Año Retiro','Mes Retiro','Mes Retiro Num','Estado del beneficiarios']]

group1 = df3[['Mes Ingreso','Mes Ingreso Num','Estado del beneficiarios']].groupby(by=['Mes Ingreso', 'Mes Ingreso Num']).count().reset_index()
group1['Tag'] = np.full(len(group1),'Ingresos')
group1.columns = ['Mes','Mes Num','Cantidad de beneficiarios','Tag']

group2 = df3[['Mes Retiro','Mes Retiro Num','Estado del beneficiarios']].groupby(by=['Mes Retiro','Mes Retiro Num']).count().reset_index()
group2['Tag'] = np.full(len(group2),'Salidas')
group2.columns = ['Mes','Mes Num','Cantidad de beneficiarios','Tag']
group2['Cantidad de beneficiarios'] = group2['Cantidad de beneficiarios']*(-1)


df_plot3_1 = group1.sort_values(by='Mes Num')
df_plot3_2 = group2.sort_values(by='Mes Num')

fig3 = go.Figure(data=[
    go.Bar(name = 'Ingresos', 
            x = df_plot3_1['Mes'], 
            y = df_plot3_1['Cantidad de beneficiarios'],
            text = df_plot3_1['Cantidad de beneficiarios'],
            textposition='auto',
            marker_color='#1EA4D9'
            ),
    go.Bar(name = 'Salidas', 
            x = df_plot3_2['Mes'], 
            y = df_plot3_2['Cantidad de beneficiarios'],
            text = df_plot3_2['Cantidad de beneficiarios'],
            textposition = 'auto',
            marker_color = '#D90B1C'
            )
])
#fig3.update_layout(title_text='Cantidad de Ingresos/Salidas durante todos los meses', barmode='group')
fig3.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
#fig3.show()

# -- Figura 4: Cantidad de beneficiarios que se podrían retirar

df4 = data_encoding[data_encoding['Estado del beneficiarios'] == 'ACTIVO'][['Programa musical','Estudia actualmente','Beneficiario trabaja de manera remunerada?',
                                                                            'El beneficiario tiene alguna discapacidad? Especificar','Ha tenido anemia?',
                                                                            'Instrumento propio','Cantidad de instrumentos prestados','Estado del beneficiarios']]

group3 = df4[['Programa musical','Estado del beneficiarios']].groupby(by='Programa musical').count().reset_index()
group4 = df4[['Estudia actualmente','Estado del beneficiarios']].groupby(by='Estudia actualmente').count().reset_index()
group5 = df4[['Beneficiario trabaja de manera remunerada?','Estado del beneficiarios']].groupby(by='Beneficiario trabaja de manera remunerada?').count().reset_index()
group6 = df4[['El beneficiario tiene alguna discapacidad? Especificar','Estado del beneficiarios']].groupby(by='El beneficiario tiene alguna discapacidad? Especificar').count().reset_index()
group7 = df4[['Ha tenido anemia?','Estado del beneficiarios']].groupby(by='Ha tenido anemia?').count().reset_index()
group8 = df4[['Instrumento propio','Estado del beneficiarios']].groupby(by='Instrumento propio').count().reset_index()
group9 = df4[['Cantidad de instrumentos prestados','Estado del beneficiarios']].groupby(by='Cantidad de instrumentos prestados').count().reset_index()

cantidad_kinder     = group3[group3['Programa musical'] == 'KINDER']['Estado del beneficiarios'].sum()
cantidad_coro       = group3[group3['Programa musical'] == 'CORO']['Estado del beneficiarios'].sum()
cantidad_orquesta   = group3[group3['Programa musical'] == 'ORQUESTA']['Estado del beneficiarios'].sum()
cantidad_noestudian = group4[group4['Estudia actualmente'] == 'NO']['Estado del beneficiarios'].sum()
cantidad_trabajan   = group5[(group5['Beneficiario trabaja de manera remunerada?'] != 'NO') &
                             (group5['Beneficiario trabaja de manera remunerada?'] != 'NINGUNA') &
                             (group5['Beneficiario trabaja de manera remunerada?'] != 'NO APLICA')]['Estado del beneficiarios'].sum()
cantidad_discapacidad          = group6['Estado del beneficiarios'].sum() - group6[group6['El beneficiario tiene alguna discapacidad? Especificar'] == 'NO APLICA']['Estado del beneficiarios'].sum()
cantidad_anemia                = group7[group7['Ha tenido anemia?']  == 'SI']['Estado del beneficiarios'].sum()
cantidad_sininstrumento        = group8[group8['Instrumento propio'] != 'SI']['Estado del beneficiarios'].sum()
cantidad_instrumentosprestados = group9[group9['Cantidad de instrumentos prestados'] != 0]['Estado del beneficiarios'].sum()        



df5 = pd.DataFrame(y_predict, columns = ['Cantidad'])
df5['Tag'] = df5['Cantidad'].apply(lambda x: 'PERMANECE ACTIVO' if x == 1 else 'POSIBLE INACTIVO')
df_plot5 = df5.groupby(by='Tag').count().reset_index()

fig4 = go.Figure(data=[go.Pie(
                        labels=list(df_plot5['Tag']), 
                        values=list(df_plot5['Cantidad']), 
                        hole=.5)
])
fig4.update_traces(marker=dict(colors=['royalblue', 'red']))
#fig4.update_layout(title_text='Cantidad de beneficiarios que se podrían retirar')
#fig4.show()

# -- Figura 5: Top 5 de variables que más afectan al retiro de beneficiarios

#plt.title('Top 5 de variables que más afectan al retiro de beneficiarios')
fig5 = shap.summary_plot(shap_values_log, plot_type='bar', max_display = 5)
plt.xlabel('Magnitud')
#plt.show()

# -- Figura 5: Top 5 de variables que más afectan al retiro de beneficiarios
fig6 = px.pie(data_encoding['Tipo de centro de estudios'].value_counts().to_frame(name = 'total').reset_index(), values='total', names='index')

####################################################################################################################
################################  Seccion de ML: Perfilamiento de Estudiantes    ###################################
####################################################################################################################

# file name, I'm using *.pickle as a file extension
filename = "decision_tree.pickle"

# load model
km = pickle.load(open(filename, "rb"))

data_encoding = data_encoding.dropna().copy()
data_cl = data_encoding.assign(eje_estudiante1  = np.where(data_encoding['Cantidad de hermanos'] == 0, 4,
                                                  np.where(data_encoding['Cantidad de hermanos'] <= 2, 2.5,
                                                  np.where(data_encoding['Cantidad de hermanos'] <= 3, 1.5,
                                                  np.where(data_encoding['Cantidad de hermanos'] <= 4, 1,
                                                  np.where(data_encoding['Cantidad de hermanos'] <= 5, 0.5, 0))))),
                                eje_estudiante2 = np.where(data_encoding['Edad'] <= 8,  1.5,
                                                  np.where(data_encoding['Edad'] <= 9,  2,
                                                  np.where(data_encoding['Edad'] <= 11, 3,
                                                  np.where(data_encoding['Edad'] <= 14, 2.5,
                                                  np.where(data_encoding['Edad'] <= 17, 1.5,
                                                  np.where(data_encoding['Edad'] <= 20, 1, 0.5)))))),
                                eje_estudiante3 = np.where(data_encoding['Sexo'] == 'F', 1.5, 1),
                                eje_estudiante4 = np.where(data_encoding['Tipo de centro de estudios'] == 'PUBLICO', 1,
                                                  np.where(data_encoding['Tipo de centro de estudios'] == 'PRIVADO', 2,
                                                  np.where(data_encoding['Tipo de centro de estudios'] == 'PARROQUIAL', 0.5, 0))),
                                eje_estudiante5 = np.where(data_encoding['Hobbies'] == 'ARTE', 2,
                                                  np.where(data_encoding['Hobbies'] == 'CONOCIMIENTO', 1.5,
                                                  np.where(data_encoding['Hobbies'] == 'DEPORTE', 1, 0.5))),
                                eje_estudiante6 = np.where(data_encoding['Estudia actualmente'] == 'SI', 2, 1),
                                eje_programa1   = np.where(data_encoding['Dias_en_SPP'] <= 500, 3,
                                                  np.where(data_encoding['Dias_en_SPP'] <= 1000, 1.5,
                                                  np.where(data_encoding['Dias_en_SPP'] <= 2000, 1, 0.5))),
                                eje_programa2   = np.where(data_encoding['Programa musical'] == 'CORO', 3,
                                                  np.where(data_encoding['Programa musical'] == 'ORQUESTA', 1.5,
                                                  np.where(data_encoding['Programa musical'] == 'KINDER', 1, 0.5))),
                                eje_programa3   = np.where(data_encoding['Grupo'] == 'FORMACION INFANTIL', 3,
                                                  np.where(data_encoding['Grupo'] == 'FORMACION JUVENIL', 1,
                                                  np.where(data_encoding['Grupo'] == 'KINDER', 1.1, 0.5))),
                                eje_programa4   = np.where(data_encoding['Transcion_domicilio_colegio'] == 1, 0.5, 1.5),
                                eje_programa5   = np.where(data_encoding['Proximidad_domicilio_colegio'] == 1, 1.5, 0.5),
                                eje_programa6   = np.where(data_encoding['Tiene beca de estudios?'] == 'NO', 1, 2),
                                ).filter(regex = 'eje*').copy()

data_cl  = data_cl.assign(eje_estudiante = data_cl.filter(regex = 'eje_estudiante*').sum(axis=1),
                          eje_programa   = data_cl.filter(regex = 'eje_programa*').sum(axis=1))\
                  .filter(['eje_estudiante', 'eje_programa'])

# Predict cluster
y_km = km.predict(data_cl)

data_encoding['cluster'] = y_km 

data_encoding = data_encoding.assign(cluster =   np.where(data_encoding['cluster'] == 0, 'Perfil 1',
                                                 np.where(data_encoding['cluster'] == 1, 'Perfil 1', 
                                                 np.where(data_encoding['cluster'] == 6, 'Perfil 1',
                                                 np.where(data_encoding['cluster'] == 2, 'Perfil 2',
                                                 np.where(data_encoding['cluster'] == 5, 'Perfil 2',
                                                 np.where(data_encoding['cluster'] == 3, 'Perfil 3',
                                                 np.where(data_encoding['cluster'] == 4, 'Perfil 4', 'None'))))))))

dummy_columns = ['Estado del beneficiarios', 'Programa musical', 'Grupo', 'Sexo',
                 'Tipo de centro de estudios', 'Hobbies']

data_encoding = pd.get_dummies(data_encoding, prefix_sep='.', columns=dummy_columns)

data_encoding.rename(
    columns={"Sexo.M": "Masculino", "Sexo.F": "Femenino", "Hobbies.ARTE": "Arte", "Hobbies.CONOCIMIENTO": "Conocimiento", "Hobbies.DEPORTE": "Deporte",
             "Hobbies.OTROS": "Otros", "Tipo de centro de estudios.PUBLICO": "Colegio_publico", "Tipo de centro de estudios.PRIVADO": "Privado"},
    inplace=True,
)

columnas = ['Masculino', 'Cantidad de hermanos', 'Transcion_domicilio_colegio', 
            'Arte', 'Conocimiento', 'Deporte', 'Colegio_publico']

dbase_norm = data_encoding[columnas].apply(lambda x: x/max(x))
dbase_norm['cluster'] = y_km
dbase_norm = dbase_norm.assign(cluster = np.where(dbase_norm['cluster'] == 0, 'Perfil 1',
                                         np.where(dbase_norm['cluster'] == 1, 'Perfil 1', 
                                         np.where(dbase_norm['cluster'] == 6, 'Perfil 1',
                                         np.where(dbase_norm['cluster'] == 2, 'Perfil 2',
                                         np.where(dbase_norm['cluster'] == 5, 'Perfil 2',
                                         np.where(dbase_norm['cluster'] == 3, 'Perfil 3',
                                         np.where(dbase_norm['cluster'] == 4, 'Perfil 4', 'None'))))))))


def complete_age(age, amount):
    age = list(age)
    new_amount = []
    for i in range(0, 31):
        if i in age:
            new_amount.append(amount[age.index(i)])
        else:
            new_amount.append(0)    

    return np.array(new_amount)

def color_survived(val):
    color = 'green' if val >= 0.9 else 'red'
    return f'background-color: {color}'

####################################################################################################################
################################  Seccion de Dashboard: Desarrollo de Interfaz   ###################################
####################################################################################################################


st.set_page_config(page_title            = 'Sinfonia Por El Peru',
				   page_icon             = ':bar_chart:',
				   layout                = 'wide',
                   initial_sidebar_state = 'expanded')





## ---------------------------------------------------------------------------------------------------


with st.sidebar:
    selected = option_menu(
            menu_title = "Dashboard",
            options    = ["Main Page", "Cluster", "ML Analisis"],
            icons      = ['house', 'people', 'book'],
            menu_icon  = 'cast',
            default_index = 0,
            styles={
                "container"        : {"padding": "5!important", "background-color": "#fafafa"},
                "icon"             : {"color": "red", "font-size": "25px"}, 
                "nav-link"         : {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#1EA4D9"},
    }
        )
if selected == 'Main Page':

    st.markdown("""
    <style>
    .big-font {
        font-size:70px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    a, b = st.columns([2, 1])
    #a, b = st.columns((0.07,1))

    with a:
        #st.text("")
        st.image("logo_spp.png", width=400)
    with b:
        #st.header("Sinfonía Por El Perú")
        st.markdown('<p class="big-font">Sinfonía Por El Perú</p>', unsafe_allow_html=True)

    row1_1, row1_2 = st.columns([1, 2])

    with row1_1:
        st.subheader("Cantidad de beneficiarios por Género")
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True)

    with row1_2:
        st.subheader("Cantidad de Ingresos/Salidas durante todos los meses")
        st.plotly_chart(fig3, theme="streamlit", use_container_width=True)

    row2_1, row2_2, row2_3 = st.columns(3)

    with row2_1:
        st.subheader("Cantidad de beneficiarios por Edad")
        st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

    with row2_2:
        st.subheader("Cantidad de beneficiarios que se podrían retirar")
        st.plotly_chart(fig4, theme="streamlit", use_container_width=True)


    with row2_3:
        st.subheader("Distribucion de estudiantes por centros de estudios")
        st.plotly_chart(fig6, theme="streamlit", use_container_width=True)        


    #st.title(f"You have selected {selected}")
if selected == 'Cluster':
    st.title(f"You have selected {selected}")

    cluster = ['Todos', 'Perfil 1', 'Perfil 2', 'Perfil 3', 'Perfil 4']
    cl = st.selectbox('Seleccione el cluster', cluster, help = 'Filtrar el reporte para mostrar unicamente un perfil') 
    definition = {'Todos': 'Los estudiantes principalmente lo conforman mujeres donde en su mayoria pertenecen a escuelas privadas ubicadas en distritos distintos al de sus viviendas', 
                  'Perfil 1': 'Estudiantes pertenecientes a escuelas privadas con pocos hermanos orientados a actividades artisticas y relacionadas al conocimiento', 
                  'Perfil 2': 'Estudiantes que no poseen un hobbie en particular asu vez cuyo centro de estudios en el mayor de los casos son escuelas públicas pertenecientes a un distrito diferente del cual proceden y en cuyas familias presentan por lo menos un hermano', 
                  'Perfil 3': 'Estudiantes principalmente compuesto por mujeres pertenecientes a escuelas privadas con elevado interes en actividades orientados al Arte y el conocimiento, donde en cuyas familias en el mayoria de los casos son hijos únicos', 
                  'Perfil 4': 'Estudiantes pertencientes a escuelas privadas orientadas principalmente a actividades relacionados al conocimiento y los deportes donde y quien dentro de su nucleo familiar son hijos unicos'}
    ## Data

    with st.spinner('Actualizando...'):
        
        # create three columns
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns((1,1,1,1,1))

        # fill in those three columns with respective metrics or KPIs 
        if pd.isnull(cl) or cl == 'Todos':
            db_norm = dbase_norm
            db      = data_encoding
        else:
            db_norm = dbase_norm.query('cluster == "{}"'.format(cl))
            db      = data_encoding.query('cluster == "{}"'.format(cl))

        kpi2.metric(label="Tamaño del Cluster", value = len(db_norm.index), delta= -8)
        kpi3.metric(label="Edad Promedio",      value = round(int(db['Edad'].mean()), 2), delta= 5)
        kpi4.metric(label="Grado Academico",    value = round(db['Grado_estudios'].median(), 2), delta= -2)

        c1, c2 = st.columns(2)

        with c1:
            fig_spy = go.Figure()

            fig_spy.add_trace(go.Scatterpolar(
                  r=db_norm.drop('cluster', axis = 1).mean()*5,
                  theta=columnas,
                  fill='toself',
                  name=cl
            ))

            fig_spy.update_layout(
              polar=dict(
                radialaxis=dict(
                  visible=True,
                  range=[0, 5]
                )),
              showlegend=True
            )


            st.plotly_chart(fig_spy, theme="streamlit", use_container_width=True)

        with c2:
            st.title(cl)
            st.markdown(definition[cl])

        gr1, gr2, gr3 = st.columns(3)

        with gr1:
            st.subheader('Edad de la Poblacion por Sexo')
            y = list(range(0, 31, 1))
            

            men_bins   = complete_age(db.query('Masculino == 1')['Edad'].value_counts().index.values, db.query('Masculino == 1')['Edad'].value_counts().values)
            women_bins = complete_age(db.query('Femenino == 1')['Edad'].value_counts().index.values, db.query('Femenino == 1')['Edad'].value_counts().values*(-1))

            layout = go.Layout(yaxis = go.layout.YAxis(title='Age'),
                               xaxis = go.layout.XAxis(
                                   range    = [-60, 60],
                                   tickvals = [-50, -35, -15, 0, 15, 35, 50],
                                   ticktext = [50, 35, 15, 0, 15, 35, 50],
                                   title    = 'Number'),
                               barmode = 'overlay',
                               bargap  = 0.1)
            
            data = [go.Bar(y=y*2,
                           x=men_bins,
                           orientation='h',
                           name='Men',
                           hoverinfo='x',
                           marker=dict(color='seagreen')
                           ),
                    go.Bar(y=y,
                           x=women_bins,
                           orientation='h',
                           name='Women',
                           text=-1 * women_bins.astype('int'),
                           hoverinfo='text',
                           marker=dict(color='powderblue')
                           )]

            figAge = go.Figure(data, layout = layout)
            st.plotly_chart(figAge, theme="streamlit", use_container_width=True)

        with gr2:
            st.subheader('Grado Academico')
            db2 = db.assign(Grado_estudios =  db['Grado_estudios'].apply(lambda x: np.where(x == 0, 'INICIAL', 
                                                                                             np.where(x == 1, '1ERO PRIMARIA',
                                                                                             np.where(x == 2, '2DO PRIMARIA',  
                                                                                             np.where(x == 3, '3ERO PRIMARIA',
                                                                                             np.where(x == 4, '4TO PRIMARIA',  
                                                                                             np.where(x == 5, '5TO PRIMARIA', 
                                                                                             np.where(x == 6, '6TO PRIMARIA',  
                                                                                             np.where(x == 7, '1ERO SECUNDARIA', 
                                                                                             np.where(x == 8, '2DO SECUNDARIA', 
                                                                                             np.where(x == 9, '3ERO SECUNDARIA', 
                                                                                             np.where(x == 10, '4TO SECUNDARIA',  
                                                                                             np.where(x == 11, '5TO SECUNDARIA', 
                                                                                             np.where(x == 12, 'PREUNIVERSITARIO', 
                                                                                             np.where(x == 13, 'SUPERIOR UNIVERSITARIA', 'NONE')))))))))))))))).copy()
            fig_ge = px.histogram(db2, x="Grado_estudios", text_auto = True)
            fig_ge.update_layout(bargap=0.2)
            st.plotly_chart(fig_ge, theme="streamlit", use_container_width=True)
        with gr3:
            st.subheader('Programa Musical')
            db2 = db.assign(Programa_musical = np.where(db['Programa musical.CORO'] == 1, 'CORO',
                                               np.where(db['Programa musical.KINDER'] == 1, 'KINDER',
                                               np.where(db['Programa musical.ORQUESTA'] == 1, 'ORQUESTA', 'OTROS')))).copy()
            fig_ge = px.histogram(db2, x="Programa_musical", text_auto = True)
            fig_ge.update_layout(bargap=0.2)
            st.plotly_chart(fig_ge, theme="streamlit", use_container_width=True)


if selected == 'ML Analisis':

    st.title(f"You have selected {selected}")

    st.subheader("Variables que más afectan al retiro de beneficiarios")
    st.pyplot(fig5) 

    data = X_test.sort_values(by=['Prediction Probability of 1'], ascending=False)\
                 .drop(['Prediction Probability of 0', 'Estado del beneficiarios', 'Estado del beneficiarios predict'], axis='columns')

    dataFinal = data.merge(data_encoding.filter(['Numero de Documento de identidad', 'cluster']), on='Numero de Documento de identidad', how='left')

    columnasDescriptivas = ['Apellido Paterno', 'Apellido Materno', 'Nombres', 'Numero de Documento de identidad',
                            'Programa musical', 'Grupo', 'Nacionalidad', 'Sexo', 'Cantidad de hermanos', 'Edad', 'Dias_en_SPP',
                            'Nivel_academico', 'Grado_estudios', 'Score Education', 'Score Economic', 'Score Health', 
                            'Score Musical Interest', 'Score Total',
                            'cluster', 'Prediction Probability of 1']

    cm = sns.light_palette("xkcd:copper", as_cmap=True)

    st.dataframe(dataFinal.filter(columnasDescriptivas)
                          .style.background_gradient(cmap=cm, subset = ['Prediction Probability of 1'])
                          .format({'Prediction Probability of 1': "{:.2%}"}))

## ---------------------------------------------------------------------------------------------------


