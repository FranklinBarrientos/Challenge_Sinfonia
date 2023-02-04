import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_option_menu import option_menu

import shap

from Scripts.Sinfonia_preprocessing import Sinfonia
from Scripts.Sinfonia_ML import ML_LogisticRegression
from Scripts.Functions import *

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

####################################################################################################################
########################################  Seccion de data: Extracción    ###########################################
####################################################################################################################

data = pd.read_csv("DATA/DATA.csv")
dist = pd.read_excel("DATA/distance_tables.xlsx", sheet_name = 'distance', index_col='Unnamed: 0')
neigh = pd.read_excel("DATA/distance_tables.xlsx", sheet_name = 'neighboor', index_col='Unnamed: 0')


####################################################################################################################
######################################  Seccion de ML: Modelo de Fugas    #########################################
####################################################################################################################

preprocess = Sinfonia(df = data, distancia_data = dist, neighboor_data = neigh)

preprocess.Cleaning()
data_encoding = preprocess.FeatureEngineering()
data_final = preprocess.Encoding()

model = ML_LogisticRegression(df = data_final)
(X_train, X_test, y_predict, logistic_accuracy, shap_values_log) = model.Prediction()


####################################################################################################################
################################  Seccion de ML: Perfilamiento de Estudiantes    ###################################
####################################################################################################################

data_perfil_estudiante = preprocess.PerfilEstudiante(filename = "Models/decision_tree.pickle")

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
            options    = ["Main Page", "Perfil de Estudiante", "Modelo de Fuga de Estudiantes"],
            icons      = ['house', 'people', 'book'],
            menu_icon  = 'cast',
            default_index = 0,
            styles={
                "container"        : {"padding": "5!important", "background-color": "#fafafa"},
                "icon"             : {"color": "red", "font-size": "20px"}, 
                "nav-link"         : {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#1EA4D9"},
            }
        )

if selected == 'Main Page':

    st.markdown("""<style>.big-font {font-size:55px !important;}</style>""", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image("Logo/logo_spp.png", width=400)
    with col2:
        st.markdown('<p class="big-font">Sinfonía Por El Perú</p>', unsafe_allow_html=True)


    row1_1, row1_2 = st.columns([1, 2])
    with row1_1:

        df_plot1 = plot_Fig1(data_encoding)

        fig1 = go.Figure(data=[go.Bar( 
                            x=df_plot1['Género'], 
                            y=df_plot1['Cantidad de beneficiarios'],
                            text = df_plot1['Cantidad de beneficiarios'],
                            textposition='auto')
        ])
        fig1.update_traces(marker_color=['#D90B1C', '#1EA4D9'])

        st.subheader("Cantidad de beneficiarios por Género")
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True)

    with row1_2:

        (df_plot3_1, df_plot3_2) = plot_Fig3(data_encoding)

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

        fig3.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))

        st.subheader("Cantidad de Ingresos/Salidas durante todos los meses")
        st.plotly_chart(fig3, theme="streamlit", use_container_width=True)

    
    row2_1, row2_2 = st.columns(2)
    with row2_1:

        df_plot2 = plot_Fig2(data_encoding)

        fig2 = go.Figure(data=[go.Pie(
                                labels=list(df_plot2['Tag']), 
                                values=list(df_plot2['Cantidad de beneficiarios']), 
                                hole=.5)
        ])
        fig2.update_traces(marker=dict(colors=['#1EA4D9', '#8C1F85', '#8FBF26']))

        st.subheader("Cantidad de beneficiarios por Edad")
        st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

    with row2_2:

        df_plot4 = plot_Fig4(data_encoding)

        fig4 = px.pie(df_plot4, values='total', names='index')
        
        st.subheader("Cantidad de estudiantes por centros de estudios")
        st.plotly_chart(fig4, theme="streamlit", use_container_width=True)        


if selected == 'Perfil de Estudiante':
    st.title(f"Perfil de Estudiante")

    cluster = ['Todos', 'Perfil 1', 'Perfil 2', 'Perfil 3', 'Perfil 4']
    cl = st.selectbox('Seleccione el Perfil del estudiante a analizar', cluster, help = 'Filtrar el reporte para mostrar unicamente un perfil') 
    definition = {'Todos': 'Los estudiantes principalmente lo conforman mujeres donde en su mayoria pertenecen a escuelas privadas ubicadas en distritos distintos al de sus viviendas', 
                  'Perfil 1': 'Estudiantes pertenecientes a escuelas privadas con pocos hermanos orientados a actividades artisticas y relacionadas al conocimiento', 
                  'Perfil 2': 'Estudiantes que no poseen un hobbie en particular asu vez cuyo centro de estudios en el mayor de los casos son escuelas públicas pertenecientes a un distrito diferente del cual proceden y en cuyas familias presentan por lo menos un hermano', 
                  'Perfil 3': 'Estudiantes principalmente compuesto por mujeres pertenecientes a escuelas privadas con elevado interes en actividades orientados al Arte y el conocimiento, donde en cuyas familias en el mayoria de los casos son hijos únicos', 
                  'Perfil 4': 'Estudiantes pertencientes a escuelas privadas orientadas principalmente a actividades relacionados al conocimiento y los deportes donde y quien dentro de su nucleo familiar son hijos unicos'}

    with st.spinner('Actualizando...'):
        
        kpi1, kpi2, kpi3, kpi4 = st.columns((1,1,1,1))

        if pd.isnull(cl) or cl == 'Todos':
            db = data_perfil_estudiante
        else:
            db = data_perfil_estudiante.query('cluster == "{}"'.format(cl))

        kpi1.metric(label="Cantidad de estudiantes", value = len(db.index))
        kpi2.metric(label="Edad Promedio",      value = round(int(db['Edad'].mean()), 2))
        kpi3.metric(label="Grado Academico Promedio",    value = round(db['Grado_estudios'].median(), 2))
        kpi4.metric(label="Porcentage de estudiantes activos", value = round(db['Estado_del_beneficiario'].sum()/len(db)*100, 2))

        c1, c2 = st.columns(2)

        with c1:
            
            db2 = plot_Fig5(db)

            fig_spy = go.Figure()
            fig_spy.add_trace(go.Scatterpolar(
                  r=db2.mean()*5,
                  theta=db2.columns,
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
            (men_df, women_df) = plot_Fig6(db)

            layout = go.Layout(yaxis = go.layout.YAxis(title='Age',autorange=True),
                               xaxis = go.layout.XAxis(title='Cantidad de estudiantes',autorange=True),
                               barmode = 'overlay',
                               bargap  = 0.1)
            
            data = [go.Bar(y=men_df['Edad'],
                           x=men_df['Cantidad'],
                           orientation='h',
                           name='Men',
                           hoverinfo='x',
                           marker=dict(color='seagreen')
                           ),
                    go.Bar(y=women_df['Edad'],
                           x=women_df['Cantidad'],
                           orientation='h',
                           name='Women',
                           text=-1 * women_df['Cantidad'].astype('int'),
                           hoverinfo='text',
                           marker=dict(color='powderblue')
                           )]

            figAge = go.Figure(data, layout = layout)

            st.subheader('Edad de la Poblacion por Sexo')
            st.plotly_chart(figAge, theme="streamlit", use_container_width=True)

        with gr2:

            db2 = plot_Fig7(db)

            fig_ge = px.histogram(db2, x="Grado_estudios", text_auto = True)
            fig_ge.update_layout(bargap=0.2)

            st.subheader('Grado Academico')
            st.plotly_chart(fig_ge, theme="streamlit", use_container_width=True)

        with gr3:
            
            db2 = plot_Fig8(db)

            fig_ge = px.histogram(db2, x="Programa_musical", text_auto = True)
            fig_ge.update_layout(bargap=0.2)

            st.subheader('Programa Musical')
            st.plotly_chart(fig_ge, theme="streamlit", use_container_width=True)


if selected == 'Modelo de Fuga de Estudiantes':

    st.title(f"Modelo de Fuga de Estudiantes")

    col1, col2 = st.columns([1,2])

    with col1:
        df_plot9 = plot_Fig9(y_predict)

        fig9 = go.Figure(data=[go.Pie(
                                labels=list(df_plot9['Tag']), 
                                values=list(df_plot9['Cantidad']), 
                                hole=.5)
        ])
        fig9.update_traces(marker=dict(colors=['royalblue', 'red'])) 

        st.subheader("Cantidad de beneficiarios que se podrían retirar")
        st.plotly_chart(fig9,theme="streamlit", use_container_width=True)      

    with col2:

        fig10 = shap.summary_plot(shap_values_log, plot_type='bar', max_display = 5, show=False)
        plt.xlabel('Magnitud')
        
        st.subheader("Variables que más afectan al retiro de beneficiarios")
        st.pyplot(fig10) 

    dataFinal = plot_Fig11(X_test,data_final,data_perfil_estudiante)

    cm = sns.light_palette("xkcd:copper", as_cmap=True)
    st.dataframe(dataFinal.style.background_gradient(cmap=cm, subset = ['Prediction Probability of 1'])
                          .format({'Prediction Probability of 1': "{:.2%}"}))

    
    csv = convert_df(dataFinal)

    st.download_button(
        "Descargar data",
        csv,
        "data_prueba.csv",
        "text/csv",
        )