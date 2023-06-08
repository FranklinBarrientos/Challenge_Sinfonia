import unicodedata
import datetime
import calendar
import numpy as np
import pandas as pd

score_education_total = 1 + 4 + 4 + 2 + 3
score_economic_total = 2 + 1 + 1 + 2 + 1
score_health_total = 2 + 5
score_interest_total = 3

def color_survived(val):
    color = 'green' if val >= 0.9 else 'red'
    return f'background-color: {color}'

def Normalize_data(x):
    valor_string = unicodedata.normalize('NFKD',str(x))
    valor_string = valor_string.encode('ASCII','ignore')
    valor_string = valor_string.decode('utf-8').upper().strip()

    if valor_string == 'NAN':
        valor_string = 'NO APLICA'
    
    return valor_string.upper().strip()

def Location(x,y):

    if ((x == 'LIMA') or (x == 'CALLAO')) or ((y == 'LIMA') or (y == 'CALLAO')):
        return True
    else:
        return False

def Validate_date(x):
    if x != 'NO APLICA' :
        try:
            return datetime.datetime.strptime(x, "%d/%m/%Y")
        except:
            return datetime.datetime.strptime(x, "%m/%d/%Y")
    else:
        return datetime.date.today()


def Nacionalidad(x):
    if ('PERU' in x) or ('PEUANA' in x):
        nacionalidad = 'PERUANA'
    else:
        nacionalidad = 'OTRO'
    
    return nacionalidad

def Education_Score(estudia,nivel_academico,grado_estudio,centro_estudios,actividad_extracurricular):
    
    score = 0
    
    if(estudia=='SI'):
        score += 1
    
    score += nivel_academico
    
    score += grado_estudio
    
    if((centro_estudios=='PARROQUIAL') or (centro_estudios=='PUBLICO')):
        score += 1
    elif (centro_estudios=='PRIVADO'):
        score += 2

    
    if (actividad_extracurricular == 'CONOCIMIENTO'):
        score += 3
    elif (actividad_extracurricular == 'ARTE'):
        score += 2
    elif (actividad_extracurricular == 'DEPORTE'):
        score += 1
    
    return score/score_education_total

def Economic_Score(centro_estudios, beca, trabajo_remunerado, seguro_medico, instrumento_propio):

    score = 0
    
    if((centro_estudios == 'PARROQUIAL') or (centro_estudios == 'PUBLICO')):
        score += 1
    elif (centro_estudios == 'PRIVADO'):
        score += 2

    if(beca == 'SI'):
        score += 1

    if(('NO' in trabajo_remunerado) or ('NINGUNA' in trabajo_remunerado)):
        score += 1

    if(seguro_medico == 'PARTICULAR') or (seguro_medico == 'EPS'):
        score += 2
    elif(seguro_medico != 'NINGUNO'):
        score += 1
    
    if(instrumento_propio == 'SI'):
        score += 1
    
    return score/score_economic_total

def Health_Score(seguro_medico, alergias, restricciones_alimentarias, discapacidad, trastorno, anemia):

    score = 0
    
    if(seguro_medico == 'PARTICULAR') or (seguro_medico == 'EPS'):
        score += 2
    elif(seguro_medico != 'NINGUNO'):
        score += 1

    if(alergias == 'NO APLICA'):
        score += 1

    if(restricciones_alimentarias == 'NO APLICA'):
        score += 1
    
    if(discapacidad == 'NO APLICA'):
        score += 1    

    if(trastorno == 'NO APLICA'):
        score += 1
    
    if(anemia != 'SI'):
        score += 1
        
    return score/score_health_total

def Musical_Interest_Score(elenco_central, instrumento_propio, instrumento_prestado):

    score = 0
    
    if(elenco_central != 'NO APLICA'):
        score += 1
        
    if(instrumento_propio == 'SI'):
        score += 1
    
    if(instrumento_prestado != 0):
        score += 1
    
    return score/score_interest_total

def Total_Score(education,economic,health,musical_interest):

    den = score_education_total+score_economic_total+score_health_total+score_interest_total
    num = education*score_education_total + economic*score_economic_total + health*score_health_total + musical_interest*score_interest_total

    return num/den

def plot_Fig1(df, threshold_fecha):

    mask = (df['Fecha de ingreso del beneficiario a SPP'] <= threshold_fecha) & (df['Estado del beneficiarios'] == 'ACTIVO')

    df1 = df[mask].copy()[['Estado del beneficiarios','Sexo']]

    df_plot1 = df1.groupby(by='Sexo').count().reset_index()
    df_plot1.columns = ['Género', 'Cantidad de beneficiarios']

    return df_plot1

def plot_Fig2(df, threshold_fecha):
    mask = (df['Fecha de ingreso del beneficiario a SPP'] <= threshold_fecha) & (df['Estado del beneficiarios'] == 'ACTIVO')

    df2 = df[mask].copy()[['Estado del beneficiarios','Edad']]
    df2['Tag'] = np.where(df2['Edad'] < 9, 'Menor a 9 años', np.where(df2['Edad'] > 15,'Mayor a 15 años','Entre 9 y 15 años'))

    df_plot2 = df2[['Tag','Estado del beneficiarios']].groupby(by='Tag').count().reset_index()
    df_plot2.columns = ['Tag','Cantidad de beneficiarios']

    return df_plot2

def plot_Fig3(df, threshold_fecha):

    mask_df3 = df['Fecha de retiro del beneficiario'] != datetime.datetime.today().strftime('%Y-%m-%d')
    mask = (df['Fecha de ingreso del beneficiario a SPP'] <= threshold_fecha) & mask_df3

    df3 = df[mask].copy()[['Año Ingreso','Mes Ingreso','Mes Ingreso Num','Año Retiro','Mes Retiro','Mes Retiro Num','Estado del beneficiarios']]

    group1 = df3[['Mes Ingreso','Mes Ingreso Num','Estado del beneficiarios']].groupby(by=['Mes Ingreso', 'Mes Ingreso Num']).count().reset_index()
    group1['Tag'] = np.full(len(group1),'Ingresos')
    group1.columns = ['Mes','Mes Num','Cantidad de beneficiarios','Tag']

    group2 = df3[['Mes Retiro','Mes Retiro Num','Estado del beneficiarios']].groupby(by=['Mes Retiro','Mes Retiro Num']).count().reset_index()
    group2['Tag'] = np.full(len(group2),'Salidas')
    group2.columns = ['Mes','Mes Num','Cantidad de beneficiarios','Tag']
    group2['Cantidad de beneficiarios'] = group2['Cantidad de beneficiarios']*(-1)

    for mes in range(1,13):
        if not mes in group1['Mes Num'].unique():
            new_value = {'Mes':calendar.month_name[mes],'Mes Num':mes,'Cantidad de beneficiarios':0,'Tag':'Ingresos'}
            group1 = group1.append(new_value,ignore_index=True)
        
        if not mes in group2['Mes Num'].unique():
            new_value = {'Mes':calendar.month_name[mes],'Mes Num':mes,'Cantidad de beneficiarios':0,'Tag':'Salidas'}
            group2 = group2.append(new_value,ignore_index=True)

    df_plot3_1 = group1.sort_values(by='Mes Num')
    df_plot3_2 = group2.sort_values(by='Mes Num')

    return (df_plot3_1, df_plot3_2)

def plot_Fig4(df, threshold_fecha):

    mask = (df['Fecha de ingreso del beneficiario a SPP'] <= threshold_fecha)

    df4 = df[mask].copy()

    df_plot4 = df4['Tipo de centro de estudios'].value_counts().to_frame(name = 'total').reset_index()
    
    return df_plot4

def plot_Fig5(df):

    columns = ['promedio Masculino', 'promedio Cantidad de hermanos', 'promedio Transcion_domicilio_colegio', 
                'promedio Arte', 'promedio Conocimiento', 'promedio Deporte', ' promedio Colegio publico']

    return df[columns]

def plot_Fig6(df):

    men_df   = pd.DataFrame(df.query('Masculino == 1')['Edad'].value_counts().reset_index()).rename(columns={'Edad':'Cantidad','index':'Edad'})
    women_df = pd.DataFrame((df.query('Femenino == 1')['Edad'].value_counts()*(-1)).reset_index()).rename(columns={'Edad':'Cantidad','index':'Edad'})

    return (men_df,women_df)

def plot_Fig7(df):

    db2 = df.assign(Grado_estudios =  df['Grado_estudios'].apply(lambda x: np.where(x == 1, 'INICIAL', 
                                                                           np.where(x == 2.1, '1ERO PRIMARIA',
                                                                           np.where(x == 2.2, '2DO PRIMARIA',  
                                                                           np.where(x == 2.3, '3ERO PRIMARIA',
                                                                           np.where(x == 2.4, '4TO PRIMARIA',  
                                                                           np.where(x == 2.6, '5TO PRIMARIA', 
                                                                           np.where(x == 2.8, '6TO PRIMARIA',  
                                                                           np.where(x == 3, '1ERO SECUNDARIA', 
                                                                           np.where(x == 3.2, '2DO SECUNDARIA', 
                                                                           np.where(x == 3.4, '3ERO SECUNDARIA', 
                                                                           np.where(x == 3.6, '4TO SECUNDARIA',  
                                                                           np.where(x == 3.8, '5TO SECUNDARIA', 
                                                                           np.where(x == 4, 'PREUNIVERSITARIO', 
                                                                           np.where(x == 5, 'SUPERIOR UNIVERSITARIA', 'NO MENCIONA')))))))))))))))).copy()    

    return db2

def plot_Fig8(df):

    db2 = df.assign(Programa_musical =  np.where(df['Programa musical.CORO']     == 1, 'CORO',
                                        np.where(df['Programa musical.KINDER']   == 1, 'KINDER',
                                        np.where(df['Programa musical.ORQUESTA'] == 1, 'ORQUESTA', 'OTROS')))).copy()
    
    return db2

def plot_Fig9(y_predict):

    df5 = pd.DataFrame(y_predict, columns = ['Cantidad'])
    df5['Tag'] = df5['Cantidad'].apply(lambda x: 'PERMANECE ACTIVO' if x == 1 else 'POSIBLE INACTIVO')
    df_plot5 = df5.groupby(by='Tag').count().reset_index()

    return df_plot5

def plot_Fig11(X_test,df_final, df_perfil):

    df = pd.merge(X_test,df_final['Numero de Documento de identidad'], left_index=True, right_index=True, how='left')
    data = df.sort_values(by=['Prediction Probability of 1'], ascending=False)\
                    .drop(['Prediction Probability of 0', 'Estado del beneficiarios', 'Estado del beneficiarios predict'], axis='columns')
    
    dataFinal = data.merge(df_perfil.filter(['Numero de Documento de identidad', 'cluster', 'Apellido Paterno', 'Apellido Materno', 'Nombres']), 
                           on='Numero de Documento de identidad', 
                           how='left')

    columnasDescriptivas = ['Apellido Paterno', 'Apellido Materno', 'Nombres', 'Numero de Documento de identidad',
                            'Programa musical', 'Grupo', 'Nacionalidad', 'Sexo', 'Cantidad de hermanos', 'Edad', 'Dias_en_SPP',
                            'Nivel_academico', 'Grado_estudios', 'Score Education', 'Score Economic', 'Score Health', 
                            'Score Musical Interest', 'Score Total',
                            'cluster', 'Prediction Probability of 1']

    map_columns = {'Nivel_academico' : {1: 'INICIAL',
                                        2: 'PRIMARIA',
                                        3: 'SECUNDARIA',
                                        4: 'PREUNIVERSITARIO',
                                        5: 'UNIVERSITARIO',
                                        0: 'OTROS'},
                        
                   'Programa musical' : {1: 'CORO',
                                         2: 'KINDER',
                                         3: 'ORQUESTA'},
                        
                    'Grupo' : {1: 'FORMACION INFANTIL',
                               2: 'FORMACION JUVENIL',
                               3: 'KINDER'},
                        
                    'Nacionalidad' : {1: 'PERUANA',
                                      0: 'OTROS'},
                        
                    'Sexo' : {1: 'M',
                              0: 'F'},

                    'Grado_estudios' : {1:  'INICIAL',
                                        2.1:'1ERO PRIMARIA', 
                                        2.2:'2DO PRIMARIA', 
                                        2.3:'3ERO PRIMARIA', 
                                        2.4:'4TO PRIMARIA', 
                                        2.6:'5TO PRIMARIA', 
                                        2.8:'6TO PRIMARIA', 
                                        3:  '1ERO SECUNDARIA', 
                                        3.2:'2DO SECUNDARIA', 
                                        3.4:'3ERO SECUNDARIA', 
                                        3.6:'4TO SECUNDARIA', 
                                        3.8:'5TO SECUNDARIA', 
                                        4:  'PREUNIVERSITARIO', 
                                        5:  'SUPERIOR UNIVERSITARIA', 
                                        0:  'OTROS'}
                    }
        
    columns_categorical = ['Nivel_academico','Programa musical','Grupo','Nacionalidad','Sexo', 'Grado_estudios']
    for column in columns_categorical:
        dataFinal[column] = dataFinal[column].apply(lambda x: map_columns[column][x])
    dataFinal['Edad'] = dataFinal['Edad'].astype(int)

    return dataFinal.filter(columnasDescriptivas)

def plot_Fig12(df):

    mask = (df['Estado del beneficiarios'] == 'ACTIVO')

    df1 = df[mask].copy()[['Estado del beneficiarios','Fecha_ingreso']]

    df_plot1 = df1.groupby(by='Fecha_ingreso').count().reset_index()
    df_plot1.columns = ['Fecha_ingreso', 'Cantidad de beneficiarios']

    return df_plot1

def plot_Fig13(df):
    
    df2 = df.copy()[['Estado del beneficiarios','Fecha de ingreso al beneficiario al Elenco Central']]
    df2['Tag'] = np.where(df2['Fecha de ingreso al beneficiario al Elenco Central'].notnull(), 
                          'Elenco Central', 
                          'No Elenco Central')

    df_plot2 = df2[['Tag','Estado del beneficiarios']].groupby(by='Tag').count().reset_index()
    df_plot2.columns = ['Tag','Cantidad de beneficiarios']

    return df_plot2

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')