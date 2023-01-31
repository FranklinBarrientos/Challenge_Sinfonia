import re
import pandas as pd
import numpy as np
import unicodedata
import Functions

class Sinfonia():

    def __init__(self,df):
        self.data = df
        self.dist = pd.read_excel("DATA/distance_tables.xlsx", sheet_name = 'distance', index_col='Unnamed: 0')
        self.neig = pd.read_excel("DATA/distance_tables.xlsx", sheet_name = 'neighboor', index_col='Unnamed: 0')

    def Cleaning(self):

        normalize_column = []
        for columna in self.data.columns:

            valor_string = unicodedata.normalize('NFKD',columna)
            valor_string = valor_string.encode('ASCII','ignore')
            valor_string = valor_string.decode('utf-8')
            normalize_column.append(valor_string.strip())

        self.data.columns = normalize_column

        self.data['Fecha de ingreso del beneficiario a SPP'][self.data['Fecha de ingreso del beneficiario a SPP'] == '30-ago']      = '01/01/1900'
        self.data['Fecha de ingreso del beneficiario a SPP'][self.data['Fecha de ingreso del beneficiario a SPP'] == 'Agosto 2017'] = '08/01/2017' 

        drop_list = ['Apellido Paterno', 'Apellido Materno', 'Nombres', 'Tipo de sangre','Edad','Celular del beneficiario', 'Nombre de Contacto de emergencia / Cuidador(a)', 
                     'Recibio vacuna contra el COVID-19', 'Direccion de domicilio','Codigo del instrumento\n (de acuerdo al inventario)', 'Periodo/MES','Centro de estudios','Modulo', 
                     'Numero de poliza', 'Otra discapacidad (Solo llenar cuando el campo anterior es "Otro")', 'Otro trastorno (Solo llenar cuando el campo anterior es "Otro")',
                     'Toma algun medicamento? Especificar', 'Cuantas dosis tiene?', 'Tiene alguna enfermedad previa?', 'Otra enfermedad previa', 
                     'Numero de Contacto de contacto de emergencia / Cuidador (a)', 'Parentesco con el beneficiario', 'Tipo de Documento de Identidad', 'Especialidad / Registro de voz',
                     'Motivo del retiro', 'Numero de Documento de identidad', 'Cantidad de reingresos',
                     'Fecha de inactividad', 'Fecha de ingreso'] # Habilitando datos personales de los estudiantes (Apellido Paterno, Apellido Materno, Nombres, Numero de Documento de identidad)

        data_drop = self.data.drop(columns = drop_list)

#        Latitude_Longitud_v = np.vectorize(Functions.Latitude_Longitud)
        Location_v = np.vectorize(Functions.Location)
        Normalize_data_v = np.vectorize(Functions.Normalize_data)

        ## Imputation

        for columna in data_drop.columns:
            if str(data_drop[columna].dtype) == 'object':
                data_drop[columna] = Normalize_data_v(data_drop[columna])
            elif str(data_drop[columna].dtype) == 'int64':
                data_drop[columna] = data_drop[columna].fillna(0)
            elif str(data_drop[columna].dtype) == 'float64':
                data_drop[columna] = data_drop[columna].fillna(0)


        mask_location = Location_v(data_drop['Region de domicilio'],data_drop['Provincia de domicilio'])

#        (longitude, latitude) = Latitude_Longitud_v(data_drop['Region de domicilio'],
#                                                    data_drop['Provincia de domicilio'],
#                                                    data_drop['Distrito de domicilio'])
#        data_drop['Longitud'] = longitude
#        data_drop['Latitud'] = latitude

#        mask_location = data_drop['Latitud'] < -11
        mask_fechas = (data_drop['Fecha de Nacimiento'] != 'NO APLICA') & (data_drop['Fecha de ingreso del beneficiario a SPP'] != 'NO APLICA')
        mask_total  = mask_fechas & mask_location

        data_drop = data_drop[mask_total].reset_index(drop=True)

        self.data_clean = data_drop.copy()


    def FeatureEngineering(self):

        data_feature = self.data_clean.copy()

        ARTE         = ['MUSICA', 'DANZA', 'BALLET', 'GUITARRA', 'PERCUSION', 'VIOLIN', 'CAJON', 'TEATRO', 'FLAUTA', 'ARTE', 'ZUMBA', 'MUSICA', 'BAILE', 'MARINERA', 'ACURELA', 'CANTO', 'PIANO', 'DIBUJO']
        DEPORTE      = ['KARATE', 'TAEKWONDO', 'GIMNASIA', 'CAPOEIRA', 'JUDO', 'NATACION', 'TAE KWON DO', 'FUTB', 'ESGRIMA', 'DEPORT', 'YOGA', 'FUTB']
        CONOCIMIENTO = ['AJEDREZ', 'LECTURA', 'INGLES', 'MATEMATICA', 'OFIMATICA', 'INGLES', 'ALGEBRA', 'ESTADISTICA', 'BRITANICO']

        data_feature = data_feature.assign(Hobbies = data_feature['Actividad extracurricular'].apply(lambda x: np.where(any(re.findall('|'.join(ARTE), x)), 'ARTE',
                                                                                                               np.where(any(re.findall('|'.join(DEPORTE), x)), 'DEPORTE',
                                                                                                               np.where(any(re.findall('|'.join(CONOCIMIENTO), x)), 'CONOCIMIENTO', 'OTRO')))))\
                                                                                              .drop(['Actividad extracurricular'], axis=1)


        Nacionalidad_v = np.vectorize(Functions.Nacionalidad)
        Validate_date_v = np.vectorize(Functions.Validate_date)
        
        data_feature['Nacionalidad'] = Nacionalidad_v(data_feature['Nacionalidad'])
        data_feature['Fecha de Nacimiento'] = Validate_date_v(data_feature['Fecha de Nacimiento'])
        data_feature['Fecha de ingreso del beneficiario a SPP'] = Validate_date_v(data_feature['Fecha de ingreso del beneficiario a SPP'])
        data_feature['Fecha de retiro del beneficiario'] = Validate_date_v(data_feature['Fecha de retiro del beneficiario'])
        data_feature['Fecha de ingreso al beneficiario al Elenco Central'] = Validate_date_v(data_feature['Fecha de ingreso al beneficiario al Elenco Central'])


        data_feature =  data_feature.assign(Edad             = np.where(pd.isna(data_feature['Fecha de Nacimiento']), np.nan, ((pd.to_datetime("today") - data_feature['Fecha de Nacimiento']).dt.days/365.25).round().astype('int', errors='ignore')),
                                            Dias_en_SPP      = np.where(data_feature['Fecha de retiro del beneficiario'].isnull, (pd.to_datetime("today") - data_feature['Fecha de ingreso del beneficiario a SPP']).dt.days, 
                                                                                                                                 (data_feature['Fecha de retiro del beneficiario'] - data_feature['Fecha de ingreso del beneficiario a SPP']).dt.days),
                                            #Transcion_elenco = np.where(data_feature['Fecha de ingreso al beneficiario al Elenco Central'].isna(), 0, 1),
                                            #Elenco_prev_dias = (data_feature['Fecha de ingreso al beneficiario al Elenco Central'] - data_feature['Fecha de ingreso del beneficiario a SPP']).dt.days,
                                            #Elenco_after_dias= (data_feature['Fecha de retiro del beneficiario'] - data_feature['Fecha de ingreso al beneficiario al Elenco Central']).dt.days,
                                            Transcion_domicilio_colegio     = np.where(data_feature['Distrito de domicilio'] != data_feature['Distrito del centro de estudios'], 1, 0),
                                            Distancia_domicilio_colegio     = data_feature[['Distrito de domicilio', 'Distrito del centro de estudios']].apply(lambda x: self.dist[x[0]][x[1]] if all([i in self.dist.columns for i in x]) else np.nan, axis = 1),
                                            Proximidad_domicilio_colegio    = data_feature[['Distrito de domicilio', 'Distrito del centro de estudios']].apply(lambda x: self.neig[x[0]][x[1]] if all([i in self.neig.columns for i in x]) else np.nan, axis = 1),
                                            Transicion_nucleo_nucleoinicial = np.where(data_feature['Nucleo'] !=  data_feature['Nucleo al que ingreso por primera vez'], 1, 0),
                                            Nivel_academico  =  data_feature['Nivel academico actual'].apply(lambda x:  np.where(x == 'INICIAL', 1, 
                                                                                                                        np.where(x == 'PRIMARIA', 2,
                                                                                                                        np.where(x == 'SECUNDARIA', 3, 
                                                                                                                        np.where(x == 'PREUNIVERSITARIO', 4,
                                                                                                                        np.where(x == 'UNIVERSITARIO', 5, 0)))))),
                                            Grado_estudios =  data_feature['Grado de estudio actual'].apply(lambda x: np.where(x == 'INICIAL', 1, 
                                                                                                                        np.where(x == '1ERO PRIMARIA', 2.1,
                                                                                                                        np.where(x == '2DO PRIMARIA', 2.2, 
                                                                                                                        np.where(x == '3ERO PRIMARIA', 2.3,
                                                                                                                        np.where(x == '4TO PRIMARIA', 2.4, 
                                                                                                                        np.where(x == '5TO PRIMARIA', 2.6,
                                                                                                                        np.where(x == '6TO PRIMARIA', 2.8, 
                                                                                                                        np.where(x == '1ERO SECUNDARIA', 3, 
                                                                                                                        np.where(x == '2DO SECUNDARIA', 3.2, 
                                                                                                                        np.where(x == '3ERO SECUNDARIA', 3.4, 
                                                                                                                        np.where(x == '4TO SECUNDARIA', 3.6, 
                                                                                                                        np.where(x == '5TO SECUNDARIA', 3.8,
                                                                                                                        np.where(x == 'PREUNIVERSITARIO', 4,
                                                                                                                        np.where(x == 'SUPERIOR UNIVERSITARIA', 5, 0)))))))))))))))).copy()

        data_feature['Año Ingreso'] = data_feature['Fecha de ingreso del beneficiario a SPP'].dt.year
        data_feature['Mes Ingreso'] = data_feature['Fecha de ingreso del beneficiario a SPP'].dt.month_name()
        data_feature['Mes Ingreso Num'] = data_feature['Fecha de ingreso del beneficiario a SPP'].dt.month
        data_feature['Año Retiro'] = data_feature['Fecha de retiro del beneficiario'].dt.year
        data_feature['Mes Retiro'] = data_feature['Fecha de retiro del beneficiario'].dt.month_name()
        data_feature['Mes Retiro Num'] = data_feature['Fecha de retiro del beneficiario'].dt.month


#        data_feature['Cluster Location'] = Functions.Cluster_Location(data_feature)
        
        Education_Score_v        = np.vectorize(Functions.Education_Score)
        Economic_Score_v         = np.vectorize(Functions.Economic_Score)
        Health_Score_v           = np.vectorize(Functions.Health_Score)
        Musical_Interest_Score_v = np.vectorize(Functions.Musical_Interest_Score)
        Total_Score_v            = np.vectorize(Functions.Total_Score)

        data_feature['Score Education'] = Education_Score_v(data_feature['Estudia actualmente'],
                                                            data_feature['Nivel_academico'],
                                                            data_feature['Grado_estudios'],
                                                            data_feature['Tipo de centro de estudios'],
                                                            data_feature['Hobbies'])


        data_feature['Score Economic']  = Economic_Score_v( data_feature['Tipo de centro de estudios'],
                                                            data_feature['Tiene beca de estudios?'],
                                                            data_feature['Beneficiario trabaja de manera remunerada?'],
                                                            data_feature['Tipo de SEGURO MEDICO (SIS, ESSALUD, EPS, otro, ninguno)'],
                                                            data_feature['Instrumento propio'])


        data_feature['Score Health']    = Health_Score_v(data_feature['Tipo de SEGURO MEDICO (SIS, ESSALUD, EPS, otro, ninguno)'],
                                                         data_feature['Alergias'],
                                                         data_feature['Restricciones alimentarias'],
                                                         data_feature['El beneficiario tiene alguna discapacidad? Especificar'],
                                                         data_feature['El beneficiario tiene algun trastorno de la personalidad? Especificar'],
                                                         data_feature['Ha tenido anemia?'])

        data_feature['Score Musical Interest'] = Musical_Interest_Score_v(  data_feature['Fecha de ingreso al beneficiario al Elenco Central'],
                                                                            data_feature['Instrumento propio'],
                                                                            data_feature['Cantidad de instrumentos prestados'])

        data_feature['Score Total'] = Total_Score_v(data_feature['Score Education'],
                                                    data_feature['Score Economic'],
                                                    data_feature['Score Health'],
                                                    data_feature['Score Musical Interest'])

        self.data_feature_ = data_feature.copy()

        return self.data_feature_


    def Encoding(self):

        data_encoding = self.data_feature_.copy()
        
        map_columns = {'Estado del beneficiarios' : {'ACTIVO' : 1,
                                                    'INACTIVO': 0},
                        
                        'Programa musical' : {'CORO'    : 1,
                                              'KINDER'  : 2,
                                              'ORQUESTA': 3},
                        
                        'Grupo' : {'FORMACION INFANTIL': 1,
                                   'FORMACION JUVENIL' : 2,
                                   'KINDER'            : 3},
                        
                        'Nacionalidad' : {'PERUANA': 1,
                                          'OTRO'   : 0},
                        
                        'Sexo' : {'M': 1,
                                  'F': 0}
                        }
        
        columns_categorical = ['Estado del beneficiarios','Programa musical','Grupo','Nacionalidad','Sexo']

        for column in columns_categorical:

            data_encoding[column] = data_encoding[column].apply(lambda x: map_columns[column][x])

        drop_list2 = ['Nucleo','Fecha de Nacimiento','Distrito de domicilio','Provincia de domicilio','Region de domicilio','Estudia actualmente','Nivel academico actual',
                      'Grado de estudio actual','Tipo de centro de estudios','Tiene beca de estudios?','Beneficiario trabaja de manera remunerada?',
                      'Tipo de SEGURO MEDICO (SIS, ESSALUD, EPS, otro, ninguno)','Alergias','Restricciones alimentarias','El beneficiario tiene alguna discapacidad? Especificar',
                      'El beneficiario tiene algun trastorno de la personalidad? Especificar','Ha tenido anemia?','Fecha de ingreso del beneficiario a SPP',
                      'Nucleo al que ingreso por primera vez','Fecha de ingreso al beneficiario al Elenco Central','Instrumento propio', 'Cantidad de instrumentos prestados',
                      'Fecha de retiro del beneficiario','Hobbies','Año Ingreso','Mes Ingreso','Mes Ingreso Num','Año Retiro','Mes Retiro','Mes Retiro Num',
                      'Transicion_nucleo_nucleoinicial', 'Distancia_domicilio_colegio', 'Proximidad_domicilio_colegio', 'Transcion_domicilio_colegio', 'Distrito del centro de estudios']
        
        self.data_final = data_encoding.drop(columns = drop_list2).copy()

        return self.data_final
        

