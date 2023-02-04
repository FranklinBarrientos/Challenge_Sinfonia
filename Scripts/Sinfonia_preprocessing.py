import re
import pandas as pd
import numpy as np
import unicodedata
import pickle
from Scripts.Functions import *

class Sinfonia():

    def __init__(self,df, distancia_data, neighboor_data):
        self.data = df
        self.dist = distancia_data
        self.neig = neighboor_data

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
        self.data['Numero de Documento de identidad'] = self.data['Numero de Documento de identidad'].astype(int).astype(str)

        drop_list = ['Tipo de sangre','Edad','Celular del beneficiario', 'Nombre de Contacto de emergencia / Cuidador(a)', 
                     'Recibio vacuna contra el COVID-19', 'Direccion de domicilio','Codigo del instrumento\n (de acuerdo al inventario)', 'Periodo/MES','Centro de estudios','Modulo', 
                     'Numero de poliza', 'Otra discapacidad (Solo llenar cuando el campo anterior es "Otro")', 'Otro trastorno (Solo llenar cuando el campo anterior es "Otro")',
                     'Toma algun medicamento? Especificar', 'Cuantas dosis tiene?', 'Tiene alguna enfermedad previa?', 'Otra enfermedad previa', 
                     'Numero de Contacto de contacto de emergencia / Cuidador (a)', 'Parentesco con el beneficiario', 'Tipo de Documento de Identidad', 'Especialidad / Registro de voz',
                     'Motivo del retiro', 'Cantidad de reingresos',
                     'Fecha de inactividad', 'Fecha de ingreso']

        data_drop = self.data.drop(columns = drop_list)
        

        Location_v = np.vectorize(Location)
        Normalize_data_v = np.vectorize(Normalize_data)

        ## Imputation

        for columna in data_drop.columns:
            if str(data_drop[columna].dtype) == 'object':
                data_drop[columna] = Normalize_data_v(data_drop[columna])
            elif str(data_drop[columna].dtype) == 'int64':
                data_drop[columna] = data_drop[columna].fillna(0)
            elif str(data_drop[columna].dtype) == 'float64':
                data_drop[columna] = data_drop[columna].fillna(0)


        mask_location = Location_v(data_drop['Region de domicilio'],data_drop['Provincia de domicilio'])

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


        Nacionalidad_v = np.vectorize(Nacionalidad)
        Validate_date_v = np.vectorize(Validate_date)
        
        data_feature['Nacionalidad'] = Nacionalidad_v(data_feature['Nacionalidad'])
        data_feature['Fecha de Nacimiento'] = Validate_date_v(data_feature['Fecha de Nacimiento'])
        data_feature['Fecha de ingreso del beneficiario a SPP'] = Validate_date_v(data_feature['Fecha de ingreso del beneficiario a SPP'])
        data_feature['Fecha de retiro del beneficiario'] = Validate_date_v(data_feature['Fecha de retiro del beneficiario'])
        data_feature['Fecha de ingreso al beneficiario al Elenco Central'] = Validate_date_v(data_feature['Fecha de ingreso al beneficiario al Elenco Central'])


        data_feature =  data_feature.assign(Edad             = np.where(pd.isna(data_feature['Fecha de Nacimiento']), np.nan, ((pd.to_datetime("today") - data_feature['Fecha de Nacimiento']).dt.days/365.25).round().astype('int', errors='ignore')),
                                            Dias_en_SPP      = np.where(data_feature['Fecha de retiro del beneficiario'].isnull, (pd.to_datetime("today") - data_feature['Fecha de ingreso del beneficiario a SPP']).dt.days, 
                                                                                                                                 (data_feature['Fecha de retiro del beneficiario'] - data_feature['Fecha de ingreso del beneficiario a SPP']).dt.days),
                                            Transcion_domicilio_colegio     = np.where(data_feature['Distrito de domicilio'] != data_feature['Distrito del centro de estudios'], 1, 0),
                                            Distancia_domicilio_colegio     = data_feature[['Distrito de domicilio', 'Distrito del centro de estudios']].apply(lambda x: self.dist[x[0]][x[1]] if all([i in self.dist.columns for i in x]) else np.nan, axis = 1),
                                            Proximidad_domicilio_colegio    = data_feature[['Distrito de domicilio', 'Distrito del centro de estudios']].apply(lambda x: self.neig[x[0]][x[1]] if all([i in self.neig.columns for i in x]) else np.nan, axis = 1),
                                            Transicion_nucleo_nucleoinicial = np.where(data_feature['Nucleo'] !=  data_feature['Nucleo al que ingreso por primera vez'], 1, 0),
                                            Nivel_academico  =  data_feature['Nivel academico actual'].apply(lambda x:  np.where(x == 'INICIAL', 1, 
                                                                                                                        np.where(x == 'PRIMARIA', 2,
                                                                                                                        np.where(x == 'SECUNDARIA', 3, 
                                                                                                                        np.where(x == 'PREUNIVERSITARIO', 4,
                                                                                                                        np.where(x == 'UNIVERSITARIO', 5, 0)))))),
                                            Grado_estudios =  data_feature['Grado de estudio actual'].apply(lambda x:   np.where(x == 'INICIAL', 1, 
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

        Education_Score_v        = np.vectorize(Education_Score)
        Economic_Score_v         = np.vectorize(Economic_Score)
        Health_Score_v           = np.vectorize(Health_Score)
        Musical_Interest_Score_v = np.vectorize(Musical_Interest_Score)
        Total_Score_v            = np.vectorize(Total_Score)

        data_feature['Score Education'] = Education_Score_v(data_feature['Estudia actualmente'],
                                                            data_feature['Nivel_academico'],
                                                            data_feature['Grado_estudios'],
                                                            data_feature['Tipo de centro de estudios'],
                                                            data_feature['Hobbies'])


        data_feature['Score Economic'] = Economic_Score_v( data_feature['Tipo de centro de estudios'],
                                                            data_feature['Tiene beca de estudios?'],
                                                            data_feature['Beneficiario trabaja de manera remunerada?'],
                                                            data_feature['Tipo de SEGURO MEDICO (SIS, ESSALUD, EPS, otro, ninguno)'],
                                                            data_feature['Instrumento propio'])


        data_feature['Score Health'] = Health_Score_v(data_feature['Tipo de SEGURO MEDICO (SIS, ESSALUD, EPS, otro, ninguno)'],
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
                      'Hobbies','Año Ingreso','Mes Ingreso','Mes Ingreso Num','Año Retiro','Mes Retiro','Mes Retiro Num','Transicion_nucleo_nucleoinicial', 
                      'Distancia_domicilio_colegio', 'Proximidad_domicilio_colegio', 'Transcion_domicilio_colegio', 'Distrito del centro de estudios']
        
        self.data_final = data_encoding.drop(columns = drop_list2).copy()

        return self.data_final
        
    def PerfilEstudiante(self, filename):

        km = pickle.load(open(filename, "rb"))

        ## CORREGIR LA LINEA 210 PORQUE ELIMINAS MUCHOS VALORES DEBIDO A QUE LA COLUMNA Distancia_domicilio_colegio presenta valores nan

        data_encoding = self.data_feature_.reset_index(drop = True).copy()

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

        y_km = km.predict(data_cl)

        data_encoding['cluster'] = y_km

        data_encoding = data_encoding.assign(cluster =  np.where(data_encoding['cluster'] == 0, 'Perfil 1',
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

        dbase_norm.columns = ['promedio Masculino', 'promedio Cantidad de hermanos', 'promedio Transcion_domicilio_colegio', 
                              'promedio Arte', 'promedio Conocimiento', 'promedio Deporte', ' promedio Colegio publico']

        data_encoding = pd.concat([data_encoding,dbase_norm],axis=1)

        data_encoding['Estado_del_beneficiario'] = data_encoding['Estado del beneficiarios.ACTIVO'].values

        self.data_perfil_estudiante = data_encoding 

        return self.data_perfil_estudiante
