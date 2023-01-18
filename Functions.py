import unicodedata
import datetime
#from arcgis.geocoding import geocode
#from arcgis.gis import GIS
#gis = GIS()
#from sklearn.cluster import KMeans

score_education_total = 1 + 4 + 4 + 2 + 3
score_economic_total = 2 + 1 + 1 + 2 + 1
score_health_total = 2 + 5
score_interest_total = 3

def Normalize_data(x):
    valor_string = unicodedata.normalize('NFKD',str(x))
    valor_string = valor_string.encode('ASCII','ignore')
    valor_string = valor_string.decode('utf-8').upper().strip()

    if valor_string == 'NAN':
        valor_string = 'NO APLICA'
    
    return valor_string.upper().strip()

#def Latitude_Longitud(x,y,z):
#    multi_field_address = { 
#                    'CntryName': 'Perú',
#                    'Region': x,
#                    'Subregion': y,
#                    'City': z,
#                    }
#    esrihq = geocode(multi_field_address)
#    longitude = esrihq[0]['location']['x']
#    latitude = esrihq[0]['location']['y']    
#
#    return (longitude, latitude)

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

#def Cluster_Location(df):
#
#    kmeans_model = KMeans(n_clusters = 4,random_state=3)
#    kmeans_model.fit(df[['Latitud','Longitud']])
#
#    return kmeans_model.labels_

