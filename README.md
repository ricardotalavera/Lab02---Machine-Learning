![](https://iat.es/wp-content/uploads/2020/12/ia-en-medicina.jpg)


# PI02 - MACHINE LEARNING - ESTADIA EN HOSPITAL

### 1.- INTRODUCCION

En esta ocasión enfrentamos un caso de clasificación con algoritmos supervisados. Se nos proporcionan dos archivos al estilo Kaggle, uno para entrenar al algoritmo y otro dataset sin la variable dependiente, para correr el modelo y confrontar la efectividad de nuestra predicción.

La variable Independiente se refiere a decidir si la estadia de un paciente en el hospital puede ser clasificada como larga (mayor a 8dias) o como corta (menor o igual a 8 dias).

###  2.- EDA

#### 2.1.- CARGA DEL ARCHIVO DE TRAINING
La carga se efectuo con la libreria PANDAS de PYTHON, el archivo a cargar fue un CSV, sin mayores problemas de encoding, la carga transcurre sin novedad. El archivo contó con 410,000 registros, estructurado como sigue :

-     Column                            | Non-Null Count   Dtype  
-     ------                            | --------------   -----  
-    Available Extra Rooms in Hospital  |410000 non-null  int64  
-    Department                         |410000 non-null  object 
-    Ward_Facility_Code                 |410000 non-null  object 
-    doctor_name                        |410000 non-null  object 
-    staff_available                    |410000 non-null  int64  
-    patientid                          |410000 non-null  int64  
-    Age                                |410000 non-null  object 
-    gender                             |410000 non-null  object 
-    Type of Admission                  |410000 non-null  object 
-    Severity of Illness                |410000 non-null  object 
-    health_conditions                  |410000 non-null  object 
-    Visitors with Patient              |410000 non-null  int64  
-    Insurance                          |410000 non-null  object 
-    Admission_Deposit                  |410000 non-null  float64
-    Stay (in days)                     |410000 non-null  int64  

#### 2.2.- REVISION DE LA ESTRUCTURA DEL DATASET
Como siguiente paso, se procedió a observar la estructura del dataset (vista superior), a fin de ver la cantidad y tipo de  variables que conforman los datos con los que deberemos trabajar. Se ubicó que teníamos 15 variables en total, de las cuales solo 6 eran de caracter numérico. Lo anterior ya nos plantea una estrategia partida de análisis de datos.

#### 2.3.- PRESENCIA DE NAs O NULLs
Como tercera etapa se procedió a revisar si se contaba con valores perdidos, lo cual se pudo apreciar rapidamente a través de total a traves de columnas del dataset, acompañado de n mapa de calor para una rápida visualización. El dataset no presento valores perdidos o nulos.

![](https://491090.fs1.hubspotusercontent-na1.net/hub/491090/hubfs/Alondra/machine%20learning-1.jpg?width=950&name=machine%20learning-1.jpg)

#### 2.4.- TRANSFORMACION DE DATOS - PARTE I
Seguidamente se procedió a efectuar unos cambios en los nombres de las columnas, ya que muchos campos tenian nombres de tamaño largo y acompañados espacios intermedios. Siempre cualquier titulo de variable con puntos o espacios o guiones medios, debe en la medida de lo posible evitarse para aliviar la programción por venir. Lo anterior claramente aplica a caracteres extraños.

Asimismo de procedió a crear una columna de caracter dicotómica y numerica, que nos permitiera luego tomarla como nuestra variable a predecir en la clasificación. Claro esta que la información en la columna "Stay (in days)" no permitió efectuar esta transformación sin inconvenientes. Posteriormente se elimino la columna original "Stay (in days)".

Finalmente se reviso la presencia de duplicados a nivel de fila completa, en cuyo caso se procedia a su eliminación.

#### 2.5.- BALANCEO DE LAS CLASES DE LA VARIABLE DEPENDIENTE
En esta etapa del EDA es imprescindible revisar que la variable a clasificar tenga justamente las clases no desbalancedas. Se puede tomar como regla general que dos clases se encuentran desbalanceadas cuando la clase menor es menor o igual al 5% de la clase mayoritaria. En caso este fenómeno se dé, el científico de datos deberá corregir esta situación, para lo cual tendrá dos caminos :
-	Elimina registros de la clase mayoritaria hasta que se consideren las clases balanceadas.
-	Aumenta registros sintéticos de la clase minoritaria hasta que el desbalanceo no se encuentre presente.
En el caso de nuestro dataset las clases se pudieron considerar como balanceadas.

#### 2.6.- ANALISIS DE VARIABLES CUANTITATIVAS
Se debe tener en cuenta que nuestra variable a clasificar, si bien es cierto se muestra en PYTHON de tipo cuantitativo, en realidad es cualitativa o factor con dos niveles de existencia. Por tanto un analisis de matriz de correlaciones de Pearson no aplica para con nuestra variable depediente. La matriz de correlaciones se efectúo para conocer si los potenciales predictores numéricos correlacionaban entre si. Adicionalmente se efectúo un gráfico de funciónes de densidad para cada variable numerica y a la vez este gráfico era por cada clase de la variable dependiente. Con el fin de observar y poder concluir que variable numerica podía ayudarnos en distinguir diferencias entre las clases de la variable "y" o variable dependiente. Normalmente si las funciones de densidad de una variable cuantitativa, se encuentran traslapadas para dos clases de una variable cualitativa, es un indicador que dicho predictor numérico podra brindar información al modelo.
En caso se encuentren predictores que correlacionen, el cientifico de datos deberá : o quedarse con una sola variable (ya que reportaran al modelo una duplicación de información) o fusionar ambas variables correalcionadas a traves de algún algoritmo matemático (suma, multiplicación etc).
En nuestro caso las variables cuantitativas tenían escada correlación entre ellas, pero también poca capacidad de ayudar al modelo en la clasificación.

#### 2.7.- ANALISIS DE VARIABLES CUALITATIVAS
El objetivo en esta parte es poder dilucidar que variables de caracter cualitativo presentan asociación con nuestra variable dependiente. Claramente en este caso se debe acudir a una prueba de independencia cuyo estadistico sigue una distribución chi-cuadrado. Dicha prueba se efectúo para cada variable cualitativa respecto a la variable dependiente. Solo se obvio en aquellos casos que la tabla de contingencia presentó muchos ceros, fenómeno que sesga la prueba. En casos donde la prueba no era factible de aplicar, se concluyó gráficamente.
Los resultados fueron que todas las variables tenian asociación con la variable dependiente excepto la variable "insurance".

#### 2.8.- TRANSFORMACION DE DATOS - PARTE II
Visto todo lo anterior, solo se procedió transformar las variables cualitativas a codificación numerica con Label Encoder de PHYTON y todas las variables se estandarizaron para un uso comun a cualquier modelo.

Se culmina esta etapa dividiendo los datos en training y en test. Se escogió una proporcion de 30% para test y 70% para training.


![](https://www.hd-tecnologia.com/imagenes/articulos/2015/01/Los-expertos-quieren-proteger-a-los-humanos-de-la-Inteligencia-Artificial.jpg)


###  3.- MODELADO

Siendo un problema de clasificación se usaron los siguientes modelos :

- Regresión Logistica.
- Regresión Logistica optimizada.
- Arbol de Clasificación.
- Random Forest.
- KNN (K-Nearest Neighbours)
- Redes Neuronales
- Un ensamble de { regresión Logistica + Arbol de Clasificación + KNN}

En los modelos de arboles, KNN y Random Forest se usaron grillas de optimización de hiperparámetros.

El objetivo era obtener el maximo RECALL o SENSIBILIDAD, definida como :

$$**RECALL = TP/ (TP * FN)**$$

donde :

- **TP = TRUE POSITIVE o cantidad de verdaderos positivos**
- **FN = FALSE NEGATIVE o cantidad de falsos negativos**

Es bueno recalcar que modelos con muy alta sensibilidad pierden especificidad (aciertos en verdaderos negativos), la idea es buscar un balance o también dependerá del objetivo perseguido.

Finalmente se sumo modularidad con pipelines. Fueron tres pipelines (uno por cada modelo escogido).

###  4.- CONCLUSIONES FINALES.

Como palabras finales se desea resaltar la importancia de tener una buen análisis y sobre todo comprensión de las variables que se estan modelando, los modelos reaccionarán según el input que les brindemos. Si se desea un cambio significativo en los ratios de predicción, deberá venir de brindarle al modelo mejor o máyor información y no necesariamente de aplicar modelos diferentes.
