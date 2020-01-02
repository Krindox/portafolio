'''
Descripcion
Esta es la implementacion de una red neuronal en la cual el programa pedira al usuario entradas como por ejemplo el numero de capas
que desea, la cantidad de neuronas por cada capa y otros datos para que al final se genere una grafica mostrando como la red neuronal
aprende mediante el algoritmo de BackPropagation y disminuye su porcentaje de error
Autor
Jose Cubillos 
Nota: La implementacion de esta red neuronal se hizo basado en la guia del video de dotCSV, que se encuentra en el siguiente link:
https://www.youtube.com/watch?v=W8AeOXa_FqU


'''

#librerias necesarias
import numpy as np
from matplotlib import pyplot as plt

#variables necesarias
errorTotal = 0
entradas = 0
topologia = []

#Datos digitados por el usuario
#Cantidad de entradas de la Red Neuronal
print("Ingrese la cantidad de entradas que desea para la Red Neuronal : ")
entradas = input()
entradas = int(entradas)
#Cantidad de salidas de la Red Neuronal
print("Ingrese la cantidad de salidas que desea para la Red Neuronal : ")
salidas = input()
salidas = int(salidas)
#Se da a escoger entre si los datasets seran aleatorios o digitados por el usuario
print("Desea digitar los valores de las entradas y las salidas. Digite S, de lo contrario seran generados aleatoriamente")
datosDigitados = input()
if datosDigitados != 'S':
    # Se crean los datasets aleatoriamente
    datasetin = np.random.rand(1, entradas)
    datasetout = np.random.rand(1, salidas)
else:
    # Se crean los datasets aleatoriamente para despues ser reemplazados por los valores que digite el usuario
    datasetin = np.random.rand(1, entradas)
    datasetout = np.random.rand(1, salidas)
    for i in range(entradas):
        print("Digite valor de entrada " + str(i+1))
        datoE = input()
        datoE = float(datoE)
        datasetin[0][i] = datoE
    for j in range(salidas):
        print("Digite valor de salida " + str(j+1))
        datoS = input()
        datoS = float(datoS)
        datasetout[0][j] = datoS

#Cantidad de capas ocultas que tendra la red neuronal
print("Ingrese la cantidad de capas ocultas que desea : ")
cantCapas = input()
cantCapas = int(cantCapas)

#Se crea un vector que es donde estara guardada la topologia de la red es decir  cantidad de neuronas por cada capa y se inicia llenando con la cantidad de entradas
topologia.append(entradas)
for i in range(cantCapas):
    #Por cada capa se pregunta el numero de neuronas que se quiere y se agrega al vector topologia
    print("Ingrese la cantidad de neuronas en la capa " + str(i+1) + ":")
    cantNeuronas = input()
    cantNeuronas = int(cantNeuronas)
    topologia.append(cantNeuronas)
#Se pregunta el numero de error que desea el usuario el cual sera a su vez el que determine hasta que momento se iterara en el entrenamiento de la red
print("Ingrese el numero de error admisible : ")
numError = input()
numError = float(numError)
#Se pregunta la tasa de aprendizaje usada en el gradiente decendiente 
print("Ingrese la tasa de aprendizaje (se recomienda 0.5) : ")
tasaApren = input()
tasaApren = float(tasaApren)
#Finalmente al vector de topologia se le agrega al final la cantidad de salidas de la red que se puede considerar como las neuronas de la ultima capa
topologia.append(salidas)
#Definimos la clase que sera usada para definir la estructura de las capas de la red neuronal
class capa_neuronal():
    def __init__(self, num_conex, num_neuronas, func_activ):
        self.func_activ = func_activ
        # De acuerdo a la investigacion realizada el parametro de b, es un valor aleatorio al igual que el W
        self.b = np.random.rand(1, num_neuronas)
        self.w = np.random.rand(num_conex, num_neuronas)




# en una variable se guardaran dos funciones anonimas con el comando lambda una es la funcion sigmoidal y la otra es su derivada
f_Sigmoidal = (lambda x: 1 / (1 + np.e ** (-x)), lambda x: x * (1 - x))
# Se deja la funcion Relu en caso de ser necesario
f_relu = lambda x: np.maximum(0, x)


# Definicion de la funcion que crea la red neuronal
def crear_red_neuronal(topologia, func_activ):
    #Como tal toda las capas de la red neuronal seran un vector por eso se añade la siguiente linea
    r_n = []
    
    for c, capa in enumerate(topologia[:-1]):
        #mediante el bucle se van agregando objetos de tipo capa_neuronal y asi se arma la red
        r_n.append(capa_neuronal(topologia[c], topologia[c+1], func_activ))
        
    return r_n

#En la siguiente variable es donde estara guardada la estructura de la red neuronal con todos sus valores
red_neuronal = crear_red_neuronal(topologia, f_Sigmoidal)

#Definimos la funcion de coste que se usa para obtener el error
f_Coste = lambda salidas_r, salidas_p: np.mean((salidas_r - salidas_p) ** 2)
#Definimos la derivada de la funcion de coste
f_coste_deri = lambda salidas_r, salidas_p: (salidas_p - salidas_r)

#Creamos la funcion encargada de entrenar la red neuronal pasando como parametro la red neuronal, 
#los datos de entrada, los datos de salida, la funcion de coste que tambien contiene su derivada y por ultimo la tasa de aprendizaje
def entrenamiento(red_neuronal, datasetin, datasetout, f_Coste, ta):
    #En la variable out se guardara tanto las sumas ponderadas (neth) como las funciones de activacion (outh) de cada capa anterior, en el primer caso no hay suma ponderada debido a que es la capa de entrada
    out = [(None, datasetin)]
    for c, capa in enumerate (red_neuronal):
        #Propagacion hacia adelante
        # El simbolo arroba significa multiplicacion de matrices, se esta multiplicando las salidas de la capa anterior por cada uno de los pesos y se le suma el parametro b, esto se hace para hallar la suma ponderada de cada neurona
        z = out[-1][1] @ red_neuronal[c].w + red_neuronal[c].b
        #Luego dicha suma ponderada o neth es la que se pasa como parametro a la funcion de activacion para hallar la salida de cada neurona y se guara en la variable a
        a = red_neuronal[c].func_activ[0](z)
        #Como se necesita en las operaciones trabajar con los resultados de la capa previa es por eso que el resultado de la capa actual se guarda en la variable auxiliar out
        out.append((z, a))
        
    #A continuacion se calculo el error con la funcion de coste, pasando como parametro la salida final de las anteriores iteraciones es decir el ultimo outh y se pasa el valor real es decir los datos de las salidas reales ingresados por el usuario
    print ("Error Total: ")    
    print(f_Coste(out[-1][1], datasetout))
    
    #BackPropagation
    #
    deltas = []
    for c in reversed(range (0, len(red_neuronal))):
        z = out[c+1][0]
        a = out[c+1][1]
        #print("impresion forma de a (funcion Activacion)")
        #print(a.shape)
        if c == len(red_neuronal) - 1:
            #Ultima Capa
            #print("impresion derivadas funcion de coste")
            #print(f_coste_deri(datasetout, a))
            #print("impresion derivadas funcion de de activacion")
            #print(red_neuronal[c].func_activ[1](a))
            #print("impresion forma de W")
            #print(red_neuronal[c].w.shape)
            deltas.insert(0, f_coste_deri(datasetout, a) * red_neuronal[c].func_activ[1](a))
            #deltas.insert
            #print("impresion de deltas")
            #print(deltas)
        else:
             
             #print("impresion forma deltas")
             #print(deltas[0].shape)
             #print("impresion forma W")
             #print(w_ant.shape)
             deltas.insert(0, deltas[0] @ w_ant.T * red_neuronal[c].func_activ[1](a))
        w_ant = red_neuronal[c].w
        #decenso del gradiente
        red_neuronal[c].w = red_neuronal[c].w - out[c][1].T @ deltas[0] * ta
        
    print(out[-1][1])
    return out[-1][1]
generaciones = []
errores = []
errorAct = 10000
gen = 0
while errorAct > numError:  
    gen += 1
    train = entrenamiento (red_neuronal, datasetin, datasetout, f_Coste, tasaApren)
    errorAct = f_Coste(train, datasetout)
    #if i % 25 == 0:
        
    errores.append(f_Coste(train, datasetout))
    generaciones.append(gen)

print("")      
print("|| Grafica de la evolucion del error a lo largo del entrenamiento ||")
plt.plot(generaciones,errores,'r-',label= 'Mejores')
plt.xlabel('Iteraciones de Entrenamiento')
plt.ylabel('Error')
plt.title('EVOLUCION DEL ERROR')
plt.show()
print("")
print("")
print("====|| Valores optimos de los parametros W en cada capa ||====")
print("")
for j in range (0, len(red_neuronal)):
    print("pesos en capa:" + str(j+1))
    print(red_neuronal[j].w[:])
#print(errores[:])



