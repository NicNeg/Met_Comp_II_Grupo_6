import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1a

# Definimos g
def g(x, n, alpha):
    return sum(np.exp(-(x - k)**2) * k**(-alpha) for k in range(1, n + 1))

# Parámetros
n = 10
alpha = 4/5
num_muestras = 500000
bins = 200

# Algoritmo de Metropolis-Hastings
def metropolis_hastings(g, num_muestras, x0=5, proposal_std=1.0):
    muestras = []
    x = x0  # Inicialización

    for _ in range(num_muestras):
        # Propuesta de nuevo valor
        x_new = x + np.random.normal(0, proposal_std)
        
        # Aceptar o rechazar
        acceptance_ratio = g(x_new, n, alpha) / g(x, n, alpha)
        if np.random.rand() < min(1, acceptance_ratio):
            x = x_new  # Aceptamos la muestra
        
        muestras.append(x)
    
    return np.array(muestras)

# Generamos las muestras
muestras = metropolis_hastings(g, num_muestras)

# Gráfica del histograma
plt.figure(figsize=(8, 6))
plt.hist(muestras, bins=bins, density=True, alpha=0.6, color='b', edgecolor='black')
plt.xlabel("x")
plt.ylabel("Densidad")
plt.title("Histograma de muestras generadas con Metropolis-Hastings")
plt.grid(True)
plt.savefig("1.a.pdf")


# 1b

# Definimos f
def f(x):
    return np.exp(-x**2)

# Calculamos la estimación de A usando muestreo de importancia
weights = f(muestras) / np.array([g(xi, n, alpha) for xi in muestras])
A_est = (np.sqrt(np.pi) * num_muestras) / np.sum(weights)

# Calculamos la desviación estándar
std_dev = np.sqrt(np.pi) * np.std(weights) / np.sqrt(num_muestras)

# Imprimimos el resultado con la incertidumbre
print(f"1.b) A estimado: {A_est} ± {std_dev}")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from numba import njit

#2 Integral de camino para difracción de Fresnel 

#Se definen las variables del sistema de la doble rendija, también los arrays de angulo & de posición z. 

D = 0.5
λ = 6.7e-7
A = 0.0004
a = 0.0001
d = 0.001

z = np.arange(-0.0039,0.0039,0.000001)
theta = np.arctan(z/D)

#Se define la intensidad para el caso clásico. Se tienen intensidades I_1 e I_2 en lugar de una sola I
#para facilitar el entendimiento de la ecuación larga, lo mismo ocurre para el caso moderno.

@njit
def I_clasico(z,theta,λ,d,a):
    
    I_1 = (np.cos((np.pi*d*np.sin(theta))/λ))**2
    I_2 = (np.sinc((a*np.sin(theta))/λ))**2
    
    I = I_1*I_2
    
    return (I/(I.max()))

I_c = I_clasico(z, theta, λ, d, a)

#Se define la intensidad para el caso moderno. 
#Se generan arrays random de x,y. Ambos acotados bajo las dimensiones fisicas del sistema de la doble rendija.
#'y' es la multiplicación de "y_signo" & "y_magnitud" debido a que se calcula primero los 'y' para una rendija
#& luego se multiplican por signos + ó - que la distribuyen entre las dos rendijas.

@njit
def I_moderno(z,D,λ,d,a):
    
    x = np.random.uniform(-A/2,A/2,100000)
    y_magnitud = np.random.uniform(d/2 - a/2,d/2 + a/2,100000)
    y_signo = np.sign(np.random.rand(100000)-1/2)
    y = y_signo*y_magnitud
    
    I_moderno = np.zeros(len(z))
    
    for k in range (0,len(z)):
        
        I = 0
        
        for i in range(0,len(x)):
            
            I_1 = np.exp((1j*np.pi*((x[i]-y[i])**2))/(λ*D))
            I_2 = np.exp((1j*np.pi*((z[k]-y[i])**2))/(λ*D))
            I += I_1*I_2
            
        I = (np.abs(((np.exp((2j*np.pi*2*D)/λ))/len(x))*I))**2
        I_moderno[k] = I
        
    return (I_moderno/(I_moderno.max()))

#Se grafican la intensida clásica y moderna vs la posición z.

I_m = I_moderno(z,D,λ,d,a)
plt.plot(z,I_c, label='Clásico')
plt.plot(z,I_m, label='Moderno')
plt.ylabel("Intensidad")
plt.xlabel("z")
plt.legend()
plt.savefig('2.pdf',bbox_inches='tight')

#3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros del sistema
N = 150  # Tamaño de la matriz
J = 0.2  # Interacción
beta = 10  # Inverso de la temperatura
num_frames = 500  # Cantidad de frames en la animación
iter_per_frame = 400  # Iteraciones entre frames

# Inicializar la malla de espines aleatoriamente
spins = np.random.choice([-1, 1], size=(N, N))

def energy_difference(spins, i, j):
    """Calcula la diferencia de energía al cambiar un espín en (i, j)."""
    neighbors = spins[(i+1) % N, j] + spins[(i-1) % N, j] + spins[i, (j+1) % N] + spins[i, (j-1) % N]
    dE = 2 * J * spins[i, j] * neighbors
    return dE

def metropolis_step(spins, beta):
    """Realiza una iteración del algoritmo de Metropolis-Hastings."""
    for _ in range(iter_per_frame):
        i, j = np.random.randint(0, N, size=2)  # Elegir un espín al azar
        dE = energy_difference(spins, i, j)
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] *= -1  # Aceptar el cambio

# Configurar la animación
fig, ax = plt.subplots()
im = ax.imshow(spins, cmap='gray', animated=True)

def update(frame):
    """Función de actualización para la animación."""
    metropolis_step(spins, beta)
    im.set_array(spins)
    return [im]
# Guardar la animación en un archivo GIF
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
ani.save("3.gif", writer="pillow", fps=30)
plt.show()

#4

import random
import matplotlib.pyplot as plt
from collections import defaultdict
import os


def limpiar_texto(texto):
    # Convertimos todo a minúsculas
    texto = texto.lower()

    # Eliminamos encabezados y pies del Proyecto Gutenberg
    lineas = texto.split('\n')
    texto_filtrado = []
    grabar = False

    for linea in lineas:
        if '*** start of the project gutenberg ebook' in linea:
            grabar = True
            continue
        if '*** end of the project gutenberg ebook' in linea:
            grabar = False
            break
        if grabar:
            texto_filtrado.append(linea)

    # Convertimos la lista de líneas en un solo string
    texto = '\n'.join(texto_filtrado)

    # Eliminamos caracteres especiales dejando solo letras, números, espacios y saltos de línea
    texto_limpio = ''
    for caracter in texto:
        if caracter.isalnum() or caracter in [' ', '\n']:
            texto_limpio += caracter

    # Eliminamos saltos de línea múltiples
    lineas = texto_limpio.split('\n')
    lineas = [linea.strip() for linea in lineas if linea.strip() != '']
    texto_limpio = '\n'.join(lineas)

    return texto_limpio


# Cargar el archivo de texto
with open('pg932.txt', 'r', encoding='utf-8') as archivo:
    texto_original = archivo.read()

# Limpiar el texto
texto_limpio = limpiar_texto(texto_original)

# Guardar el resultado en un nuevo archivo
with open('texto_limpio.txt', 'w', encoding='utf-8') as archivo:
    archivo.write(texto_limpio)


# Cargar diccionario EOWL desde una carpeta con archivos "A Words.txt", "B Words.txt", etc.
def cargar_diccionario(carpeta):
    palabras_validas = set()
    for nombre_archivo in os.listdir(carpeta):
        if nombre_archivo.endswith(".txt"):
            with open(os.path.join(carpeta, nombre_archivo), 'r', encoding='utf-8') as archivo:
                for linea in archivo:
                    palabra = linea.strip().lower()
                    if palabra:
                        palabras_validas.add(palabra)
    return palabras_validas


# Cargar el diccionario desde la carpeta
palabras_reales_basicas = cargar_diccionario('dict')

def entrenar_markov(texto, n):
    modelo = defaultdict(lambda: defaultdict(int))  # Diccionario de frecuencias

    for i in range(len(texto) - n):
        clave = texto[i:i + n]  # Extraemos un n-grama
        siguiente = texto[i + n]  # Caracter que sigue al n-grama
        modelo[clave][siguiente] += 1  # Registramos la frecuencia

    return modelo


def generar_texto(modelo, longitud, n):
    random.seed(None)
    clave = random.choice(list(modelo.keys()))
    resultado = list(clave)

    for _ in range(longitud - n):
        if clave in modelo:
            siguiente_caracteres = modelo[clave]
            caracteres = list(siguiente_caracteres.keys())
            frecuencias = list(siguiente_caracteres.values())
            siguiente = random.choices(caracteres, weights=frecuencias)[0]
            resultado.append(siguiente)
            clave = ''.join(resultado[-n:])  # Actualizar clave
        else:
            break

    return ''.join(resultado)


def analizar_texto(texto_generado):
    palabras_generadas = set(texto_generado.split())
    palabras_validas = palabras_generadas.intersection(palabras_reales_basicas)
    porcentaje_validas = (len(palabras_validas) / len(palabras_generadas)) * 100 if len(palabras_generadas) > 0 else 0

    return porcentaje_validas


def graficar_resultados(resultados):
    ns = list(resultados.keys())
    porcentajes = list(resultados.values())

    plt.figure(figsize=(10, 6))
    plt.plot(ns, porcentajes, marker='o')
    plt.title('Porcentaje de palabras válidas generadas vs. n-grama')
    plt.xlabel('n (tamaño del n-grama)')
    plt.ylabel('Porcentaje de palabras válidas')
    plt.grid(True)
    


# Entrenar modelos para varios valores de n
resultados = {}
for n in range(1, 20):
    modelo = entrenar_markov(texto_limpio, n)
    texto_generado = generar_texto(modelo, 1500, n=n)
    
    # Guardar el texto generado en un archivo separado
    with open(f'gen_text_n{n}.txt', 'w', encoding='utf-8') as archivo:
        archivo.write(texto_generado)

    porcentaje_validas = analizar_texto(texto_generado)
    resultados[n] = porcentaje_validas

graficar_resultados(resultados)

plt.savefig('certezas.png')
