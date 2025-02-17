import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.signal import peak_widths
from scipy.interpolate import krogh_interpolate
from scipy.fftpack import fftfreq
import pandas as pd

# Punto 1.a.

def datos_prueba(t_max:float, dt:float, amplitudes:NDArray[float],
 frecuencias:NDArray[float], ruido:float=0.0) -> NDArray[float]:
    ts = np.arange(0.,t_max,dt)
    ys = np.zeros_like(ts,dtype=float)
    for A,f in zip(amplitudes,frecuencias):
        ys += A*np.sin(2*np.pi*f*ts)
    ys += np.random.normal(loc=0,size=len(ys),scale=ruido) if ruido else 0
    return ts,ys

def Fourier(t:NDArray[float], y:NDArray[float], f:NDArray[float]) -> NDArray[complex]:
    """Calcula la Transformada de Fourier para un conjunto de frecuencias."""
    return np.array([np.sum(y * np.exp(-2j * np.pi * t * fi)) for fi in f])


# Datos de prueba
t_max = 10
dt = 0.01
amplitudes = np.array([1, 2.3, 1.5])
frecuencias = np.array([5, 10, 15])
ruido = 0.8

# Generamos una señal limpia (sin ruido) y una con ruido
t, y_limpia = datos_prueba(t_max, dt, amplitudes, frecuencias, ruido=0.0)
_, y_ruido = datos_prueba(t_max, dt, amplitudes, frecuencias, ruido=ruido)

# Rango de frecuencias a evaluar
f_range = np.linspace(0, 20, 500)

# Transformada de Fourier de cada señal
F_limpia = Fourier(t, y_limpia, f_range)
F_ruido = Fourier(t, y_ruido, f_range)


# Gráfica 1.a.
plt.figure(figsize=(10, 5))
plt.plot(f_range, np.abs(F_limpia), label='Sin ruido', linestyle='dashed')
plt.plot(f_range, np.abs(F_ruido), label='Con ruido', alpha=0.7)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud de la Transformada")
plt.title("Transformada de Fourier de señales con y sin ruido")
plt.legend()
plt.grid()
plt.savefig("1.a.pdf")

print("1.a) Pueden aparecer picos en la transformada que no corresponden a frecuencias reales de la señal")


# Punto 1.b.

# Parámetros
t_max_values = np.linspace(10, 300, 30)
amplitud = np.array([1.0])
frecuencia = np.array([5.0])
fwhm_values = []

'''
Variando el intervalo de tiempo desde 10s a 30s, generamos una señal de prueba sin ruido, con una frecuencia conocida
y vamos calculando su trasfomada, el pico de la frecuencia y su ancho a media altura (FWHM).
Guardamos cada valor del FWHM calculado a partir de cada valor de t_max. 
'''
for t_max in t_max_values:
    t, y = datos_prueba(t_max, dt, amplitud, frecuencia, ruido=0.0)
    TF = np.abs(Fourier(t, y, f_range))  # Transformada de Fourier

    # Encontramos el índice del pico de la frecuencia
    peak_index = np.argmax(TF)
    
    # Calculamos el ancho a media altura (FWHM)
    fwhm, _, _, _ = peak_widths(TF, [peak_index], rel_height=0.5)
    fwhm_values.append(fwhm)


# Ajuste de modelo a los datos (FWHM vs t_max).
# Utilizamos el método krogh_interpolate de scipy.interpolate, que hace un ajuste polinómico a partir de los datos disponibles para X y Y.
fit_model = krogh_interpolate(t_max_values, fwhm_values, t_max_values)


# Gráfica 1.b
plt.figure(figsize=(10, 5))
plt.loglog(t_max_values, fwhm_values, 'o-', label="Datos")
plt.loglog(t_max_values, fit_model, '--', label="Ajuste con interpolador krogh")
plt.xlabel("t_max (s)")
plt.ylabel("FWHM")
plt.legend()
plt.title("FWHM vs t_max (escala log-log)")
plt.savefig("1.b.pdf")

#punto 1c
data = np.loadtxt("OGLE-LMC-CEP-0001.dat")

# Extraer columnas
t = data[:, 0]  # Tiempo
y = data[:, 1]  # Intensidad
sigma_y = data[:, 2]  # Incertidumbre
delta_t = np.mean(np.diff(t))
f_nyquist = 1 / (2 * delta_t)
print(f'1.c) f Nyquist: "{f_nyquist:.5f}"')
delta_t  # Diferencia promedio entre puntos de tiempo
n = len(t)  # Número de puntos en la señal
frecuencias = fftfreq(n, d=delta_t)
ft=Fourier(t,y,frecuencias)
ft_mag= np.abs(ft)

indice_max = np.argmax(ft_mag)
f_true=frecuencias[indice_max]
print(f'1.c) f true: "{f_true:.5f}"')
fase_phi=np.mod(f_true*t,1)
fase_phi
plt.figure(figsize=(10, 5))
plt.scatter(fase_phi,y)
plt.xlabel("fase phi")
plt.ylabel("Intensidad")
plt.legend()
plt.title("Intensidad vs fase phi")
plt.savefig("1.c.pdf")

#punto 2a
df = pd.read_csv("H_field.csv") 
# Extraer las columnas
t = df["t"]  # Columna de tiempo
H = df["H"]  # Columna de datos
delta_t=np.mean(np.diff(t))
frecuencias=fftfreq(len(t),delta_t)
frecuencia_dominante_rapida = frecuencias[np.argmax(np.fft.rfft(H))]
frecuencia_dominante_general= np.abs(frecuencias[np.argmax(Fourier(t,H,frecuencias))])
print(f"2.a)f_fast = {frecuencia_dominante_rapida:.5f}; f_general = {frecuencia_dominante_general}")
phi_fast=np.mod(frecuencia_dominante_rapida*t,1)
phi_general=np.mod(frecuencia_dominante_general*t,1)

plt.figure(figsize=(10, 5))
plt.scatter(phi_fast,H)
plt.scatter(phi_general,H)
plt.xlabel("fase phi")
plt.ylabel("H")
plt.legend()
plt.title("H vs fase phi")
plt.savefig("2a.pdf")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import datetime as date

#Recordar quitar el '#' que llama a una función para poder ver los resultados.

#Parte 2.b Manchas Solares

#Parte 2.b.a Período del Ciclo Solar

'''
Se lee el archivo; se salta la primera linea innecesaria debido a que solo dice "American";
se saca el índice de la primera aparición de "2010" y se eliminan los datos de ese indice en adelante.
'''

archivo_date = "list_aavso-arssn_daily.txt"
df_date = pd.read_csv(archivo_date, delim_whitespace=True, skiprows=[0])
pindex = df_date.isin([2010]).any(axis=1).idxmax()
df_date = df_date.truncate(before=0, after=pindex-1)

'''
Se convierten las columnas al formato YYYY/MM/DD; se aplica la transformada rapida y la frecuencia;
se saca el índice del máximo del FFT y se busca la frecuencia que concuerda con ese valor;
la frecuencia se transforma de días a años para obtener el periodo.
'''

df_date_d = pd.to_datetime(df_date.drop(["SSN"], axis = 1))

frec_manchas = np.fft.rfftfreq(pindex, 1)
FFT_manchas = np.fft.rfft(df_date['SSN'])

ind_ciclos = np.where(abs(FFT_manchas) == abs(FFT_manchas[1:]).max())
frec_manchas_ciclos = frec_manchas[ind_ciclos]
P_solar = float((1/(frec_manchas_ciclos[0]*365.25)))
print(f'2.b.a) {P_solar = }')

#plt.loglog(frec_manchas[1:],abs(FFT_manchas[1:]))
#plt.plot(frec_manchas[1:],abs(FFT_manchas[1:]))

#Parte 2.b.b Extrapolación

'''
Se crea un df vacio para guardar los datos de la transformada inversa, además de una serie de pandas
para las fechas; se concatenan los datos de la FFT y la frecuencia a los 50 armónicos; se calculan
los días desde 2010/1/1 hasta 2025/2/17 y se crean las fechas; se genera una serie de pandas con los
datos de fechas hasta 2025; se aplica la formula para calcular la transformada inversa y se guardan
los datos; finalmente se grafican en un pdf los datos originales hasta 2010, los datos de la inversa
y se dice la predicción de manchas solares para el día 2025/2/17.
'''

def iFourier(x:NDArray[float], f:NDArray[float], d:NDArray[float]) -> NDArray[float]:
    
    IFT = np.empty(shape=0)
    
    x = x[0:50]
    f = f[0:50]
    
    d2 = pd.Series()
    
    num_dias = date.datetime(2025,2,17) - date.datetime(2010,1,1)
    num_dias = num_dias.days
    
    for i in range(num_dias+1):
        
        fecha = date.datetime(2010,1,1) + date.timedelta(days = i)
        fecha = pd.Series(fecha)
        d2 = pd.concat([d2,fecha], ignore_index=True)
        
    d3 = pd.concat([d,d2], ignore_index=True)
    t_total = pindex + num_dias
        
    for t_k in range(0, t_total+1):
        
        ifourier = 0
    
        for x_k, f_k in zip(x,f):
    
            ifourier += (x_k)*np.exp(2j*np.pi*t_k*f_k)
            
        ifourier = ifourier*(1/(pindex))
        IFT = np.append(IFT, np.real(ifourier))
            
    plt.figure(figsize=(20, 9))
    plt.plot(d,df_date["SSN"])
    plt.plot(d3,IFT)
    plt.title('Extrapolación de manchas solares')
    plt.xlabel('Años')
    plt.ylabel('Número de Manchas Solares')
    plt.savefig('2.b.pdf',bbox_inches='tight')
    plt.close()

    return IFT


IFT_manchas = iFourier(FFT_manchas, frec_manchas, df_date_d)
n_manchas_hoy = int(IFT_manchas[-1])
print(f'2.b.b) {n_manchas_hoy = }')

import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib.image as mpimg
import scipy as sc
import pandas as pd
import math 
from scipy.signal import convolve2d
import random

#crear la gráficadora
#crear la optimizadora
#crear la función de regrésion
#crear las funciones de textos
#crear funcion de ubicación
#crear la funcion de valores
def leerdocumento(nombre_archivo:str)->list:
    #loading the doc
    data = pd.read_csv(nombre_archivo,header=1,delim_whitespace=True)
    data = data[data['Year']<2012]

    Time= (data[['Year','Month','Day']])
    SSN= data[['SSN']]['SSN'].values
    return [Time,SSN]
dat=leerdocumento('list_aavso-arssn_daily.txt')
time=pd.to_datetime(dat[0])
ssn=dat[1]

def gaussianfilt(datos, sigma):
    new=sc.ndimage.gaussian_filter1d(datos,sigma)
    return new
def convolution(image, kernel):
    """
    Perform a 2D convolution between an image and a kernel.

    Parameters:
    image (ndarray): 2D array representing the grayscale image.
    kernel (ndarray): 2D array representing the kernel.

    Returns:
    ndarray: Resulting image after convolution.
    """
    # Flip the kernel both horizontally and vertically for convolution
    kernel_flipped = np.flipud(np.fliplr(kernel))

    # Perform the convolution using the kernel
    convolved_image = convolve2d(image, kernel_flipped, mode='same', boundary='fill', fillvalue=0)
    return convolved_image

def quitapersianas(image, sigma,pesodecolor):
    image = mpimg.imread(image)
    
    image = np.asarray(image)

    filter_size = 2 * int(4 * sigma + 0.5) + 1
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size//2
    n = filter_size//2
    
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian_filter[x+m, y+n] = (1/x1)*x2
            
    
    im_filtered = np.zeros_like(image, dtype=np.float32)

    im_filtered = (10000*sigma*((convolution(image, gaussian_filter)))+250*image)/pesodecolor
    return (im_filtered.astype(np.uint8))

            
catto=quitapersianas('catto.png',25,790)
capilla=quitapersianas('Noisy_Smithsonian_Castle.jpg',13.2,140000)
plt.imsave("'3.2.b'.pdf",catto,cmap='gray')
plt.imsave("'3.2.a'.pdf",capilla,cmap='gray')  




plt.imshow(capilla, cmap= 'gray')
def newfigure(alfas):
    fig = plt.figure(figsize=(8, 8))
    ax11=plt.subplot2grid((len(alfas), 2), (0, 0))
    ax11.plot(time,ssn)
    axalpha = [] # Guarda los alfas creados
    for alfa in range(len(alfas)):
        ax = plt.subplot2grid((len(alfas), 2), (alfa, 1))
        axalpha.append(ax)
        title=r'$\alpha = 1/$'+str(alfas[alfa])
        
        fig.suptitle(title)
        ax.plot(time,gaussianfilt(ssn,alfas[alfa]))
    plt.savefig("'3.1'.pdf")

newfigure([10,50,100,500,1000])


