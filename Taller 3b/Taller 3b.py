import matplotlib.pyplot as plt
import numpy as np
import csv
import math 
import imageio

import matplotlib.pyplot as plt
import numpy as np

import imageio
    

import matplotlib.animation as animation
import scipy.ndimage


#T.3b.1 Poisson en un disco:
    
#Se define un array que funcionara como la malla de la ecuacion diferencial.

malla = np.arange(-1.15,1.151,0.01)

#Se define la función que aplicara el metodo de diferencias finitas para afuera y dentro del circulo unitario.
    
@njit
def poisson_circulo_2():
    
    phi = np.zeros((len(malla)+1,len(malla)+1))
    phiprime = np.zeros((len(malla)+1,len(malla)+1))
    delta = 1
    ite = 0
    
    while ite < 15000:
        
        for i in range(len(malla)+1):
            for j in range(len(malla)+1):
                    
                hip = np.sqrt(malla[j-1]**2 + malla[i-1]**2)
                
                if hip >= 1:
                    
                    theta = np.arcsin(malla[j-1]/(np.sqrt(malla[j-1]**2 + malla[i-1]**2)))
                    phi[i,j] = np.sin(7*theta)
                    phiprime[i,j] = phi[i,j]
                
                if hip < 1:
                    
                    phiprime[i,j] = (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])*0.25 - (0.01**2)*np.pi*(malla[i-1]+malla[j-1])
                        
                    delta = abs(np.subtract(phi[i,j],phiprime[i,j]))
                    phi,phiprime = phiprime,phi
           
        ite +=1
  
    return phi

#Se grafica

e2 = poisson_circulo_2()

fig2, ax = plt.subplots(subplot_kw={"projection": "3d"})
mallax = malla
mallay = malla
mallax,mallay = np.meshgrid(mallax,mallay)
ax.set_axis_off()
plt.title('Muestra de solución', y=1)
ax.view_init(30, 30)
ax.plot_surface(mallax, mallay, e2[:-1,:-1], cmap="jet")
plt.savefig('1.png',bbox_inches='tight')

#taller 3b.2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros
L = 2        # Longitud del dominio
Nx = 100     # Número de puntos espaciales
dx = L / (Nx - 1)
c = 1        # Velocidad de la onda
dt = 0.8 * dx / c  # Asegurar estabilidad (Condición de Courant)
Nt = 200     # Número de pasos de tiempo

# Inicialización de la onda
def inicializar(Nx, dx):
    x = np.linspace(0, L, Nx)
    u0 = np.exp(-125 * (x - 1)**2)
    return x, u0

# Esquema de diferencias finitas explícitas
def evolucionar_onda(u_prev, u, C, frontera):
    Nx = len(u)
    u_next = np.zeros_like(u)
    
    for i in range(1, Nx - 1):
        u_next[i] = 2 * u[i] - u_prev[i] + C**2 * (u[i+1] - 2*u[i] + u[i-1])
    
    # Condiciones de frontera
    if frontera == 'Dirichlet':
        u_next[0] = 0
        u_next[-1] = 0
    elif frontera == 'Neumann':
        u_next[0] = u_next[1]
        u_next[-1] = u_next[-2]
    elif frontera == 'Periodicas':
        u_next[0] = u_next[-2]
        u_next[-1] = u_next[1]
    
    return u_next

# Simulación y animación
fronteras = ['Dirichlet', 'Neumann', 'Periodicas']
x, u0 = inicializar(Nx, dx)
u_prev = np.copy(u0)
u = np.copy(u0)
C = c * dt / dx

fig, axes = plt.subplots(3, 1, figsize=(6, 8))
lines = [ax.plot(x, u0, label=f'Condición {fronteras[i]}')[0] for i, ax in enumerate(axes)]
for ax in axes:
    ax.set_xlim(0, L)
    ax.set_ylim(-1, 1)
    ax.legend()

def actualizar(frame):
    global u_prev, u
    for i, frontera in enumerate(fronteras):
        u_next = evolucionar_onda(u_prev, u, C, frontera)
        lines[i].set_ydata(u_next)
        u_prev, u = u, u_next
    return lines

ani = animation.FuncAnimation(fig, actualizar, frames=Nt, interval=30, blit=True)
ani.save('2.gif', writer='ffmpeg', fps=60)

#taller 3b.3    

x = np.linspace(-35, 37, 1200, endpoint=False)  # Espaciado en x

tiempo = np.linspace(0, 20, 584)  # Tiempo desde 0 hasta 2000
  # Paso de tiempo
alfa = 0.022  # Parámetro de dispersión

# Condición inicial
y = np.cos(np.pi * x)

def compute_finite_diff_coeffs(k, orden=1):
    """
    Calcula los coeficientes de diferencias finitas para una derivada de orden arbitrario
    usando exactamente k puntos en total.
    """
    x_local = np.linspace(-k//2, k//2, k)  # Generar k puntos simétricos correctamente
    A = np.vander(x_local, increasing=True).T  # Matriz de Vandermonde (k x k)
    b = np.zeros(k)
    b[orden] = Smath.factorial(orden)  # Factorial para obtener la derivada correcta

    # Resolver el sistema para obtener los coeficientes de diferencias finitas
    coef = np.linalg.solve(A, b)

    return coef  # Retorna la lista de coeficientes para k puntos


coeffs={1:compute_finite_diff_coeffs(4, 1),3:compute_finite_diff_coeffs(4, 3) }
#solo usar k pares.
def valuefuncderivator(funcvalues, h,orden):
    """
    Calcula derivadas usando coeficientes precomputados para diferencias finitas de k puntos.
    Usa periodicidad para manejar los extremos correctamente.
    """
    N = len(funcvalues)
    derivada = np.zeros_like(funcvalues, dtype=float)
    k = len(coeffs[orden])  # Cantidad de puntos usados para la derivada
    mitad_k = k // 2  # Número de puntos a la izquierda y derecha

    for i in range(N):
        # Determinar los índices periódicos alrededor de i, asegurando exactamente k puntos
        indices = np.arange(i - mitad_k, i + mitad_k) % N  # Envoltura periódica

        # Aplicar la fórmula de diferencias finitas con los coeficientes correctos
        derivada[i] = np.sum(coeffs[orden] * funcvalues[indices])/h

    return derivada



    return derivada
def integrador(yvalues,xvalues):
    integral=np.trapz(yvalues,xvalues)
    return integral

def tokamak(x, y, alfa, tiempo):
    """
    Simula la ecuación de Korteweg–de Vries (KdV) eliminando 2 puntos en los extremos en cada iteración.
    
    Parámetros:
    - x: Array con la malla espacial
    - y: Array con la condición inicial
    - alfa: Coeficiente de la ecuación
    - tiempo: Array con la evolución temporal
    
    Retorna:
    - xfinal: Malla final después de eliminar puntos en los bordes en cada iteración.
    - yfin: Matriz con la evolución de la función en el tiempo.
    """
    yfin = np.zeros((len(tiempo), len(x)))
    yfin[0] = y
    delt = tiempo[1] - tiempo[0]
    delx = x[1] - x[0]

    for n in range(len(tiempo) - 1):
        # Cálculo de derivadas de primer y tercer orden
        d1 = valuefuncderivator(yfin[n], delx, 1)
        d3 = valuefuncderivator(yfin[n], delx, 3)

        # Evolución temporal
        yfin[n + 1] = yfin[n] - delt * (d1 + alfa**2 * d3)

        # Eliminar 2 valores en los extremos
        yfin = np.delete(yfin, [0], axis=1)  # Elimina las 2 primeras columnas
        yfin = np.delete(yfin, [-1], axis=1)  # Elimina las 2 últimas columnas
        x = np.delete(x, [0], axis=0)  # Elimina los 2 primeros puntos de x
        x = np.delete(x, [-1], axis=0)  # Elimina los 2 últimos puntos de x

    return [x, yfin]

ydt=tokamak(x,y,alfa,tiempo)

frames=[]

for t in range(len(tiempo)):  
    fig, ax = plt.subplots()
    ax.plot(ydt[0], ydt[1][t], color='blue')
    ax.set_ylim(-1.2, 1.2)
    ax.set_title(f'Tiempo {tiempo[t]:.1f}')
    
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(frame)
    plt.close(fig)
masa=[]
momento=[]
energia=[]
    
for y in ydt[1]:
    masa.append(integrador(y, ydt[0]))
    momento.append(integrador(y**2,ydt[0]))
    energia.append(integrador(y*3/3-(alfa*valuefuncderivator(y, x[1] - x[0], 1))*2,ydt[0]))
fig, ax = plt.subplots(figsize=(8, 5)) 

ax.plot(tiempo, masa, label="Masa", color="blue", linestyle="-")  
ax.plot(tiempo, momento, label="Momento", color="red", linestyle="--")  
ax.plot(tiempo, energia, label="Energía", color="green", linestyle=":")  


ax.set_title("Evolución de Masa, Momento y Energía en el Tiempo")
ax.set_xlabel("Tiempo")



ax.legend(loc="best")


ax.grid(True, linestyle="--", alpha=0.6)  
plt.savefig("'3.b'.pdf")
imageio.mimsave('3a.gif', frames, fps=29, format="GIF")

#punto 3b.4
import matplotlib.pyplot as plt
import numpy as np
import math
import imageio
    

import matplotlib.animation as animation
import scipy.ndimage



x = np.linspace(-35, 37, 1200, endpoint=False)  # Espaciado en x

tiempo = np.linspace(0, 20, 584)  # Tiempo desde 0 hasta 2000
  # Paso de tiempo
alfa = 0.022  # Parámetro de dispersión

# Condición inicial
y = np.cos(np.pi * x)

def compute_finite_diff_coeffs(k, orden=1):
    """
    Calcula los coeficientes de diferencias finitas para una derivada de orden arbitrario
    usando exactamente k puntos en total.
    """
    x_local = np.linspace(-k//2, k//2, k)  # Generar k puntos simétricos correctamente
    A = np.vander(x_local, increasing=True).T  # Matriz de Vandermonde (k x k)
    b = np.zeros(k)
    b[orden] = np.math.factorial(orden)  # Factorial para obtener la derivada correcta

    # Resolver el sistema para obtener los coeficientes de diferencias finitas
    coef = np.linalg.solve(A, b)

    return coef  # Retorna la lista de coeficientes para k puntos


coeffs={1:compute_finite_diff_coeffs(4, 1),3:compute_finite_diff_coeffs(4, 3) }
#solo usar k pares.
def valuefuncderivator(funcvalues, h,orden):
    """
    Calcula derivadas usando coeficientes precomputados para diferencias finitas de k puntos.
    Usa periodicidad para manejar los extremos correctamente.
    """
    N = len(funcvalues)
    derivada = np.zeros_like(funcvalues, dtype=float)
    k = len(coeffs[orden])  # Cantidad de puntos usados para la derivada
    mitad_k = k // 2  # Número de puntos a la izquierda y derecha

    for i in range(N):
        # Determinar los índices periódicos alrededor de i, asegurando exactamente k puntos
        indices = np.arange(i - mitad_k, i + mitad_k) % N  # Envoltura periódica

        # Aplicar la fórmula de diferencias finitas con los coeficientes correctos
        derivada[i] = np.sum(coeffs[orden] * funcvalues[indices])/h

    return derivada



    return derivada
def integrador(yvalues,xvalues):
    integral=np.trapz(yvalues,xvalues)
    return integral

def tokamak(x, y, alfa, tiempo):
    """
    Simula la ecuación de Korteweg–de Vries (KdV) eliminando 2 puntos en los extremos en cada iteración.
    
    Parámetros:
    - x: Array con la malla espacial
    - y: Array con la condición inicial
    - alfa: Coeficiente de la ecuación
    - tiempo: Array con la evolución temporal
    
    Retorna:
    - xfinal: Malla final después de eliminar puntos en los bordes en cada iteración.
    - yfin: Matriz con la evolución de la función en el tiempo.
    """
    yfin = np.zeros((len(tiempo), len(x)))
    yfin[0] = y
    delt = tiempo[1] - tiempo[0]
    delx = x[1] - x[0]

    for n in range(len(tiempo) - 1):
        # Cálculo de derivadas de primer y tercer orden
        d1 = valuefuncderivator(yfin[n], delx, 1)
        d3 = valuefuncderivator(yfin[n], delx, 3)

        # Evolución temporal
        yfin[n + 1] = yfin[n] - delt * (d1 + alfa**2 * d3)

        # Eliminar 2 valores en los extremos
        yfin = np.delete(yfin, [0], axis=1)  # Elimina las 2 primeras columnas
        yfin = np.delete(yfin, [-1], axis=1)  # Elimina las 2 últimas columnas
        x = np.delete(x, [0], axis=0)  # Elimina los 2 primeros puntos de x
        x = np.delete(x, [-1], axis=0)  # Elimina los 2 últimos puntos de x

    return [x, yfin]

ydt=tokamak(x,y,alfa,tiempo)

frames=[]

for t in range(len(tiempo)):  
    fig, ax = plt.subplots()
    ax.plot(ydt[0], ydt[1][t], color='blue')
    ax.set_ylim(-1.2, 1.2)
    ax.set_title(f'Tiempo {tiempo[t]:.1f}')
    
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(frame)
    plt.close(fig)
masa=[]
momento=[]
energia=[]
    
for y in ydt[1]:
    masa.append(integrador(y, ydt[0]))
    momento.append(integrador(y**2,ydt[0]))
    energia.append(integrador(y**3/3-(alfa*valuefuncderivator(y, x[1] - x[0], 1))**2,ydt[0]))
fig, ax = plt.subplots(figsize=(8, 5)) 

ax.plot(tiempo, masa, label="Masa", color="blue", linestyle="-")  
ax.plot(tiempo, momento, label="Momento", color="red", linestyle="--")  
ax.plot(tiempo, energia, label="Energía", color="green", linestyle=":")  


ax.set_title("Evolución de Masa, Momento y Energía en el Tiempo")
ax.set_xlabel("Tiempo")



ax.legend(loc="best")


ax.grid(True, linestyle="--", alpha=0.6)  
plt.savefig("'3.b'.pdf")
imageio.mimsave('3a.gif', frames, fps=29, format="GIF")


# Parámetros del problema
L = 2.0  # Dimensiones del tanque
dx = 0.02
dy = dx

c_base = 0.5 # Velocidad de onda base en m/s
c_lente = 0.1
dt = dx*c_lente/0.75
Tmax = 2 # Tiempo de simulación

Nx, Ny = int(L/2 / dx), int(L / dy)
x = np.linspace(0, L/2, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

# Estado de la onda
u_prev = np.zeros((Ny, Nx))
u_curr = np.zeros((Ny, Nx))
u_next = np.zeros((Ny, Nx))

# Configuración de la velocidad variable
c = np.full((Ny, Nx), c_base)

# Definir la región del lente
lente_region = ((X - L/4)**2 + 3*(Y - L/2)**2 <= 1/25) & (Y > L/2)
c[lente_region] = c_lente  # Reducir la velocidad en la región del lente

# Configuración de la pared con rendija
w_y, w_x = 0.04, 0.4  # Ancho de la pared y rendija
pared_central = (np.abs(Y - L/2) <= w_y/2) & ~((X >= (L/4 - w_x/2)) & (X <= (L/4 + w_x/2)))    
pared_bordes = (np.isclose(X, 0)) | (np.isclose(X, L/2)) | (np.isclose(Y, 0)) | (np.isclose(Y, L))
pared = pared_central | pared_bordes

# Configuración de la fuente
frec = 10  # Frecuencia en Hz
fuente_x, fuente_y = 0.5, 0.5
ix, iy = int(fuente_x/dx), int(fuente_y/dy)

# Crear figura para animación
fig, ax = plt.subplots()
im = ax.imshow(u_curr, cmap='bwr', origin='lower', extent=[0, L/2, 0, L], vmin=-0.01, vmax=0.01)
ax.contourf(X, Y, pared, levels=[0.5, 1], colors='black', alpha=0.3)
ax.contourf(X, Y, lente_region, levels=[0.5, 1], colors='green', alpha=0.6)
ax.set_title("Simulación de Onda en 2D")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
plt.colorbar(im)

# Función de actualización para la animación
def update(frame):
    global u_prev, u_curr, u_next

    # Cálculo del Laplaciano con scipy.ndimage
    laplacian = scipy.ndimage.laplace(u_curr) / dx**2

    u_next[:, :] = 2 * u_curr - u_prev + (dt**2 * c**2) * laplacian
    
    # Aplicar condiciones de frontera
    u_next[pared] = 0
    
    # Aplicar la fuente sinusoidal
    u_next[iy, ix] += np.sin(2 * np.pi * frec * frame * dt) * 0.01
    
    # Intercambio de referencias
    u_prev, u_curr, u_next = u_curr, u_next, u_prev  

    # Actualizar la imagen
    im.set_array(u_curr)
    return [im]

# Crear animación
ani = animation.FuncAnimation(fig, update, frames=int(Tmax/dt), interval=10, blit=True)

# Guardar el video sin usar imageio
ani.save("4.a.gif", writer="ffmpeg", fps=int(Tmax/dt)/20)

