import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from numba import njit





#1.Balística:

#T.3a.1.Ángulo de alcance horizontal máximo:
    
#Se definen las variables y el array de la cte de fricción.

v_0_1 = 10
m_1 = 10
g_1 = 9.773
Beta_1 = np.logspace(-3,np.log10(2))

#Se define la función para solve_ivp con 'Y' sirviendo como el array de posiciones y velocidades.
#Se devuelve un array de las velocidades y aceleraciones.

@njit
def funx(t,Y,theta,B):
    
    x,y,vx,vy = Y
    
    vh = np.hypot(vx,vy)
    
    ax = -((B*vh)/m_1)*vx
    ay = -((B*vh)/m_1)*vy - g_1
    df = np.array([vx,vy,ax,ay])
    
    return df

#Se define la función para solve_ivp junto con las condiciones iniciales.

def xmaxf(theta,B):
    
    vx_0 = v_0_1*np.cos(theta)
    vy_0 = v_0_1*np.sin(theta)
    Y_0= np.array([0,0,vx_0,vy_0])

    sol = solve_ivp(
        fun=funx,
        t_span=(0,10),
        y0=Y_0,
        args=(theta,B),
        max_step=0.01,
        method ='LSODA',
        events = ev_xmaxf,
        dense_output=True)

    xmax = sol.y[0][-1]
    
    return -xmax

#Se define el evento para calcular en el solve_ivp

def ev_xmaxf(t,Y,theta,B):
    
    x,y,vx,vy = Y
    
    return y

ev_xmaxf.terminal = 2

#Se usa un minimize para hallar el angulo maximo a partir de la distancia -xmax del solve_ivp

def thetamax(theta,B):
    
    tdf = np.empty(0)
    
    for i in B:
    
        thetamax = minimize_scalar(xmaxf,args=(i),bounds=np.deg2rad([0.,50]))
        tdf = np.append(tdf,thetamax.x)
        
    return tdf

#Se grafica el angulo max vs fricción.
'''
angulos = np.rad2deg(thetamax(np.pi/4,Beta_1))
plt.plot(Beta_1,angulos)
plt.ylabel("Theta Max en °")
plt.xscale("log")
plt.savefig('1.a.pdf',bbox_inches='tight')
plt.clf()
'''

#Al igual que antes se define el solve_ivp pero se sacan las velocidades y se calcula la velocidad final.

def vmaxf(theta,B):
    
    vx_0 = v_0_1*np.cos(theta)
    vy_0 = v_0_1*np.sin(theta)
    Y_0= np.array([0,0,vx_0,vy_0])

    sol = solve_ivp(
        fun=funx,
        t_span=(0,10),
        y0=Y_0,
        args=(theta,B),
        max_step=0.01,
        method ='LSODA',
        events = ev_xmaxf,
        dense_output=True)
    vxmax = sol.y[2][-1]
    vymax = sol.y[3][-1]
    vh = np.hypot(vxmax,vymax)
    
    return vh

#Se usa calcula la diferencia de energía

def Energia(theta,B):
    
    dE = np.empty(0)
    
    for i in B:
        
        vmax = vmaxf(theta, i)
        E = m_1*0.5*(vmax**2 - v_0_1**2)
        dE = np.append(dE,E)
        
    return dE

#Se grafica la diferencia de energia vs fricción.
'''
E = Energia(np.pi/4,Beta_1)
plt.plot(Beta_1,E)
plt.ylabel("Diferencia de Energía")
plt.xscale("log")
plt.savefig('1.b.pdf',bbox_inches='tight') 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2a - Sin Larmor

alpha = 1 / 137.035999206
dt = 0.01  # Paso de tiempo
T_max = 20

# Condiciones iniciales
x, y = 1.0, 0.0
vx, vy = 0.0, 1.0

# Función de la aceleración (Fuerza de Coulomb / masa del electrón)
def aceleracion(x, y):
    r = (x**2 + y**2)**(1/2)
    ax = -x / r**3
    ay = -y / r**3
    return ax, ay

# Valores necesarios
time = [0]
x_vals, y_vals = [x], [y]
vx_vals, vy_vals = [vx], [vy]
r_vals, energy_vals = [], []

# Para calcular período
veces_por_paso = []
previous_y_sign = np.sign(y)


# Integración con RK4
num_steps = int(T_max / dt)
for step in range(num_steps):
    # k1
    ax1, ay1 = aceleracion(x, y)
    k1vx, k1vy = dt * ax1, dt * ay1
    k1x, k1y = dt * vx, dt * vy
    
    # k2
    ax2, ay2 = aceleracion(x + k1x/2, y + k1y/2)
    k2vx, k2vy = dt * ax2, dt * ay2
    k2x, k2y = dt * (vx + k1vx/2), dt * (vy + k1vy/2)
    
    # k3
    ax3, ay3 = aceleracion(x + k2x/2, y + k2y/2)
    k3vx, k3vy = dt * ax3, dt * ay3
    k3x, k3y = dt * (vx + k2vx/2), dt * (vy + k2vy/2)
    
    # k4
    ax4, ay4 = aceleracion(x + k3x, y + k3y)
    k4vx, k4vy = dt * ax4, dt * ay4
    k4x, k4y = dt * (vx + k3vx), dt * (vy + k3vy)
    
    # Actualización de posición y velocidad
    x += (k1x + 2*k2x + 2*k3x + k4x) / 6
    y += (k1y + 2*k2y + 2*k3y + k4y) / 6
    vx += (k1vx + 2*k2vx + 2*k3vx + k4vx) / 6
    vy += (k1vy + 2*k2vy + 2*k3vy + k4vy) / 6
    
    # Guardamos datos
    time.append(time[-1] + dt)
    x_vals.append(x)
    y_vals.append(y)
    vx_vals.append(vx)
    vy_vals.append(vy)
    
    # Detectar cruces por y = 0 (pasa por la misma posición en x)
    current_y_sign = np.sign(y)
    if current_y_sign != previous_y_sign:
        veces_por_paso.append(time[-1])
    previous_y_sign = current_y_sign

# Calcular el período promedio
if len(veces_por_paso) > 1:
    periodos = np.diff(veces_por_paso)
    P_sim = np.mean(periodos)  # Promedio de los períodos calculados
    
    # Cálculo del período teórico con la tercera ley de Kepler
    P_teo = 2 * np.pi  # Para r = 1 en unidades atómicas, la ley de Kepler da P = 2π
    
    print(f'P_sim = {P_sim:.5f} unidades de tiempo atómico')
    print(f'P_teo = {P_teo:.5f} unidades de tiempo atómico')


# 2b - Con Larmor

time = [0]
x_vals, y_vals = [x], [y]
vx_vals, vy_vals = [vx], [vy]
r_vals, energy_vals = [0], [0]

# Integración con RK4 considerando pérdida de energía por Larmor
num_steps = int(T_max / dt)
for step in range(num_steps):
    # k1
    ax1, ay1 = aceleracion(x, y)
    k1vx, k1vy = dt * ax1, dt * ay1
    k1x, k1y = dt * vx, dt * vy
    
    # k2
    ax2, ay2 = aceleracion(x + k1x/2, y + k1y/2)
    k2vx, k2vy = dt * ax2, dt * ay2
    k2x, k2y = dt * (vx + k1vx/2), dt * (vy + k1vy/2)
    
    # k3
    ax3, ay3 = aceleracion(x + k2x/2, y + k2y/2)
    k3vx, k3vy = dt * ax3, dt * ay3
    k3x, k3y = dt * (vx + k2vx/2), dt * (vy + k2vy/2)
    
    # k4
    ax4, ay4 = aceleracion(x + k3x, y + k3y)
    k4vx, k4vy = dt * ax4, dt * ay4
    k4x, k4y = dt * (vx + k3vx), dt * (vy + k3vy)
    
    # Actualización de posición y velocidad
    x += (k1x + 2*k2x + 2*k3x + k4x) / 6
    y += (k1y + 2*k2y + 2*k3y + k4y) / 6
    vx += (k1vx + 2*k2vx + 2*k3vx + k4vx) / 6
    vy += (k1vy + 2*k2vy + 2*k3vy + k4vy) / 6
    
    # Aplicar la pérdida de energía por Larmor
    a2 = ax1**2 + ay1**2
    v2 = vx**2 + vy**2
    v_factor = np.sqrt(max(v2 - (4/3) * a2 * alpha**3 * dt, 0)) / np.sqrt(v2)
    vx *= v_factor
    vy *= v_factor
    
    # Guardamos datos
    time.append(time[-1] + dt)
    x_vals.append(x)
    y_vals.append(y)
    vx_vals.append(vx)
    vy_vals.append(vy)
    r_vals.append(np.sqrt(x**2 + y**2))
    energy_vals.append(0.5 * (vx**2 + vy**2) - 1 / np.sqrt(x**2 + y**2))
    
    # Detectar cruces por y = 0
    current_y_sign = np.sign(y)
    if current_y_sign != previous_y_sign:
        veces_por_paso.append(time[-1])
    previous_y_sign = current_y_sign
    
    # Verificar si el electrón ha caído al núcleo
    if np.sqrt(x**2 + y**2) < 0.01:
        break

# Tiempo de caída en attosegundos (1 unidad atómica de tiempo = 24.2 as)
t_fall = time[-1] * 24.2
print(f'Tiempo de caída del electrón: {t_fall:.5f} attosegundos')

# Gráfica trayectoria del electrón
plt.figure(figsize=(6,6))
plt.plot(x_vals, y_vals, label='Órbita con pérdida de energía')
plt.scatter([0], [0], color='red', label='Protón')  # El protón en el origen
plt.xlabel("x (unidades de a0)")
plt.ylabel("y (unidades de a0)")
plt.title("Órbita del electrón con radiación de Larmor")
plt.legend()
plt.grid()
plt.savefig("2.b.XY.pdf")

# Gráfica energía total, energía cinética y radio
fig, axs = plt.subplots(3, 1, figsize=(6, 9))
axs[0].plot(time, energy_vals, label="Energía Total")
axs[0].set_ylabel("Energía")
axs[0].legend()

axs[1].plot(time, [0.5 * (vx**2 + vy**2) for vx, vy in zip(vx_vals, vy_vals)], label="Energía Cinética")
axs[1].set_ylabel("Energía Cinética")
axs[1].legend()

axs[2].plot(time, r_vals, label="Radio")
axs[2].set_xlabel("Tiempo (unidades atómicas)")
axs[2].set_ylabel("Radio")
axs[2].legend()

plt.tight_layout()
plt.savefig("2.b.diagnostics.pdf")

#T.3a.4 Comprobación de la Relatividad General
#T.3a.4a simular
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema

# Parámetros de la órbita de Mercurio
a = 0.38709893                        # UA
e = 0.20563069
mu = 1.327e20  # Constante gravitacional solar en m^3/s^2
alpha = 1e-8  # Parámetro de corrección (ajústalo según el caso)
UA_m = 1.496e11  # Conversión de UA a metros

# Condiciones iniciales (convertidas a metros y m/s)
x0 = a * (1 + e) * UA_m
y0 = 0
vx0 = 0
vy0 = np.sqrt((mu / (a * UA_m)) * ((1 - e) / (1 + e)))


# Ecuaciones de movimiento
def equations(t, state):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    factor = -mu / r**2 * (1 + alpha / r**2)
    ax = factor * (x / r)
    ay = factor * (y / r)
    return [vx, vy, ax, ay]

# Integración numérica
t_span = (0, 10)  # Un año en segundos
t_eval = np.linspace(0, t_span[1], 8000)
sol = solve_ivp(equations, t_span, [x0, y0, vx0, vy0], t_eval=t_eval, method='RK45')



# Configurar la animación
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("x (UA)")
ax.set_ylabel("y (UA)")
ax.set_title("Simulación de la órbita de Mercurio")
ax.grid()

# Dibujar el sol
ax.scatter(0, 0, color='yellow', label='Sol', s=100)

# Línea de trayectoria
trajectory, = ax.plot([], [], 'b', label='Órbita de Mercurio')
point, = ax.plot([], [], 'ro')  # Punto que representa a Mercurio

# Función de actualización para la animación
def update(frame):
    trajectory.set_data(sol.y[0][:frame+1] / UA_m, sol.y[1][:frame+1] / UA_m)
    point.set_data([sol.y[0][frame] / UA_m], [sol.y[1][frame] / UA_m])  # Convertir en listas
    return trajectory, point

# Crear animación
ani = animation.FuncAnimation(fig, update, frames=len(sol.t), interval=20, blit=True)

# Guardar animación como video
ani.save("mercury_orbit.gif", writer=animation.PillowWriter(fps=60))
plt.legend()



#punto b
# Encontrar periastro y apoastro
r_vals = np.sqrt(sol.y[0]**2 + sol.y[1]**2)
peri_indices = argrelextrema(r_vals, np.less)[0]
apo_indices = argrelextrema(r_vals, np.greater)[0]

# Calcular ángulos de precesión
times_peri = sol.t[peri_indices]
angles_peri = np.arctan2(sol.y[1][peri_indices], sol.y[0][peri_indices])
angles_peri = np.unwrap(angles_peri)  # Evita saltos bruscos

# Convertir ángulos a grados y a arcosegundos
angles_degrees = np.degrees(angles_peri)
angles_arcsec = angles_degrees * 3600  # 1° = 3600 arcsec

# Ajuste lineal de la precesión
coeffs = np.polyfit(times_peri, angles_arcsec, 1)
precession_rate = coeffs[0]  # Pendiente en arcsec/siglo

# Graficar precesión
plt.figure(figsize=(8, 6))
plt.plot(times_peri / (3.154e7 * 100), angles_arcsec, 'bo', label="Datos")
plt.plot(times_peri / (3.154e7 * 100), np.polyval(coeffs, times_peri), 'r-', label=f"Ajuste: {precession_rate:.4f} arcsec/siglo")
plt.xlabel("Tiempo (siglos)")
plt.ylabel("Ángulo de precesión (arcsec)")
plt.title("Precesión de la órbita de Mercurio")
plt.legend()
plt.grid()
plt.savefig("a.3.b.pdf")






#T.3a.4.Cuantización de la Energía:
    
#Se definen las variables y el array de valores iniciales.
    
h_4 = 0.01
x_4 = np.arange(0,6.01,h_4)
E_4 = np.arange(0,10,0.5)
fun_inicial = np.array([[1,0],[0,1]])

#Se define la función para el integrador RK4, 'Y' sirve como el array de f y su derivada.
#Se devuelve un array de la primera y segunda derivada

def f_4_prima(r,x,E):
    
    f = r[0]
    y = r[1]
    
    df = y
    dy = (x**2 - 2*E)*f
    
    return np.array([df,dy], float)

#Se define el integrador RK4

def f_4(val,E):
    
    f = val[0]
    y = val[1]
    
    r = np.array([f,y], float)
    ff = np.empty(0)
    ff = np.append(ff,f)
    
    for i in range(0,len(x_4)-1):
        
        k1 = f_4_prima(r,x_4[i],E)
        k2 = f_4_prima(r+0.5*h_4*k1,x_4[i]+0.5*h_4,E)
        k3 = f_4_prima(r+0.5*h_4*k2,x_4[i]+0.5*h_4,E)
        k4 = f_4_prima(r+h_4*k3,x_4[i]+h_4,E)
        r += h_4*(k1+2*k2+2*k3+k4)/6
        ff = np.append(ff,r[0])
        
    if val[0] == 1:
        
        ff = ff/4
        ff2 = np.flip(ff)
        
    if val[0] == 0:
        
        ff = ff/2
        ff2 = -np.flip(ff)
    
    ff_final = np.concatenate((ff2, ff), axis = 0)
    
    return ff_final

#Se encuentran las energias permitidas

def encontrar_E(val,f):
    
    E_per_sim = np.empty(0)
    E_per_anti = np.empty(0)
    
    for i in E_4:
        
        f = f_4(val[0],i)
        f_abs = abs(f)
        
        if f_abs[-1] < 0.1:
            
            E_per_sim = np.append(E_per_sim, i)
            
    for i in E_4:
        
        f = f_4(val[1],i)
        f_abs = abs(f)
        
        if f_abs[-1] < 0.1:
            
            E_per_anti = np.append(E_per_anti, i)
            
    return (E_per_sim,E_per_anti) 


E_permitidos = encontrar_E(fun_inicial,f_4)

x_4_flip = -np.flip(x_4)
x_4_final = np.concatenate((x_4_flip, x_4), axis=0)


x_parabola = np.arange(0,6,0.1)
x_parabola = np.concatenate((-np.flip(x_parabola),x_parabola),axis = 0)
y_parabola = (0.7*x_parabola)**2


#Y se grafican

def grafica(E):
    
    plt.figure(figsize=(5,8))
    ax = plt.gca()
    ax.set_xlim([-6, 6])
    ax.set_ylim([0, 10])
    plt.plot(x_parabola,y_parabola,color = '#D3D3D3', linestyle = '-')
    
    for i in E[0]:
    
        plt.axhline(y = i, color = '#D3D3D3', linestyle = '-') 
        plt.plot(x_4_final,f_4(fun_inicial[0],i)+i)
  
    for i in E[1]:
    
        plt.axhline(y = i, color = '#D3D3D3', linestyle = '-')   
        plt.plot(x_4_final,f_4(fun_inicial[1],i)+i)
        
'''      
grafica(E_permitidos)    
plt.ylabel("Energía")
plt.savefig('4.pdf',bbox_inches='tight')
'''


