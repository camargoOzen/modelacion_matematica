import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ── Constantes ────────────────────────────────────────────────────────────────
g = 9.81        # [m/s2]
P = 101325.0    # [Pa]
M_N2 = 0.02802  # [kg/mol]
R_u = 8.314     # [J/mol·K]

# ── Datos experimentales ──────────────────────────────────────────────────────
z_data = np.array([0.10, 0.22, 0.78, 1.12])           # [m]
T_data = np.array([550,  700,  1000, 850 ]) + 273.15   # [K]

# ── Parámetros de la partícula ────────────────────────────────────────────────
dp = 150e-6                  # diámetro [m]
rho_p = 810.0                # densidad [kg/m3]
cp = 1520.0                  # capacidad calorífica [J/kg·K]
Vp = np.pi * dp**3 / 6       # volumen [m3]
Aps = np.pi * dp**2          # área superficial [m2]
Apf = np.pi * dp**2 / 4      # área frontal [m2]
mp = rho_p * Vp              # masa [kg]

# ── Flujo volumétrico ─────────────────────────────────────────────────────────
Q_lpm = 230.5
Q0 = Q_lpm / 1000.0 / 60.0  # [m3/s]

# ── Condiciones iniciales y tiempo ───────────────────────────────────────────
t0 = 0.0
tf = 5.0
z0 = 0.10           # posición inicial [m]
v0 = 0.0            # velocidad inicial [m/s]
Tp0 = 550 + 273.15  # temperatura inicial [K]
y0 = [z0, v0, Tp0]

# ── Interpolador de temperatura del gas ──────────────────────────────────────
Tg_func = interp1d(z_data, T_data,
                   fill_value=(T_data[0], T_data[-1]),
                   bounds_error=False)

def T_g(z):
    return float(Tg_func(np.clip(z, 0.0, 1.12)))

# ── Propiedades del N2 ───────────────────────────────────────────────────────
def rho_g(T):
    return P * M_N2 / (R_u * T)

def mu_g(T):
    mu0, T0, S = 1.663e-5, 300.0, 111.0
    return mu0 * (T / T0)**1.5 * (T0 + S) / (T + S)

def k_g(T):
    return 0.0242 * (T / 273.15)**0.82

def cp_g(T):
    a = [29.105, -1.139e-3, 4.148e-6, -2.963e-9]
    return (a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3) / M_N2

def pr_g(T):
    return mu_g(T) * cp_g(T) / k_g(T)

# ── Geometría del reactor ─────────────────────────────────────────────────────
def D_reactor(z):
    if z <= 0.40:
        return 0.067
    elif z < 0.60:
        return 0.067 + (0.081 - 0.067) * (z - 0.40) / 0.20
    else:
        return 0.081

def area(z):
    return np.pi * D_reactor(z)**2 / 4.0

# ── Flujo másico y velocidad del gas ─────────────────────────────────────────
mdot = rho_g(T_data[0]) * Q0   # constante [kg/s]

def u_g(z, T=None):
    if T is None:
        T = T_g(z)
    return (mdot / rho_g(T)) / area(z)

# ── Correlaciones ─────────────────────────────────────────────────────────────
def reynolds(v_rel, T):
    return rho_g(T) * abs(v_rel) * dp / mu_g(T)

def coeficiente_arrastre(Re):
    if Re < 1e-10:
        return 24.0 / max(Re, 1e-10)
    if Re <= 1000:
        return 24.0 / Re * (1 + 0.15 * Re**0.687)
    return 0.44

def h_conv(v_rel, T):
    Re = reynolds(v_rel, T)
    Nu = 2.0 + 0.6 * Re**0.5 * pr_g(T)**(1.0/3.0)
    return Nu * k_g(T) / dp

# ── Sistema de EDOs ───────────────────────────────────────────────────────────
def modelo(t, y):
    z, v, Tp = y

    Tg   = T_g(z)
    rho  = rho_g(Tg)
    ug   = u_g(z, Tg)
    v_rel = ug - v

    Re = reynolds(v_rel, Tg)
    Cd = coeficiente_arrastre(Re)
    h  = h_conv(v_rel, Tg)

    dzdt  = v
    dvdt  = (
        -g
        + (rho / rho_p) * g
        + (3 * Cd * rho / (4 * rho_p * dp)) * (ug - v)**2
    )
    dTpdt = (h * Aps / (mp * cp)) * (Tg - Tp)

    return np.array([dzdt, dvdt, dTpdt])

# =============================================================================
#  IMPLEMENTACIÓN RK4 
# =============================================================================

def rk4_paso(f, t, y, h):
    k1 = f(t,           y)
    k2 = f(t + h/2,     y + h/2 * k1)
    k3 = f(t + h/2,     y + h/2 * k2)
    k4 = f(t + h,       y + h   * k3)

    return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def rk4_integrar(f, t_span, y0, h=1e-5, z_max=1.12):
    t0, tf = t_span
    y = np.array(y0, dtype=float)
    t = t0

    # Listas de almacenamiento
    t_list = [t]
    y_list = [y.copy()]

    while t < tf:
        h_actual = min(h, tf - t)

        y = rk4_paso(f, t, y, h_actual)
        t = t + h_actual

        t_list.append(t)
        y_list.append(y.copy())

        # Evento de salida: partícula alcanza la altura máxima del reactor
        if y[0] >= z_max:
            print(f"  → Partícula alcanzó z_max = {z_max} m en t = {t:.5f} s")
            break

    t_sol = np.array(t_list)
    y_sol = np.array(y_list)   # shape: (n_pasos+1, 3)

    return t_sol, y_sol


# =============================================================================
#  EJECUCIÓN
# =============================================================================
print("Integrando con RK4 (paso fijo h = 1e-5 s) ...")
t_sol, y_sol = rk4_integrar(modelo, (t0, tf), y0, h=1e-5, z_max=1.12)

# Extraer variables de estado
z_sol  = y_sol[:, 0]          # posición [m]
v_sol  = y_sol[:, 1]          # velocidad [m/s]
Tp_sol = y_sol[:, 2] - 273.15 # temperatura [°C]

print(f"Temperatura máxima de la partícula : {Tp_sol.max():.2f} °C")
print(f"Altura máxima alcanzada            : {z_sol.max():.4f} m")
print(f"Tiempo de tránsito                 : {t_sol[-1]:.5f} s")
print(f"Número de pasos realizados         : {len(t_sol)-1}")

# ── Post-proceso: recalcular h y CD en cada punto ────────────────────────────
h_sol  = np.zeros(len(t_sol))
Cd_sol = np.zeros(len(t_sol))
ug_sol = np.zeros(len(t_sol))
vr_sol = np.zeros(len(t_sol))

for i in range(len(t_sol)):
    Tg      = T_g(z_sol[i])
    ug_i    = u_g(z_sol[i], Tg)
    vrel_i  = ug_i - v_sol[i]
    Re_i    = reynolds(vrel_i, Tg)
    h_sol[i]  = h_conv(vrel_i, Tg)
    Cd_sol[i] = coeficiente_arrastre(Re_i)
    ug_sol[i] = ug_i
    vr_sol[i] = vrel_i

# =============================================================================
#  GRÁFICAS
# =============================================================================

plt.rcParams.update({
    'font.family'     : 'sans-serif',
    'font.size'       : 11,
    'axes.titlesize'  : 12,
    'axes.labelsize'  : 11,
    'legend.fontsize' : 9,
    'figure.dpi'      : 130,
})

def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(nombre, dpi=150, bbox_inches='tight')
    print(f"  Guardada: {nombre}")

COLOR = 'steelblue'

# ── 1. Posición ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_sol, z_sol, color=COLOR, lw=1.8)
ax.axhline(1.12, ls='--', color='gray', lw=0.9, label='$z_{max} = 1.12$ m')
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Posición $z$ [m]")
ax.set_title("Posición axial de la partícula de carbonizado")
ax.legend()
ax.grid(True, alpha=0.3)
guardar(fig, "fig1_posicion.png")
plt.close()

# ── 2. Velocidades ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_sol, v_sol,  color='steelblue', lw=1.8, label='$v_p$ — partícula')
ax.plot(t_sol, ug_sol, color='tomato',    lw=1.8, ls='--', label='$u_g$ — gas')
ax.plot(t_sol, vr_sol, color='seagreen',  lw=1.4, ls='-.', label='$v_{rel} = u_g - v_p$')
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Velocidad [m/s]")
ax.set_title("Velocidades de la partícula y del gas")
ax.legend()
ax.grid(True, alpha=0.3)
guardar(fig, "fig2_velocidades.png")
plt.close()

# ── 3. Temperatura ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_sol, Tp_sol, color='orangered', lw=1.8, label='$T_p(t)$ — partícula')
# Añadir perfil del gas mapeado en el tiempo
Tg_t = np.array([T_g(z) - 273.15 for z in z_sol])
ax.plot(t_sol, Tg_t, color='gray', lw=1.2, ls='--', label='$T_g(z(t))$ — gas')
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Temperatura [°C]")
ax.set_title("Temperatura de la partícula de carbonizado")
ax.legend()
ax.grid(True, alpha=0.3)
guardar(fig, "fig3_temperatura.png")
plt.close()

# ── 4. Temperatura vs Altura ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
z_plot  = np.linspace(0.10, 1.12, 400)
Tg_plot = np.array([T_g(z) - 273.15 for z in z_plot])
ax.plot(z_sol,  Tp_sol,  color='firebrick', lw=1.8, label='$T_p(z)$ — partícula')
ax.plot(z_plot, Tg_plot, color='gray',      lw=1.4, ls='--', label='$T_g(z)$ — gas (interpolado)')
ax.scatter(z_data, T_data - 273.15, color='black', zorder=5,
           s=40, label='Termocuplas PT1–PT4')
ax.set_xlabel("Altura $z$ [m]")
ax.set_ylabel("Temperatura [°C]")
ax.set_title("Temperatura vs altura en el reactor")
ax.legend()
ax.grid(True, alpha=0.3)
guardar(fig, "fig4_temperatura_altura.png")
plt.close()

# ── 5. Coeficiente de convección ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_sol, h_sol, color='purple', lw=1.8)
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("$h$ [W/m²·K]")
ax.set_title("Coeficiente de transferencia de calor por convección $h(t)$")
ax.grid(True, alpha=0.3)
guardar(fig, "fig5_h_conveccion.png")
plt.close()

# ── 6. Coeficiente de arrastre ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_sol, Cd_sol, color='darkorange', lw=1.8)
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("$C_D$ [—]")
ax.set_title("Coeficiente de arrastre $C_D(t)$")
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
guardar(fig, "fig6_CD_arrastre.png")
plt.close()

# =============================================================================
#  ANÁLISIS DE SENSIBILIDAD — variación de cp
# =============================================================================
def post_procesar(t_sol, y_sol):
    z_sol  = y_sol[:, 0]
    v_sol  = y_sol[:, 1]
    Tp_sol = y_sol[:, 2] - 273.15
    n = len(t_sol)
    h_sol  = np.zeros(n)
    Cd_sol = np.zeros(n)
    ug_sol = np.zeros(n)
    vr_sol = np.zeros(n)
    for i in range(n):
        Tg        = T_g(z_sol[i])
        ug_i      = u_g(z_sol[i], Tg)
        vrel_i    = ug_i - v_sol[i]
        Re_i      = reynolds(vrel_i, Tg)
        h_sol[i]  = h_conv(vrel_i, Tg)
        Cd_sol[i] = coeficiente_arrastre(Re_i)
        ug_sol[i] = ug_i
        vr_sol[i] = vrel_i
    return z_sol, v_sol, Tp_sol, h_sol, Cd_sol, ug_sol, vr_sol



casos_cp = {
    'C1 — 10 %  (760)'  : 0.10 * cp,
    'C2 — 25 %  (1140)' : 0.25 * cp,
    'C3 — 100 % (1520) [base]': 1.00 * cp,
    'C4 — 200 % (2280)' : 2.00 * cp,
    'C5 — 300 % (3040)' : 3.00 * cp,
}

# Colores y estilos para los 5 casos
colores   = ['#e6194b', '#f58231', '#3cb44b', '#4363d8', '#911eb4']
estilos   = ['-',       '--',      '-',       '--',      '-.']
lwidths   = [1.4,       1.4,       2.2,       1.4,       1.4]

resultados_sens = {}

print("\nAnálisis de sensibilidad (cp):")
for label, cp_val in casos_cp.items():
    cp = cp_val   # actualiza cp para el closure de modelo

    def modelo_sens(t, y, _cp=cp_val):
        z, v, Tp = y
        Tg    = T_g(z)
        rho   = rho_g(Tg)
        ug    = u_g(z, Tg)
        vrel  = ug - v
        Re    = reynolds(vrel, Tg)
        Cd    = coeficiente_arrastre(Re)
        h     = h_conv(vrel, Tg)
        dzdt  = v
        dvdt  = -g + (rho/rho_p)*g + (3*Cd*rho/(4*rho_p*dp))*vrel**2
        dTpdt = (h * Aps / (mp * _cp)) * (Tg - Tp)
        return np.array([dzdt, dvdt, dTpdt])

    t_s, y_s = rk4_integrar(modelo_sens, (t0, tf), y0)
    z_s, v_s, Tp_s, h_s, Cd_s, ug_s, vr_s = post_procesar(t_s, y_s)
    resultados_sens[label] = {
        't': t_s, 'z': z_s, 'v': v_s, 'Tp': Tp_s,
        'h': h_s, 'Cd': Cd_s, 'cp': cp_val
    }
    print(f"  {label:35s}  T_max = {Tp_s.max():.2f} °C   t_final = {t_s[-1]:.5f} s")

# ── S1. Temperatura de la partícula — todos los casos ────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for (label, res), color, ls, lw in zip(resultados_sens.items(),
                                        colores, estilos, lwidths):
    ax.plot(res['t'], res['Tp'], color=color, ls=ls, lw=lw, label=label)

ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Temperatura $T_p$ [°C]")
ax.set_title("Sensibilidad de $T_p(t)$ a la capacidad calorífica $c_p$")
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3)
guardar(fig, "sens1_Tp_tiempo.png")
plt.close()

# ── S2. Temperatura vs Altura — todos los casos ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(z_plot, Tg_plot, color='black', lw=1.2, ls=':', label='$T_g(z)$ — gas')
for (label, res), color, ls, lw in zip(resultados_sens.items(),
                                        colores, estilos, lwidths):
    ax.plot(res['z'], res['Tp'], color=color, ls=ls, lw=lw, label=label)

ax.scatter(z_data, T_data - 273.15, color='black', zorder=6, s=45)
ax.set_xlabel("Altura $z$ [m]")
ax.set_ylabel("Temperatura [°C]")
ax.set_title("Sensibilidad de $T_p(z)$ a la capacidad calorífica $c_p$")
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3)
guardar(fig, "sens2_Tp_altura.png")
plt.close()

# ── S3. Temperatura máxima vs cp ──────────────────────────────────────────────
cp_vals  = [r['cp'] for r in resultados_sens.values()]
Tmax_vals = [r['Tp'].max() for r in resultados_sens.values()]
pct_vals  = [v/cp*100 for v in cp_vals]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(pct_vals, Tmax_vals, 'o-', color='steelblue', lw=2, ms=8)
for pct, Tmax, label in zip(pct_vals, Tmax_vals, resultados_sens.keys()):
    ax.annotate(f"{Tmax:.1f} °C",
                xy=(pct, Tmax), xytext=(6, 4),
                textcoords='offset points', fontsize=8.5)
ax.axvline(100, ls='--', color='gray', lw=0.9, label='Caso base (100 %)')
ax.set_xlabel("$c_p$ como porcentaje del valor nominal [%]")
ax.set_ylabel("$T_{p,\\mathrm{max}}$ [°C]")
ax.set_title("Temperatura máxima de la partícula vs $c_p$")
ax.legend()
ax.grid(True, alpha=0.3)
guardar(fig, "sens3_Tmax_vs_cp.png")
plt.close()

# ── S4. Coeficiente h — todos los casos ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for (label, res), color, ls, lw in zip(resultados_sens.items(),
                                        colores, estilos, lwidths):
    ax.plot(res['t'], res['h'], color=color, ls=ls, lw=lw, label=label)
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("$h$ [W/m²·K]")
ax.set_title("Coeficiente de convección $h(t)$ — análisis de sensibilidad")
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
guardar(fig, "sens4_h_tiempo.png")
plt.close()

# =============================================================================
#  RESUMEN EN CONSOLA
# =============================================================================
print("\n" + "="*60)
print(f"{'Caso':<38} {'cp':>8} {'T_max':>10} {'t_final':>10}")
print("-"*60)
for label, res in resultados_sens.items():
    print(f"{label:<38} {res['cp']:>8.0f} {res['Tp'].max():>9.2f}°C "
          f"{resultados_sens[label]['t'][-1]:>9.5f}s")
print("="*60)