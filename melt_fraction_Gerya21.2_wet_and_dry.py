import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from functions.mandyocIO import read_datasets

def calc_melt_dry(To,Po):

    P=np.asarray(Po)/1.0E6 # Pa -> MPa
    T=np.asarray(To)+273.15 #oC -> Kelvin

    Tsol = 1394 + 0.132899*P - 0.000005104*P**2
    cond = P>10000.0
    Tsol[cond] = 2212 + 0.030819*(P[cond] - 10000.0)

    Tliq = 2073 + 0.114*P

    X = P*0

    cond=(T>Tliq) #melt
    X[cond]=1.0

    cond=(T<Tliq)&(T>Tsol) #partial melt
    X[cond] = ((T[cond]-Tsol[cond])/(Tliq[cond]-Tsol[cond]))

    return(X)

def calc_melt_wet(To,Po):
    P=np.asarray(Po)/1.0E6 # Pa -> MPa
    T=np.asarray(To)+273.15 #oC -> Kelvin

    Tsol = 1240 + 49800/(P + 323)
    cond = P>2400.0
    Tsol[cond] = 1266 - 0.0118*P[cond] + 0.0000035*P[cond]**2

    Tliq = 2073 + 0.114*P

    X = P*0

    cond=(T>Tliq)
    X[cond]=1.0

    cond=(T<Tliq)&(T>Tsol)
    X[cond] = ((T[cond]-Tsol[cond])/(Tliq[cond]-Tsol[cond]))

    return(X)
##############################################################################################
dataset = ['pressure',
		   'temperature']
model_path = 'RFT_Clc1_DT200_PT1292oC_lit80km_1x1km2_NHK'
dataset = read_datasets(model_path, dataset)

Hmax = float(dataset.lz)
g = 15.0
rhom = 3300.0
Pmax = rhom*g*Hmax

idx = 25#len(dataset.time)-1
melt_dry = calc_melt_dry(dataset.temperature[idx], dataset.pressure[idx])
print(f'{float(np.max(dataset.pressure[idx])/1.0e9)} GPa')

plt.close()
fig, ax = plt.subplots(1, 1, figsize=(8,6), constrained_layout=True)

im = ax.contourf(np.asarray(dataset.temperature[idx]).T, np.asarray(dataset.pressure[idx]/1.0E9).T, melt_dry.T*100)
# im = ax.imshow(melt_dry.T*100., origin='lower', extent=(0, 1000, -300 + 40, 0 + 40), vmin=0, vmax=np.max(melt_dry)*100)
fig.colorbar(im, label='melt content [%]',
	orientation='vertical',
    fraction=0.015, pad=0.02)

ax.text(0, 1.01, f'Time: {np.round(float(dataset.time[idx]), 2)} Myr', transform=ax.transAxes)
ax.set_xlabel("Temperature ($^\circ$C)")
ax.set_ylabel("Pressure (GPa)")
fig.savefig("Melt_frac_Gerya_Dry_21.2_teste.png")
#############################################################

melt_wet = calc_melt_dry(dataset.temperature[idx], dataset.pressure[idx])
print(f'{float(np.max(dataset.pressure[idx])/1.0e9)} GPa') 
plt.close()
fig, ax = plt.subplots(1, 1, figsize=(8,6), constrained_layout=True)
# im = ax.contourf(np.asarray(dataset.temperature[idx]).T, np.asarray(dataset.pressure[idx]/1.0E9).T, melt_wet.T*100)
im = ax.imshow(melt_wet.T*100., origin='lower', extent=(0, 1000, -300 + 40, 0 + 40), vmin=0, vmax=np.max(melt_wet)*100)
fig.colorbar(im, label='melt content [%]',
	orientation='vertical',
    fraction=0.015, pad=0.02)
plt.xlabel("Temperature ($^\circ$C)")
plt.ylabel("Pressure (GPa)")
plt.savefig("Melt_frac_Gerya_wet_21.2_teste.png")

Hmax = 660.0E3
g = 10.0
rhom = 3300.0

Pmax = rhom*g*Hmax

P = np.linspace(0.0,Pmax,101)

T = np.linspace(0.0,2000.0,101)

TT,PP = np.meshgrid(T,P)

print(f'{np.max(PP)/1.0e9} GPa')

X = calc_melt_dry(TT,PP)
plt.close()
plt.contourf(TT, PP/1.0E9, X)
plt.xlabel("Temperature ($^\circ$C)")
plt.ylabel("Pressure (GPa)")
plt.savefig("Melt_frac_Gerya_Dry_21.2.png")


print(f'{np.max(PP)/1.0e9} GPa')
plt.close()
X = calc_melt_wet(TT,PP)
plt.contourf(TT,PP/1.0E9,X)
plt.xlabel("Temperature ($^\circ$C)")
plt.ylabel("Pressure (GPa)")
plt.savefig("Melt_frac_Gerya_Wet_21.2.png")
plt.close()

