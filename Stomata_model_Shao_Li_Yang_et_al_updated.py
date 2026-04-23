# Stomata model - Wt and mutants simulations reported in Shao-Li Yang et al. 2026
import numpy as np
from scipy.integrate import odeint
import os
import matplotlib.pyplot as plt

# ODE stomata aperture model
def Stomata(a,t,parameters): 
    # Parameters controlling TxD regulatory branches
    Km1=parameters['Km1'] # T activation of TOT3 (and indirectly OST1) - Branch 1
    Km2=parameters['Km2'] # OST1 activation of UBP24, resulting in stabilization of AHA1 - Branch 2
    Km3=parameters['Km3'] # OST1 inactivation of TOT3 - Branch 3
    Km4=parameters['Km4'] # OST1 activation of SLAC1 - Branch 4
    Km5=parameters['Km5'] # OST1 independent ABA-regulation of stomata - Branch 5
    
    # Mutant parameters - set to disable specific regulations
    mutost1=parameters['mutost1'] # ost1 mutant 
    mutslac1=parameters['mutslac1'] # slac1 mutant 
    noupb24=parameters['noupb24'] # no UBP24 regulation by OST1
    notot3inact=parameters['notot3inact'] # no TOT3 inactivation by OST1
    
    # Input conditions and parameters
    ABA=parameters['aba']
    Temp=parameters['temp']
    prod=0.01
    KmABA=25
    KmPP2C=50
    KmTOT3P2=50
    KmUBP24=50 
    decay=prod/100 # for all variables to have a max value of of 100
    phosph=prod*100  # faster than production
    dephosph_basal=phosph/10 # basal dephosphorylation < active phosphorylation
    dephosph_active=phosph #  active dephosphorylation == active phosphorylation

    # Initial condition
    SNRK2S=a[0]
    TOT3=a[1]
    TOT3P1=a[2]
    TOT3P2=a[3]
    PP2C=a[4]
    AHA=a[5]
    AHAP=a[6]
    UBP24=a[7]
    UBP24P=a[8]
    SLAC1=a[9]
    SLAC1P=a[10]

    # Regulatory functions
    ABAregulation=ABA**3/(KmABA**3+ABA**3)
    Tempregulation=Temp**3/(Km1**3+Temp**3) 
    SNRK2SRegulationUBP4=SNRK2S**2/(Km2**2+SNRK2S**2)*(1-noupb24)
    SNRK2SRegulationTOT3i=SNRK2S**2/((Km3)**2+SNRK2S**2)*(1-notot3inact) 
    SNRK2SRegulationSLAC1=SNRK2S**2/((Km4)**2+SNRK2S**2) 
    PP2CRegulation=PP2C**2/(KmPP2C**2+PP2C**2)
    PP2CRegulationneg=KmPP2C**2/(KmPP2C**2+PP2C**2)
    TOT3P2Regulation=TOT3P2**2/(KmTOT3P2**2+TOT3P2**2)
    UBP24PRegulation=KmUBP24**2/(KmUBP24**2+UBP24P**2) 
    
    # ODEs
    dSNRK2S=prod*(PP2CRegulationneg+Tempregulation)*(1-mutost1)-decay*SNRK2S 
    dTOT3=prod-decay*TOT3-phosph*TOT3*(SNRK2SRegulationTOT3i+Tempregulation)+dephosph_basal*(TOT3P1+TOT3P2)+dephosph_active*(TOT3P1*PP2CRegulation)
    dTOT3P1=-decay*TOT3P1+phosph*TOT3*(SNRK2SRegulationTOT3i)-dephosph_basal*(TOT3P1)-dephosph_active*(TOT3P1*PP2CRegulation)  # Branch3
    dTOT3P2=-decay*TOT3P2+phosph*TOT3*(Tempregulation)-dephosph_basal*(TOT3P2) # Branch1
    dPP2C=prod*(1-ABAregulation)-decay*PP2C 
    dAHA=prod-decay*AHA*(0.1+0.9*UBP24PRegulation)-phosph*AHA*(TOT3P2Regulation)+dephosph_basal*AHAP+dephosph_active*(AHAP*PP2CRegulation)#+dephosph_active*(AHAP*SNRK2SRegulationSLAC1) 
    dAHAP=-decay*AHAP*(0.1+0.9*UBP24PRegulation)+phosph*AHA*(TOT3P2Regulation)-dephosph_basal*AHAP-dephosph_active*(AHAP*PP2CRegulation)#-dephosph_active*(AHAP*SNRK2SRegulationSLAC1)  
    dUBP24=prod-decay*UBP24-phosph*UBP24*(SNRK2SRegulationUBP4)+dephosph_basal*UBP24P  # Branch2
    dUBP24P=-0.5*decay*UBP24P+phosph*UBP24*(SNRK2SRegulationUBP4)-dephosph_basal*UBP24P # UBP24P is more stable 
    dSLAC1=prod*(1-mutslac1)-decay*SLAC1-SLAC1*SNRK2SRegulationSLAC1+SLAC1P*dephosph_basal+SLAC1P*dephosph_active*PP2CRegulationneg
    dSLAC1P=SLAC1*SNRK2SRegulationSLAC1-dephosph_basal*SLAC1P-decay*SLAC1P-SLAC1P*dephosph_active*PP2CRegulationneg # Branch4
    return(dSNRK2S,dTOT3,dTOT3P1,dTOT3P2,dPP2C,dAHA,dAHAP,dUBP24,dUBP24P,dSLAC1,dSLAC1P)

os.makedirs('output', exist_ok=True)

########### Initialize parameters and variables
mut_test=range(4) # 4 simulations: Wt, ost1, Wt no UBP24 regulation, Wt no TOT3Pi regulation
Finalvalues = np.zeros((len(mut_test), 2)) # For each simulation we save stomata aperture for min and max temperature, representing 21 and 42 degrees C

timerunning=50000.1 
times = np.arange(0, timerunning, 0.2)
IC = [0,0,0,0,0,0,0,0,0,0,0]
tempTest = np.arange(9, 13.6, 0.5)  # min and max, representing temperaturesfrom 21 to 42 degrees C
tempnames = [str(int(round(v))) for v in np.linspace(21, 42, len(tempTest))]
basal_stomata_aperture = 0.1
aha_output = np.zeros((len(tempTest), len(mut_test),np.shape(IC)[0]))  # here we save all nodes for Wt and ost1 

# Km values regulating different branches of stomata aperture regulation
Km1=40
Km2=50
Km3=75
Km4=75
Km5=20*2 # update - ABA OST1-independent pathway
KmAHA=50
KmSLAC1P=50 # update - slac1 now modelled with a un- and a -phosphorylated form.

########### Initialize parameters and variables
timerunning=50000.1 
times = np.arange(0, timerunning, 0.2)
IC = [0,0,0,0,0,0,0,0,0,0,0]
tempTest = np.arange(9, 13.6, 0.5)  # min and max, representing temperaturesfrom 21 to 42 degrees C
tempnames = [str(int(round(v))) for v in np.linspace(21, 42, len(tempTest))]
basal_stomata_aperture = 0.1

#############################################################################
########### Wt, ost1 mutant, and variants simulation for temperature response
#############################################################################

mut_test=range(4) # 4 simulations: Wt, ost1, Wt no UBP24 regulation, Wt no TOT3Pi regulation
Finalvalues = np.zeros((len(mut_test), 2)) # For each simulation we save stomata aperture for min and max temperature, representing 21 and 42 degrees C
aha_output = np.zeros((len(tempTest), len(mut_test),np.shape(IC)[0]))  # here we save all nodes for Wt and ost1 

## Control 
idx=0 
cont=-1
for j in range(len(tempTest)):
    parameters = {'Km1':Km1,'Km2':Km2,'Km3':Km3,'Km4':Km4,'Km5':Km5,'aba': 0, 'temp': tempTest[j],'mutost1': 0,'mutslac1': 0,'noupb24':0,'notot3inact':0}  

    result = odeint(Stomata, IC, times, args=(parameters,))
    aha_output[j,idx,:] = result[-1,:]  
    stomata_aperture=(basal_stomata_aperture+result[-1, 6]**2/ (KmAHA**2 + result[-1, 6]**2)*(KmSLAC1P**2/ (KmSLAC1P**2 + result[-1, 10]**2)))*(Km5**2/(Km5**2+0**2))
    if j==0 or j==len(tempTest)-1:
        cont+=1
        Finalvalues[idx, cont] = stomata_aperture

#OST1 mutant 
idx=1 
cont=-1
for j in range(len(tempTest)):
    parameters = {'Km1':Km1,'Km2':Km2,'Km3':Km3,'Km4':Km4,'Km5':Km5,'aba': 0, 'temp': tempTest[j],'mutost1': 1,'mutslac1': 0,'noupb24':0,'notot3inact':0}
    result = odeint(Stomata, IC, times, args=(parameters,))
    aha_output[j,idx,:] = result[-1,:]  
    stomata_aperture=(basal_stomata_aperture+result[-1, 6]**2/ (KmAHA**2 + result[-1, 6]**2)*(KmSLAC1P**2/ (KmSLAC1P**2 + result[-1, 10]**2)))*(Km5**2/(Km5**2+0**2))
    if j==0 or j==len(tempTest)-1:
        cont+=1
        Finalvalues[idx, cont] = stomata_aperture


# Wt no UBP24 regulation 
idx=2 
cont=-1
for j in range(len(tempTest)):
    parameters = {'Km1':Km1,'Km2':Km2,'Km3':Km3,'Km4':Km4,'Km5':Km5,'aba': 0, 'temp': tempTest[j],'mutost1': 0,'mutslac1': 0,'noupb24':1,'notot3inact':0}
    result = odeint(Stomata, IC, times, args=(parameters,))
    aha_output[j,idx,:] = result[-1,:]  
    stomata_aperture=(basal_stomata_aperture+result[-1, 6]**2/ (KmAHA**2 + result[-1, 6]**2)*(KmSLAC1P**2/ (KmSLAC1P**2 + result[-1, 10]**2)))*(Km5**2/(Km5**2+0**2))
    if j==0 or j==len(tempTest)-1:
        cont+=1
        Finalvalues[idx, cont] = stomata_aperture

# Wt no TOT3Pi regulation 
idx=3 
cont=-1
for j in range(len(tempTest)):
    parameters = {'Km1':Km1,'Km2':Km2,'Km3':Km3,'Km4':Km4,'Km5':Km5,'aba': 0, 'temp': tempTest[j],'mutost1': 0,'mutslac1': 0,'noupb24':0,'notot3inact':1}
    result = odeint(Stomata, IC, times, args=(parameters,))
    aha_output[j,idx,:] = result[-1,:]  
    stomata_aperture=(basal_stomata_aperture+result[-1, 6]**2/ (KmAHA**2 + result[-1, 6]**2)*(KmSLAC1P**2/ (KmSLAC1P**2 + result[-1, 10]**2)))*(Km5**2/(Km5**2+0**2))
    if j==0 or j==len(tempTest)-1:
        cont+=1
        Finalvalues[idx, cont] = stomata_aperture

#### Plot 1 - Wt, ost1 mutant, and variants simulation for temperature response
colores2=['black', 'red', 'blue', 'green']
labels = ['Wt', 'ost1', 'Wt, no OST1-UPB24', 'Wt, no OST1-TOT3i']
fig, ax = plt.subplots(figsize=(8, 4))

x = np.arange(len(labels))
width = 0.35

ax.bar(x - width/2, Finalvalues[:, 0], width, label='21°', color='#ADD8E6')
ax.bar(x + width/2, Finalvalues[:, 1], width, label='42°', color='#A52A2A')
ax.set_ylabel('Stomata aperture')
ax.set_title('Stomata aperture in response to temperature')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper left')
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('output/Wt-ost1-variants.svg', dpi=300, bbox_inches='tight')
plt.savefig('output/Wt-ost1-variants.png', dpi=200, bbox_inches='tight')
plt.show()


############################################################################
###########  Wt temperature response
############################################################################
#### Plot 2 - Wt stomata aperture response to different simulated temperatures
stomata_temperatures = (basal_stomata_aperture+(aha_output[:, :, 6]**2/ (KmAHA**2 + aha_output[:, :, 6]**2))*(KmSLAC1P**2/ (KmSLAC1P**2 + aha_output[:, :, 10]**2)))*(Km5**2/(Km5**2+0**2))  
fig, ax = plt.subplots(figsize=(8, 6))
plottingvalues=[0,3,7,-1] # plotting subsample of temperatures simulated
tempNamesSubsample = [tempnames[i] for i in plottingvalues]
cont=0
for idx in plottingvalues:
    ax.bar(cont, stomata_temperatures[idx, 0], color=['darkblue', 'beige', 'darkorange', 'brown'][cont], width=0.6)
    cont+=1
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Stomata aperture')
ax.set_title('Stomata aperture vs Temperature')
ax.set_xticks(range(len(tempNamesSubsample)))
ax.set_xticklabels(tempNamesSubsample)
ax.set_xlim()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('output/TemperaturexStomata.png', dpi=200, bbox_inches='tight')
plt.savefig('output/TemperaturexStomata.svg', dpi=300, bbox_inches='tight')
plt.show()


############################################################################
####################### Wt, ost1, slac1 mutants: control, Heat, ABA, and Heat+ABA
############################################################################
Finalvalues = [0,0,0,0]
ABAvalues=[0,50] #extreme values for ABA: 0 and 50 microM
labels=['Control','Heat','ABA','Heat x ABA']
colores=["black","darkred","darkblue","darkgreen"]

times = np.arange(0, timerunning, 0.2)
IC = [0,0,0,0,0,0,0,0,0,0,0]
tempTest = np.array([tempTest[0], tempTest[-1]])
tempnames = ("21","42")


#### Combined plot for all three simulations
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
simulation_configs = [
    {'m1': 0, 'm2': 0, 'title': 'Wild-type', 'ax': axes[0]},
    {'m1': 1, 'm2': 0, 'title': 'ost1 mutant', 'ax': axes[1]},
    {'m1': 0, 'm2': 1, 'title': 'slac1 mutant', 'ax': axes[2]}
]

for config in simulation_configs:
    m1 = config['m1']
    m2 = config['m2']
    ax = config['ax']
    Finalvalues = [0, 0, 0, 0]
    
    aha_output = np.zeros((len(tempTest)*len(ABAvalues), np.shape(IC)[0])) 
    cont = -1
    for abaidx in ABAvalues: # ABA 0 and 50 
        for tempidx in tempTest: # Temp 21 and 42 degrees C
            cont += 1
            parameters = {'Km1':Km1,'Km2':Km2,'Km3':Km3,'Km4':Km4,'Km5':Km5,'aba': abaidx, 'temp': tempidx,'mutost1': m1,'mutslac1': m2,'noupb24':0,'notot3inact':0}
            result = odeint(Stomata, IC, times, args=(parameters,))
            aha_output[cont,:] = result[-1,:]  
            stomata_aperture = (basal_stomata_aperture+result[-1, 6]**2/ (KmAHA**2 + result[-1, 6]**2)*(KmSLAC1P**2/ (KmSLAC1P**2 + result[-1, 10]**2)))*(Km5**2/(Km5**2+abaidx**2))
            Finalvalues[cont] = stomata_aperture
    
    ax.bar(labels, Finalvalues, color=colores)
    ax.set_ylabel('Stomata aperture')
    ax.set_title(f'Stomata aperture - {config["title"]}')
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('output/Heat_x_ABA.png', dpi=200, bbox_inches='tight')
plt.savefig('output/Heat_x_ABA.svg', dpi=300, bbox_inches='tight')
plt.show()