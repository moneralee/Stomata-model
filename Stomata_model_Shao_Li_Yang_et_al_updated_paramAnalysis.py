# Stomata model - Wt and mutants simulations reported in Shao-Li Yang et al. 2026
import numpy as np
from scipy.integrate import odeint
import os
import matplotlib.pyplot as plt
import time

start_time = time.time()

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
    prod=parameters['prod']
    KmABA=parameters['KmABA']
    KmPP2C=parameters['KmPP2C']
    KmTOT3P2=parameters['KmTOT3P2']
    KmUBP24=parameters['KmUBP24']
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

param_analysis=10000 # number of parameter sets simulated. 

########### Initialize parameters and variables
timerunning=50000.1 
times = np.arange(0, timerunning, 0.2)
IC = [0,0,0,0,0,0,0,0,0,0,0]
tempTest = np.arange(9, 13.6, 0.5)  # min and max, representing temperaturesfrom 21 to 42 degrees C
tempnames = [str(int(round(v))) for v in np.linspace(21, 42, len(tempTest))]
basal_stomata_aperture = 0.1

mut_test=range(4) # 4 simulations: Wt, ost1, Wt no UBP24 regulation, Wt no TOT3Pi regulation
Finalvalues = np.zeros((len(mut_test), 2,param_analysis)) # For each simulation we save stomata aperture for min and max temperature, representing 21 and 42 degrees C
stomata_temperatures = np.zeros((4, param_analysis))  # here we save stomata aperture for all temperatures simulated, for Wt and ost1 mutants, and variants.
FinalvaluesHxD = np.zeros((4*3, param_analysis)) # For each simulation we save stomata aperture for min and max temperature, representing 21 and 42 degrees C

for param in range(param_analysis):
    ########### Initialize parameters and variables
    aha_output = np.zeros((len(tempTest), len(mut_test),np.shape(IC)[0]))  # here we save all nodes for Wt and ost1 

    # Km values regulating different branches of stomata aperture regulation
    # All parameters are randomly sampled within a 10% range around the values.
    Km1=np.random.uniform(40 * 0.9, 40 * 1.1)
    Km2=np.random.uniform(50 * 0.9, 50 * 1.1)
    Km3=np.random.uniform(75 * 0.9, 75 * 1.1)
    Km4=np.random.uniform(75 * 0.9, 75 * 1.1)
    Km5=np.random.uniform(40 * 0.9, 40 * 1.1) # update - ABA OST1-independent pathway
    KmAHA=np.random.uniform(50 * 0.9, 50 * 1.1)
    KmSLAC1P=np.random.uniform(50 * 0.9, 50 * 1.1) # update - slac1 now modelled with a un- and a -phosphorylated form.
    prod=np.random.uniform(0.01 * 0.9, 0.01 * 1.1)
    KmABA=np.random.uniform(25 * 0.9, 25 * 1.1)
    KmPP2C=np.random.uniform(50 * 0.9, 50 * 1.1)
    KmTOT3P2=np.random.uniform(50 * 0.9, 50 * 1.1)
    KmUBP24=np.random.uniform(50 * 0.9, 50 * 1.1)

    #############################################################################
    ########### Wt, ost1 mutant, and variants simulation for temperature response
    #############################################################################

    mut_test=range(4) # 4 simulations: Wt, ost1, Wt no UBP24 regulation, Wt no TOT3Pi regulation
    #Finalvalues = np.zeros((len(mut_test), 2)) # For each simulation we save stomata aperture for min and max temperature, representing 21 and 42 degrees C
    aha_output = np.zeros((len(tempTest), len(mut_test),np.shape(IC)[0]))  # here we save all nodes for Wt and ost1 

    ## Control 
    idx=0 
    cont=-1
    for j in range(len(tempTest)):
        parameters = {'Km1':Km1,'Km2':Km2,'Km3':Km3,'Km4':Km4,'Km5':Km5,'aba': 0, 'temp': tempTest[j],'mutost1': 0,'mutslac1': 0,'noupb24':0,'notot3inact':0,'prod':prod,'KmABA':KmABA,'KmPP2C':KmPP2C, 'KmTOT3P2':KmTOT3P2,'KmUBP24':KmUBP24}  

        result = odeint(Stomata, IC, times, args=(parameters,))
        aha_output[j,idx,:] = result[-1,:]  
        stomata_aperture=(basal_stomata_aperture+result[-1, 6]**2/ (KmAHA**2 + result[-1, 6]**2)*(KmSLAC1P**2/ (KmSLAC1P**2 + result[-1, 10]**2)))*(Km5**2/(Km5**2+0**2))
        if j==0 or j==len(tempTest)-1:
            cont+=1
            Finalvalues[idx, cont,param] = stomata_aperture

    #OST1 mutant 
    idx=1 
    cont=-1
    for j in range(len(tempTest)):
        parameters = {'Km1':Km1,'Km2':Km2,'Km3':Km3,'Km4':Km4,'Km5':Km5,'aba': 0, 'temp': tempTest[j],'mutost1': 1,'mutslac1': 0,'noupb24':0,'notot3inact':0,'prod':prod,'KmABA':KmABA,'KmPP2C':KmPP2C, 'KmTOT3P2':KmTOT3P2,'KmUBP24':KmUBP24}  
        result = odeint(Stomata, IC, times, args=(parameters,))
        aha_output[j,idx,:] = result[-1,:]  
        stomata_aperture=(basal_stomata_aperture+result[-1, 6]**2/ (KmAHA**2 + result[-1, 6]**2)*(KmSLAC1P**2/ (KmSLAC1P**2 + result[-1, 10]**2)))*(Km5**2/(Km5**2+0**2))
        if j==0 or j==len(tempTest)-1:
            cont+=1
            Finalvalues[idx, cont,param] = stomata_aperture


    # Wt no UBP24 regulation 
    idx=2 
    cont=-1
    for j in range(len(tempTest)):
        parameters = {'Km1':Km1,'Km2':Km2,'Km3':Km3,'Km4':Km4,'Km5':Km5,'aba': 0, 'temp': tempTest[j],'mutost1': 0,'mutslac1': 0,'noupb24':1,'notot3inact':0,'prod':prod,'KmABA':KmABA,'KmPP2C':KmPP2C, 'KmTOT3P2':KmTOT3P2,'KmUBP24':KmUBP24}
        result = odeint(Stomata, IC, times, args=(parameters,))
        aha_output[j,idx,:] = result[-1,:]  
        stomata_aperture=(basal_stomata_aperture+result[-1, 6]**2/ (KmAHA**2 + result[-1, 6]**2)*(KmSLAC1P**2/ (KmSLAC1P**2 + result[-1, 10]**2)))*(Km5**2/(Km5**2+0**2))
        if j==0 or j==len(tempTest)-1:
            cont+=1
            Finalvalues[idx, cont,param] = stomata_aperture

    # Wt no TOT3Pi regulation 
    idx=3 
    cont=-1
    for j in range(len(tempTest)):
        parameters = {'Km1':Km1,'Km2':Km2,'Km3':Km3,'Km4':Km4,'Km5':Km5,'aba': 0, 'temp': tempTest[j],'mutost1': 0,'mutslac1': 0,'noupb24':0,'notot3inact':1,'prod':prod,'KmABA':KmABA,'KmPP2C':KmPP2C, 'KmTOT3P2':KmTOT3P2,'KmUBP24':KmUBP24}
        result = odeint(Stomata, IC, times, args=(parameters,))
        aha_output[j,idx,:] = result[-1,:]  
        stomata_aperture=(basal_stomata_aperture+result[-1, 6]**2/ (KmAHA**2 + result[-1, 6]**2)*(KmSLAC1P**2/ (KmSLAC1P**2 + result[-1, 10]**2)))*(Km5**2/(Km5**2+0**2))
        if j==0 or j==len(tempTest)-1:
            cont+=1
            Finalvalues[idx, cont,param] = stomata_aperture

    ###########################################
    # Stomata aperture at different temperatures
    plottingvalues=[0,3,7,-1] # plotting subsample of temperatures simulated
    stomata_temperatures[0,param] = (basal_stomata_aperture+(aha_output[0, 0, 6]**2/ (KmAHA**2 + aha_output[0, 0, 6]**2))*(KmSLAC1P**2/ (KmSLAC1P**2 + aha_output[0, 0, 10]**2)))*(Km5**2/(Km5**2+0**2))  
    stomata_temperatures[1,param] = (basal_stomata_aperture+(aha_output[3, 0, 6]**2/ (KmAHA**2 + aha_output[3, 0, 6]**2))*(KmSLAC1P**2/ (KmSLAC1P**2 + aha_output[3, 0, 10]**2)))*(Km5**2/(Km5**2+0**2))  
    stomata_temperatures[2,param] = (basal_stomata_aperture+(aha_output[7, 0, 6]**2/ (KmAHA**2 + aha_output[7, 0, 6]**2))*(KmSLAC1P**2/ (KmSLAC1P**2 + aha_output[7, 0, 10]**2)))*(Km5**2/(Km5**2+0**2))  
    stomata_temperatures[3,param] = (basal_stomata_aperture+(aha_output[-1, 0, 6]**2/ (KmAHA**2 + aha_output[-1, 0, 6]**2))*(KmSLAC1P**2/ (KmSLAC1P**2 + aha_output[-1, 0, 10]**2)))*(Km5**2/(Km5**2+0**2))  

    ###########################################
    # Stomata aperture HxD Wt and mutants
    ABAvalues=[0,50] #extreme values for ABA: 0 and 50 microM
    labels=['Control','Heat','ABA','Heat x ABA']
    colores=["black","darkred","darkblue","darkgreen"]

    tempTestHxD = np.array([tempTest[0], tempTest[-1]])
    tempnamesHxD = ("21","42")


    #### Combined plot for all three simulations
    simulation_configs = [
        {'m1': 0, 'm2': 0},
        {'m1': 1, 'm2': 0},
        {'m1': 0, 'm2': 1}
    ]
    cont = -1
    for config in simulation_configs:
        m1 = config['m1']
        m2 = config['m2']
        
        aha_output = np.zeros((len(tempTestHxD)*len(ABAvalues)*3, np.shape(IC)[0])) 
        for abaidx in ABAvalues: # ABA 0 and 50 
            for tempidx in tempTestHxD: # Temp 21 and 42 degrees C
                cont += 1
                parameters = {'Km1':Km1,'Km2':Km2,'Km3':Km3,'Km4':Km4,'Km5':Km5,'aba': abaidx, 'temp': tempidx,'mutost1': m1,'mutslac1': m2,'noupb24':0,'notot3inact':0,'prod':prod,'KmABA':KmABA,'KmPP2C':KmPP2C, 'KmTOT3P2':KmTOT3P2,'KmUBP24':KmUBP24}
                result = odeint(Stomata, IC, times, args=(parameters,))
                aha_output[cont,:] = result[-1,:]  
                stomata_aperture = (basal_stomata_aperture+result[-1, 6]**2/ (KmAHA**2 + result[-1, 6]**2)*(KmSLAC1P**2/ (KmSLAC1P**2 + result[-1, 10]**2)))*(Km5**2/(Km5**2+abaidx**2))
                FinalvaluesHxD[cont,param] = stomata_aperture


##################################################################################
############################### Plots #########################################
#### Plot 1 - Wt, ost1 mutant, and variants simulation for temperature response
fig, ax = plt.subplots(figsize=(10, 5))
violin_data = [
    Finalvalues[0, 0, :],
    Finalvalues[0, 1, :],
    Finalvalues[1, 0, :],
    Finalvalues[1, 1, :],
    Finalvalues[2, 0, :],
    Finalvalues[2, 1, :],
    Finalvalues[3, 0, :],
    Finalvalues[3, 1, :]
]
parts = ax.violinplot(violin_data, positions=[1, 2, 3, 4, 5, 6, 7, 8], widths=0.7, showmeans=False, showextrema=False)
for body, color in zip(parts['bodies'], ['#ADD8E6', '#A52A2A', '#ADD8E6', '#A52A2A', '#ADD8E6', '#A52A2A', '#ADD8E6', '#A52A2A']):
    body.set_facecolor(color)
    body.set_edgecolor('black')
    body.set_alpha(0.8)
box = ax.boxplot(
    violin_data,
    positions=[1, 2, 3, 4, 5, 6, 7, 8],
    widths=0.18,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color='black', linewidth=1.5),
    whiskerprops=dict(color='black', linewidth=1),
    capprops=dict(color='black', linewidth=1),
    boxprops=dict(edgecolor='black', linewidth=1, facecolor='white', alpha=0.7)
)
for element in ['boxes', 'whiskers', 'caps', 'medians']:
    for artist in box[element]:
        artist.set_zorder(3)
ax.set_ylabel('Stomata aperture')
ax.set_title('Stomata aperture in response to temperature')
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
ax.set_xticklabels(['Wt 21°', 'Wt 42°', 'ost1 21°', 'ost1 42°', 'no UBP24 21°', 'no UBP24 42°', 'no TOT3i 21°', 'no TOT3i 42°'], rotation=20)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('output/Wt-ost1-variants-violin.svg', dpi=300, bbox_inches='tight')
plt.savefig('output/Wt-ost1-variants-violin.png', dpi=200, bbox_inches='tight')
#plt.show()

#### Plot 2 - Wt stomata aperture response to different simulated temperatures

fig, ax = plt.subplots(figsize=(8, 5))
violin_data = [
    stomata_temperatures[0, :],
    stomata_temperatures[1, :],
    stomata_temperatures[2, :],
    stomata_temperatures[3, :]
]
parts = ax.violinplot(violin_data, positions=[1, 2, 3, 4], widths=0.7, showmeans=False, showextrema=False)
for body, color in zip(parts['bodies'], ['darkblue', 'beige', 'darkorange', 'brown']):
    body.set_facecolor(color)
    body.set_edgecolor('black')
    body.set_alpha(0.8)
box = ax.boxplot(
    violin_data,
    positions=[1, 2, 3, 4],
    widths=0.18,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color='black', linewidth=1.5),
    whiskerprops=dict(color='black', linewidth=1),
    capprops=dict(color='black', linewidth=1),
    boxprops=dict(edgecolor='black', linewidth=1, facecolor='white', alpha=0.7)
)
for element in ['boxes', 'whiskers', 'caps', 'medians']:
    for artist in box[element]:
        artist.set_zorder(3)
ax.set_ylabel('Stomata aperture')
ax.set_title('Stomata aperture vs Temperature')
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels([tempnames[i] for i in plottingvalues])
ax.set_xlabel('Temperature (°C)')
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('output/TemperaturexStomata-violin.svg', dpi=300, bbox_inches='tight')
plt.savefig('output/TemperaturexStomata-violin.png', dpi=200, bbox_inches='tight')
#plt.show()

#### Plot 3 - Stomata aperture in single and combined Heat x ABA treatments for Wt and mutants
fig, ax = plt.subplots(figsize=(12, 5))
violin_data = [
    FinalvaluesHxD[0, :], FinalvaluesHxD[1, :], FinalvaluesHxD[2, :], FinalvaluesHxD[3, :],
    FinalvaluesHxD[4, :], FinalvaluesHxD[5, :], FinalvaluesHxD[6, :], FinalvaluesHxD[7, :],
    FinalvaluesHxD[8, :], FinalvaluesHxD[9, :], FinalvaluesHxD[10, :], FinalvaluesHxD[11, :]
]
parts = ax.violinplot(violin_data, positions=np.arange(1, 13), widths=0.7, showmeans=False, showextrema=False)
for body, color in zip(parts['bodies'], ["#000000", '#A52A2A', "#0326C4", "#4C6E10"] * 3):
    body.set_facecolor(color)
    body.set_edgecolor('black')
    body.set_alpha(0.8)
box = ax.boxplot(
    violin_data,
    positions=np.arange(1, 13),
    widths=0.18,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color='black', linewidth=1.5),
    whiskerprops=dict(color='black', linewidth=1),
    capprops=dict(color='black', linewidth=1),
    boxprops=dict(edgecolor='black', linewidth=1, facecolor='white', alpha=0.7)
)
for element in ['boxes', 'whiskers', 'caps', 'medians']:
    for artist in box[element]:
        artist.set_zorder(3)
ax.set_ylabel('Stomata aperture')
ax.set_title('Stomata aperture - HxD conditions')
ax.set_xticks(np.arange(1, 13))
ax.set_xticklabels([
    'Wt C', 'Wt H', 'Wt ABA', 'Wt HxABA',
    'ost1 C', 'ost1 H', 'ost1 ABA', 'ost1 HxABA',
    'slac1 C', 'slac1 H', 'slac1 ABA', 'slac1 HxABA'
], rotation=25)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('output/HxD-Finalvalues-violin.svg', dpi=300, bbox_inches='tight')
plt.savefig('output/HxD-Finalvalues-violin.png', dpi=200, bbox_inches='tight')
#plt.show()

print(f"Running time: {time.time() - start_time:.2f} seconds")

