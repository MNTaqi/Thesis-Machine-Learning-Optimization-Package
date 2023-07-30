#NSGA II Problems Point
PYTHONHASHSEED = 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from platypus.algorithms import NSGAII
from platypus.core import Problem
from platypus.types import Real, Integer
from platypus.operators import CompoundOperator, SBX, PM, HUX, BitFlip
import os
from sklearn.preprocessing import MinMaxScaler

#Direktori file  
os.chdir(r"C:\Users\Taqi\OneDrive - Institut Teknologi Bandung\Desktop\Kuliah\S7\TA\.Phyton\ANN\.TrialNew")
#Model untuk evaluasi import
data = pd.read_excel("ANN_Data_Numeric.xlsx",sheet_name="ANN_Data")

X = data.iloc[0:,0:6].values
min_max_scaler=MinMaxScaler()
Y = min_max_scaler.fit_transform(data.iloc[0:,6].values.reshape(-1,1))

SEA_base = 3974.82952
Baseline = np.array([SEA_base])
Baseline_transformed = min_max_scaler.transform(Baseline.reshape(-1,1))

i = 0
#Mengimport model latih ke NSGA-II
regressor_biasa = load_model("model_EA.h5")
#from keras.utils.layer_utils import print_summary
#print_summary(regressor_biasa)

#Percobaan prediksi model 
#coba = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2.35, 1]).reshape(-1,15)
#print(coba)
#print(regressor_biasa.predict((coba)))


def Opt (vars):
#Variable Definition
    #s1 = vars[0]
    s2 = vars[0]
    s3 = vars[1]
    s4 = vars[2]
    s5 = vars[3]
    s6 = vars[4]
    #met1 = vars[6]
    met2 = vars[5]
    met3 = vars[6]
    met4 = vars[7]
    met5 = vars[8]
    met6 = vars[9]
    #com1 = vars[12]
    com2 = vars[10]
    com3 = vars[11]
    com4 = vars[12]
    Met_Thi = vars[13]
    Comp_Lay = vars[14]
#Input Variable Matrix Definition
    IntVar = np.array([[s2, s3, s4, s5, s6, met2, met3, met4, met5, met6, com2, com3, com4, Met_Thi, Comp_Lay]])
    IntVar = IntVar.reshape(-1,15)
    print ("IntVar shape = "+str(IntVar.shape))
# Function Definition
    SEA = regressor_biasa.predict(IntVar)[0][0]
    print("Problem ke-"+str(i)+", SEA Shape = "+str(SEA.shape))
    print(SEA)
    print(vars)
    return [SEA], [SEA, s2, s3, s4, s5, s6, met2, met3, met4, met5, met6, com2, com3, com4, Met_Thi, Comp_Lay]

itg1=Integer(0,1)
variable_types= [itg1, itg1, itg1, itg1, itg1, itg1, itg1, itg1, itg1, itg1, itg1, itg1, itg1, Real(0.5,2.5), Real(1,15)]

#Case37: Susunan 2 Domex Protect 500 CNC
problem37 = Problem(15, 1, 16)
problem37.types[:] = variable_types
problem37.constraints[0] = ">=0" #SEA
problem37.constraints[1] = "==1" #Susunan 2
problem37.constraints[2] = "==0" #Susunan 3
problem37.constraints[3] = "==0" #Susunan 4
problem37.constraints[4] = "==0" #Susunan 5
problem37.constraints[5] = "==0" #Susunan 6
problem37.constraints[6] = "==0" #Weldox 700E
problem37.constraints[7] = "==0" #Hardox 400
problem37.constraints[8] = "==1" #Domex Protect 500
problem37.constraints[9] = "==0" #Armox 560T
problem37.constraints[10] = "==0" #Al7075-T651
problem37.constraints[11] = "==0" #Kevlar-29
problem37.constraints[12] = "==0" #S2 Glass/SC15
problem37.constraints[13] = "==0" #CFRP
problem37.constraints[14] = ">=0.5" #thickness
problem37.constraints[15] = ">=1" #lapisan
problem37.directions[0] = Problem.MAXIMIZE
problem37.function = Opt

#Case38: Susunan 2 Domex Protect 500 Kevlar-29
problem38 = Problem(15, 1, 16)
problem38.types[:] = variable_types
problem38.constraints[0] = ">=0" #SEA
problem38.constraints[1] = "==1" #Susunan 2
problem38.constraints[2] = "==0" #Susunan 3
problem38.constraints[3] = "==0" #Susunan 4
problem38.constraints[4] = "==0" #Susunan 5
problem38.constraints[5] = "==0" #Susunan 6
problem38.constraints[6] = "==0" #Weldox 700E
problem38.constraints[7] = "==0" #Hardox 400
problem38.constraints[8] = "==1" #Domex Protect 500
problem38.constraints[9] = "==0" #Armox 560T
problem38.constraints[10] = "==0" #Al7075-T651
problem38.constraints[11] = "==1" #Kevlar-29
problem38.constraints[12] = "==0" #S2 Glass/SC15
problem38.constraints[13] = "==0" #CFRP
problem38.constraints[14] = ">=0.5" #thickness
problem38.constraints[15] = ">=1" #lapisan
problem38.directions[0] = Problem.MAXIMIZE
problem38.function = Opt

#Case39: Susunan 2 Domex Protect 500 S2Glass/SC15
problem39 = Problem(15, 1, 16)
problem39.types[:] = variable_types
problem39.constraints[0] = ">=0" #SEA
problem39.constraints[1] = "==1" #Susunan 2
problem39.constraints[2] = "==0" #Susunan 3
problem39.constraints[3] = "==0" #Susunan 4
problem39.constraints[4] = "==0" #Susunan 5
problem39.constraints[5] = "==0" #Susunan 6
problem39.constraints[6] = "==0" #Weldox 700E
problem39.constraints[7] = "==0" #Hardox 400
problem39.constraints[8] = "==1" #Domex Protect 500
problem39.constraints[9] = "==0" #Armox 560T
problem39.constraints[10] = "==0" #Al7075-T651
problem39.constraints[11] = "==0" #Kevlar-29
problem39.constraints[12] = "==1" #S2 Glass/SC15
problem39.constraints[13] = "==0" #CFRP
problem39.constraints[14] = ">=0.5" #thickness
problem39.constraints[15] = ">=1" #lapisan
problem39.directions[0] = Problem.MAXIMIZE
problem39.function = Opt

#Case40: Susunan 2 Domex Protect 500 CFRP
problem40 = Problem(15, 1, 16)
problem40.types[:] = variable_types
problem40.constraints[0] = ">=0" #SEA
problem40.constraints[1] = "==1" #Susunan 2
problem40.constraints[2] = "==0" #Susunan 3
problem40.constraints[3] = "==0" #Susunan 4
problem40.constraints[4] = "==0" #Susunan 5
problem40.constraints[5] = "==0" #Susunan 6
problem40.constraints[6] = "==0" #Weldox 700E
problem40.constraints[7] = "==0" #Hardox 400
problem40.constraints[8] = "==1" #Domex Protect 500
problem40.constraints[9] = "==0" #Armox 560T
problem40.constraints[10] = "==0" #Al7075-T651
problem40.constraints[11] = "==0" #Kevlar-29
problem40.constraints[12] = "==0" #S2 Glass/SC15
problem40.constraints[13] = "==1" #CFRP
problem40.constraints[14] = ">=0.5" #thickness
problem40.constraints[15] = ">=1" #lapisan
problem40.directions[0] = Problem.MAXIMIZE
problem40.function = Opt

#Case41: Susunan 2 Armox 560T CNC
problem41 = Problem(15, 1, 16)
problem41.types[:] = variable_types
problem41.constraints[0] = ">=0" #SEA
problem41.constraints[1] = "==1" #Susunan 2
problem41.constraints[2] = "==0" #Susunan 3
problem41.constraints[3] = "==0" #Susunan 4
problem41.constraints[4] = "==0" #Susunan 5
problem41.constraints[5] = "==0" #Susunan 6
problem41.constraints[6] = "==0" #Weldox 700E
problem41.constraints[7] = "==0" #Hardox 400
problem41.constraints[8] = "==0" #Domex Protect 500
problem41.constraints[9] = "==1" #Armox 560T
problem41.constraints[10] = "==0" #Al7075-T651
problem41.constraints[11] = "==0" #Kevlar-29
problem41.constraints[12] = "==0" #S2 Glass/SC15
problem41.constraints[13] = "==0" #CFRP
problem41.constraints[14] = ">=0.5" #thickness
problem41.constraints[15] = ">=1" #lapisan
problem41.directions[0] = Problem.MAXIMIZE
problem41.function = Opt

#Case42: Susunan 2 Armox 560T Kevlar-29
problem42 = Problem(15, 1, 16)
problem42.types[:] = variable_types
problem42.constraints[0] = ">=0" #SEA
problem42.constraints[1] = "==1" #Susunan 2
problem42.constraints[2] = "==0" #Susunan 3
problem42.constraints[3] = "==0" #Susunan 4
problem42.constraints[4] = "==0" #Susunan 5
problem42.constraints[5] = "==0" #Susunan 6
problem42.constraints[6] = "==0" #Weldox 700E
problem42.constraints[7] = "==0" #Hardox 400
problem42.constraints[8] = "==0" #Domex Protect 500
problem42.constraints[9] = "==1" #Armox 560T
problem42.constraints[10] = "==0" #Al7075-T651
problem42.constraints[11] = "==1" #Kevlar-29
problem42.constraints[12] = "==0" #S2 Glass/SC15
problem42.constraints[13] = "==0" #CFRP
problem42.constraints[14] = ">=0.5" #thickness
problem42.constraints[15] = ">=1" #lapisan
problem42.directions[0] = Problem.MAXIMIZE
problem42.function = Opt

#Case43: Susunan 2 Armox 560T S2Glass/SC15
problem43 = Problem(15, 1, 16)
problem43.types[:] = variable_types
problem43.constraints[0] = ">=0" #SEA
problem43.constraints[1] = "==1" #Susunan 2
problem43.constraints[2] = "==0" #Susunan 3
problem43.constraints[3] = "==0" #Susunan 4
problem43.constraints[4] = "==0" #Susunan 5
problem43.constraints[5] = "==0" #Susunan 6
problem43.constraints[6] = "==0" #Weldox 700E
problem43.constraints[7] = "==0" #Hardox 400
problem43.constraints[8] = "==0" #Domex Protect 500
problem43.constraints[9] = "==1" #Armox 560T
problem43.constraints[10] = "==0" #Al7075-T651
problem43.constraints[11] = "==0" #Kevlar-29
problem43.constraints[12] = "==1" #S2 Glass/SC15
problem43.constraints[13] = "==0" #CFRP
problem43.constraints[14] = ">=0.5" #thickness
problem43.constraints[15] = ">=1" #lapisan
problem43.directions[0] = Problem.MAXIMIZE
problem43.function = Opt

#Case44: Susunan 2 Armox 560T CFRP
problem44 = Problem(15, 1, 16)
problem44.types[:] = variable_types
problem44.constraints[0] = ">=0" #SEA
problem44.constraints[1] = "==1" #Susunan 2
problem44.constraints[2] = "==0" #Susunan 3
problem44.constraints[3] = "==0" #Susunan 4
problem44.constraints[4] = "==0" #Susunan 5
problem44.constraints[5] = "==0" #Susunan 6
problem44.constraints[6] = "==0" #Weldox 700E
problem44.constraints[7] = "==0" #Hardox 400
problem44.constraints[8] = "==0" #Domex Protect 500
problem44.constraints[9] = "==1" #Armox 560T
problem44.constraints[10] = "==0" #Al7075-T651
problem44.constraints[11] = "==0" #Kevlar-29
problem44.constraints[12] = "==0" #S2 Glass/SC15
problem44.constraints[13] = "==1" #CFRP
problem44.constraints[14] = ">=0.5" #thickness
problem44.constraints[15] = ">=1" #lapisan
problem44.directions[0] = Problem.MAXIMIZE
problem44.function = Opt

#Case45: Susunan 2 Al7075-T651 CNC
problem45 = Problem(15, 1, 16)
problem45.types[:] = variable_types
problem45.constraints[0] = ">=0" #SEA
problem45.constraints[1] = "==1" #Susunan 2
problem45.constraints[2] = "==0" #Susunan 3
problem45.constraints[3] = "==0" #Susunan 4
problem45.constraints[4] = "==0" #Susunan 5
problem45.constraints[5] = "==0" #Susunan 6
problem45.constraints[6] = "==0" #Weldox 700E
problem45.constraints[7] = "==0" #Hardox 400
problem45.constraints[8] = "==0" #Domex Protect 500
problem45.constraints[9] = "==0" #Armox 560T
problem45.constraints[10] = "==1" #Al7075-T651
problem45.constraints[11] = "==0" #Kevlar-29
problem45.constraints[12] = "==0" #S2 Glass/SC15
problem45.constraints[13] = "==0" #CFRP
problem45.constraints[14] = ">=0.5" #thickness
problem45.constraints[15] = ">=1" #lapisan
problem45.directions[0] = Problem.MAXIMIZE
problem45.function = Opt

#Case46: Susunan 2 Al7075-T651 Kevlar-29
problem46 = Problem(15, 1, 16)
problem46.types[:] = variable_types
problem46.constraints[0] = ">=0" #SEA
problem46.constraints[1] = "==1" #Susunan 2
problem46.constraints[2] = "==0" #Susunan 3
problem46.constraints[3] = "==0" #Susunan 4
problem46.constraints[4] = "==0" #Susunan 5
problem46.constraints[5] = "==0" #Susunan 6
problem46.constraints[6] = "==0" #Weldox 700E
problem46.constraints[7] = "==0" #Hardox 400
problem46.constraints[8] = "==0" #Domex Protect 500
problem46.constraints[9] = "==0" #Armox 560T
problem46.constraints[10] = "==1" #Al7075-T651
problem46.constraints[11] = "==1" #Kevlar-29
problem46.constraints[12] = "==0" #S2 Glass/SC15
problem46.constraints[13] = "==0" #CFRP
problem46.constraints[14] = ">=0.5" #thickness
problem46.constraints[15] = ">=1" #lapisan
problem46.directions[0] = Problem.MAXIMIZE
problem46.function = Opt

#Case47: Susunan 2 Al7075-T651 S2Glass/SC15
problem47 = Problem(15, 1, 16)
problem47.types[:] = variable_types
problem47.constraints[0] = ">=0" #SEA
problem47.constraints[1] = "==1" #Susunan 2
problem47.constraints[2] = "==0" #Susunan 3
problem47.constraints[3] = "==0" #Susunan 4
problem47.constraints[4] = "==0" #Susunan 5
problem47.constraints[5] = "==0" #Susunan 6
problem47.constraints[6] = "==0" #Weldox 700E
problem47.constraints[7] = "==0" #Hardox 400
problem47.constraints[8] = "==0" #Domex Protect 500
problem47.constraints[9] = "==0" #Armox 560T
problem47.constraints[10] = "==1" #Al7075-T651
problem47.constraints[11] = "==0" #Kevlar-29
problem47.constraints[12] = "==1" #S2 Glass/SC15
problem47.constraints[13] = "==0" #CFRP
problem47.constraints[14] = ">=0.5" #thickness
problem47.constraints[15] = ">=1" #lapisan
problem47.directions[0] = Problem.MAXIMIZE
problem47.function = Opt

#Case48: Susunan 2 Al7075-T651 CFRP
problem48 = Problem(15, 1, 16)
problem48.types[:] = variable_types
problem48.constraints[0] = ">=0" #SEA
problem48.constraints[1] = "==1" #Susunan 2
problem48.constraints[2] = "==0" #Susunan 3
problem48.constraints[3] = "==0" #Susunan 4
problem48.constraints[4] = "==0" #Susunan 5
problem48.constraints[5] = "==0" #Susunan 6
problem48.constraints[6] = "==0" #Weldox 700E
problem48.constraints[7] = "==0" #Hardox 400
problem48.constraints[8] = "==0" #Domex Protect 500
problem48.constraints[9] = "==0" #Armox 560T
problem48.constraints[10] = "==1" #Al7075-T651
problem48.constraints[11] = "==0" #Kevlar-29
problem48.constraints[12] = "==0" #S2 Glass/SC15
problem48.constraints[13] = "==1" #CFRP
problem48.constraints[14] = ">=0.5" #thickness
problem48.constraints[15] = ">=1" #lapisan
problem48.directions[0] = Problem.MAXIMIZE
problem48.function = Opt

#Case49: Susunan 3 Weldox 500E CNC
problem49 = Problem(15, 1, 16)
problem49.types[:] = variable_types
problem49.constraints[0] = ">=0" #SEA
problem49.constraints[1] = "==0" #Susunan 2
problem49.constraints[2] = "==1" #Susunan 3
problem49.constraints[3] = "==0" #Susunan 4
problem49.constraints[4] = "==0" #Susunan 5
problem49.constraints[5] = "==0" #Susunan 6
problem49.constraints[6] = "==0" #Weldox 700E
problem49.constraints[7] = "==0" #Hardox 400
problem49.constraints[8] = "==0" #Domex Protect 500
problem49.constraints[9] = "==0" #Armox 560T
problem49.constraints[10] = "==0" #Al7075-T651
problem49.constraints[11] = "==0" #Kevlar-29
problem49.constraints[12] = "==0" #S2 Glass/SC15
problem49.constraints[13] = "==0" #CFRP
problem49.constraints[14] = ">=0.5" #thickness
problem49.constraints[15] = ">=1" #lapisan
problem49.directions[0] = Problem.MAXIMIZE
problem49.function = Opt

#Case50: Susunan 3 Weldox 500E Kevlar-29
problem50 = Problem(15, 1, 16)
problem50.types[:] = variable_types
problem50.constraints[0] = ">=0" #SEA
problem50.constraints[1] = "==0" #Susunan 2
problem50.constraints[2] = "==1" #Susunan 3
problem50.constraints[3] = "==0" #Susunan 4
problem50.constraints[4] = "==0" #Susunan 5
problem50.constraints[5] = "==0" #Susunan 6
problem50.constraints[6] = "==0" #Weldox 700E
problem50.constraints[7] = "==0" #Hardox 400
problem50.constraints[8] = "==0" #Domex Protect 500
problem50.constraints[9] = "==0" #Armox 560T
problem50.constraints[10] = "==0" #Al7075-T651
problem50.constraints[11] = "==1" #Kevlar-29
problem50.constraints[12] = "==0" #S2 Glass/SC15
problem50.constraints[13] = "==0" #CFRP
problem50.constraints[14] = ">=0.5" #thickness
problem50.constraints[15] = ">=1" #lapisan
problem50.directions[0] = Problem.MAXIMIZE
problem50.function = Opt

#Case51: Susunan 3 Weldox 500E S2Glass/SC15
problem51 = Problem(15, 1, 16)
problem51.types[:] = variable_types
problem51.constraints[0] = ">=0" #SEA
problem51.constraints[1] = "==0" #Susunan 2
problem51.constraints[2] = "==1" #Susunan 3
problem51.constraints[3] = "==0" #Susunan 4
problem51.constraints[4] = "==0" #Susunan 5
problem51.constraints[5] = "==0" #Susunan 6
problem51.constraints[6] = "==0" #Weldox 700E
problem51.constraints[7] = "==0" #Hardox 400
problem51.constraints[8] = "==0" #Domex Protect 500
problem51.constraints[9] = "==0" #Armox 560T
problem51.constraints[10] = "==0" #Al7075-T651
problem51.constraints[11] = "==0" #Kevlar-29
problem51.constraints[12] = "==1" #S2 Glass/SC15
problem51.constraints[13] = "==0" #CFRP
problem51.constraints[14] = ">=0.5" #thickness
problem51.constraints[15] = ">=1" #lapisan
problem51.directions[0] = Problem.MAXIMIZE
problem51.function = Opt

#Case52: Susunan 3 Weldox 500E CFRP
problem52 = Problem(15, 1, 16)
problem52.types[:] = variable_types
problem52.constraints[0] = ">=0" #SEA
problem52.constraints[1] = "==0" #Susunan 2
problem52.constraints[2] = "==1" #Susunan 3
problem52.constraints[3] = "==0" #Susunan 4
problem52.constraints[4] = "==0" #Susunan 5
problem52.constraints[5] = "==0" #Susunan 6
problem52.constraints[6] = "==0" #Weldox 700E
problem52.constraints[7] = "==0" #Hardox 400
problem52.constraints[8] = "==0" #Domex Protect 500
problem52.constraints[9] = "==0" #Armox 560T
problem52.constraints[10] = "==0" #Al7075-T651
problem52.constraints[11] = "==0" #Kevlar-29
problem52.constraints[12] = "==0" #S2 Glass/SC15
problem52.constraints[13] = "==1" #CFRP
problem52.constraints[14] = ">=0.5" #thickness
problem52.constraints[15] = ">=1" #lapisan
problem52.directions[0] = Problem.MAXIMIZE
problem52.function = Opt

#Case53: Susunan 3 Weldox 700E CNC
problem53 = Problem(15, 1, 16)
problem53.types[:] = variable_types
problem53.constraints[0] = ">=0" #SEA
problem53.constraints[1] = "==0" #Susunan 2
problem53.constraints[2] = "==1" #Susunan 3
problem53.constraints[3] = "==0" #Susunan 4
problem53.constraints[4] = "==0" #Susunan 5
problem53.constraints[5] = "==0" #Susunan 6
problem53.constraints[6] = "==1" #Weldox 700E
problem53.constraints[7] = "==0" #Hardox 400
problem53.constraints[8] = "==0" #Domex Protect 500
problem53.constraints[9] = "==0" #Armox 560T
problem53.constraints[10] = "==0" #Al7075-T651
problem53.constraints[11] = "==0" #Kevlar-29
problem53.constraints[12] = "==0" #S2 Glass/SC15
problem53.constraints[13] = "==0" #CFRP
problem53.constraints[14] = ">=0.5" #thickness
problem53.constraints[15] = ">=1" #lapisan
problem53.directions[0] = Problem.MAXIMIZE
problem53.function = Opt

#Case54: Susunan 3 Weldox 700E Kevlar-29
problem54 = Problem(15, 1, 16)
problem54.types[:] = variable_types
problem54.constraints[0] = ">=0" #SEA
problem54.constraints[1] = "==0" #Susunan 2
problem54.constraints[2] = "==1" #Susunan 3
problem54.constraints[3] = "==0" #Susunan 4
problem54.constraints[4] = "==0" #Susunan 5
problem54.constraints[5] = "==0" #Susunan 6
problem54.constraints[6] = "==1" #Weldox 700E
problem54.constraints[7] = "==0" #Hardox 400
problem54.constraints[8] = "==0" #Domex Protect 500
problem54.constraints[9] = "==0" #Armox 560T
problem54.constraints[10] = "==0" #Al7075-T651
problem54.constraints[11] = "==1" #Kevlar-29
problem54.constraints[12] = "==0" #S2 Glass/SC15
problem54.constraints[13] = "==0" #CFRP
problem54.constraints[14] = ">=0.5" #thickness
problem54.constraints[15] = ">=1" #lapisan
problem54.directions[0] = Problem.MAXIMIZE
problem54.function = Opt

#Case55: Susunan 3 Weldox 700E S2Glass/SC15
problem55 = Problem(15, 1, 16)
problem55.types[:] = variable_types
problem55.constraints[0] = ">=0" #SEA
problem55.constraints[1] = "==0" #Susunan 2
problem55.constraints[2] = "==1" #Susunan 3
problem55.constraints[3] = "==0" #Susunan 4
problem55.constraints[4] = "==0" #Susunan 5
problem55.constraints[5] = "==0" #Susunan 6
problem55.constraints[6] = "==1" #Weldox 700E
problem55.constraints[7] = "==0" #Hardox 400
problem55.constraints[8] = "==0" #Domex Protect 500
problem55.constraints[9] = "==0" #Armox 560T
problem55.constraints[10] = "==0" #Al7075-T651
problem55.constraints[11] = "==0" #Kevlar-29
problem55.constraints[12] = "==1" #S2 Glass/SC15
problem55.constraints[13] = "==0" #CFRP
problem55.constraints[14] = ">=0.5" #thickness
problem55.constraints[15] = ">=1" #lapisan
problem55.directions[0] = Problem.MAXIMIZE
problem55.function = Opt

#Case56: Susunan 3 Weldox 700E CFRP
problem56 = Problem(15, 1, 16)
problem56.types[:] = variable_types
problem56.constraints[0] = ">=0" #SEA
problem56.constraints[1] = "==0" #Susunan 2
problem56.constraints[2] = "==1" #Susunan 3
problem56.constraints[3] = "==0" #Susunan 4
problem56.constraints[4] = "==0" #Susunan 5
problem56.constraints[5] = "==0" #Susunan 6
problem56.constraints[6] = "==1" #Weldox 700E
problem56.constraints[7] = "==0" #Hardox 400
problem56.constraints[8] = "==0" #Domex Protect 500
problem56.constraints[9] = "==0" #Armox 560T
problem56.constraints[10] = "==0" #Al7075-T651
problem56.constraints[11] = "==0" #Kevlar-29
problem56.constraints[12] = "==0" #S2 Glass/SC15
problem56.constraints[13] = "==1" #CFRP
problem56.constraints[14] = ">=0.5" #thickness
problem56.constraints[15] = ">=1" #lapisan
problem56.directions[0] = Problem.MAXIMIZE
problem56.function = Opt

#Case57: Susunan 3 Hardox 400 CNC
problem57 = Problem(15, 1, 16)
problem57.types[:] = variable_types
problem57.constraints[0] = ">=0" #SEA
problem57.constraints[1] = "==0" #Susunan 2
problem57.constraints[2] = "==1" #Susunan 3
problem57.constraints[3] = "==0" #Susunan 4
problem57.constraints[4] = "==0" #Susunan 5
problem57.constraints[5] = "==0" #Susunan 6
problem57.constraints[6] = "==0" #Weldox 700E
problem57.constraints[7] = "==1" #Hardox 400
problem57.constraints[8] = "==0" #Domex Protect 500
problem57.constraints[9] = "==0" #Armox 560T
problem57.constraints[10] = "==0" #Al7075-T651
problem57.constraints[11] = "==0" #Kevlar-29
problem57.constraints[12] = "==0" #S2 Glass/SC15
problem57.constraints[13] = "==0" #CFRP
problem57.constraints[14] = ">=0.5" #thickness
problem57.constraints[15] = ">=1" #lapisan
problem57.directions[0] = Problem.MAXIMIZE
problem57.function = Opt

#Case58: Susunan 3 Hardox 400 Kevlar-29
problem58 = Problem(15, 1, 16)
problem58.types[:] = variable_types
problem58.constraints[0] = ">=0" #SEA
problem58.constraints[1] = "==0" #Susunan 2
problem58.constraints[2] = "==1" #Susunan 3
problem58.constraints[3] = "==0" #Susunan 4
problem58.constraints[4] = "==0" #Susunan 5
problem58.constraints[5] = "==0" #Susunan 6
problem58.constraints[6] = "==0" #Weldox 700E
problem58.constraints[7] = "==1" #Hardox 400
problem58.constraints[8] = "==0" #Domex Protect 500
problem58.constraints[9] = "==0" #Armox 560T
problem58.constraints[10] = "==0" #Al7075-T651
problem58.constraints[11] = "==1" #Kevlar-29
problem58.constraints[12] = "==0" #S2 Glass/SC15
problem58.constraints[13] = "==0" #CFRP
problem58.constraints[14] = ">=0.5" #thickness
problem58.constraints[15] = ">=1" #lapisan
problem58.directions[0] = Problem.MAXIMIZE
problem58.function = Opt

#Case59: Susunan 3 Hardox 400 S2Glass/SC15
problem59 = Problem(15, 1, 16)
problem59.types[:] = variable_types
problem59.constraints[0] = ">=0" #SEA
problem59.constraints[1] = "==0" #Susunan 2
problem59.constraints[2] = "==1" #Susunan 3
problem59.constraints[3] = "==0" #Susunan 4
problem59.constraints[4] = "==0" #Susunan 5
problem59.constraints[5] = "==0" #Susunan 6
problem59.constraints[6] = "==0" #Weldox 700E
problem59.constraints[7] = "==1" #Hardox 400
problem59.constraints[8] = "==0" #Domex Protect 500
problem59.constraints[9] = "==0" #Armox 560T
problem59.constraints[10] = "==0" #Al7075-T651
problem59.constraints[11] = "==0" #Kevlar-29
problem59.constraints[12] = "==1" #S2 Glass/SC15
problem59.constraints[13] = "==0" #CFRP
problem59.constraints[14] = ">=0.5" #thickness
problem59.constraints[15] = ">=1" #lapisan
problem59.directions[0] = Problem.MAXIMIZE
problem59.function = Opt

#Case60: Susunan 3 Hardox 400 CFRP
problem60 = Problem(15, 1, 16)
problem60.types[:] = variable_types
problem60.constraints[0] = ">=0" #SEA
problem60.constraints[1] = "==0" #Susunan 2
problem60.constraints[2] = "==1" #Susunan 3
problem60.constraints[3] = "==0" #Susunan 4
problem60.constraints[4] = "==0" #Susunan 5
problem60.constraints[5] = "==0" #Susunan 6
problem60.constraints[6] = "==0" #Weldox 700E
problem60.constraints[7] = "==1" #Hardox 400
problem60.constraints[8] = "==0" #Domex Protect 500
problem60.constraints[9] = "==0" #Armox 560T
problem60.constraints[10] = "==0" #Al7075-T651
problem60.constraints[11] = "==0" #Kevlar-29
problem60.constraints[12] = "==0" #S2 Glass/SC15
problem60.constraints[13] = "==1" #CFRP
problem60.constraints[14] = ">=0.5" #thickness
problem60.constraints[15] = ">=1" #lapisan
problem60.directions[0] = Problem.MAXIMIZE
problem60.function = Opt

#Case61: Susunan 3 Domex Protect 500 CNC
problem61 = Problem(15, 1, 16)
problem61.types[:] = variable_types
problem61.constraints[0] = ">=0" #SEA
problem61.constraints[1] = "==0" #Susunan 2
problem61.constraints[2] = "==1" #Susunan 3
problem61.constraints[3] = "==0" #Susunan 4
problem61.constraints[4] = "==0" #Susunan 5
problem61.constraints[5] = "==0" #Susunan 6
problem61.constraints[6] = "==0" #Weldox 700E
problem61.constraints[7] = "==0" #Hardox 400
problem61.constraints[8] = "==1" #Domex Protect 500
problem61.constraints[9] = "==0" #Armox 560T
problem61.constraints[10] = "==0" #Al7075-T651
problem61.constraints[11] = "==0" #Kevlar-29
problem61.constraints[12] = "==0" #S2 Glass/SC15
problem61.constraints[13] = "==0" #CFRP
problem61.constraints[14] = ">=0.5" #thickness
problem61.constraints[15] = ">=1" #lapisan
problem61.directions[0] = Problem.MAXIMIZE
problem61.function = Opt

#Case62: Susunan 3 Domex Protect 500 Kevlar-29
problem62 = Problem(15, 1, 16)
problem62.types[:] = variable_types
problem62.constraints[0] = ">=0" #SEA
problem62.constraints[1] = "==0" #Susunan 2
problem62.constraints[2] = "==1" #Susunan 3
problem62.constraints[3] = "==0" #Susunan 4
problem62.constraints[4] = "==0" #Susunan 5
problem62.constraints[5] = "==0" #Susunan 6
problem62.constraints[6] = "==0" #Weldox 700E
problem62.constraints[7] = "==0" #Hardox 400
problem62.constraints[8] = "==1" #Domex Protect 500
problem62.constraints[9] = "==0" #Armox 560T
problem62.constraints[10] = "==0" #Al7075-T651
problem62.constraints[11] = "==1" #Kevlar-29
problem62.constraints[12] = "==0" #S2 Glass/SC15
problem62.constraints[13] = "==0" #CFRP
problem62.constraints[14] = ">=0.5" #thickness
problem62.constraints[15] = ">=1" #lapisan
problem62.directions[0] = Problem.MAXIMIZE
problem62.function = Opt

#Case63: Susunan 3 Domex Protect 500 S2Glass/SC15
problem63 = Problem(15, 1, 16)
problem63.types[:] = variable_types
problem63.constraints[0] = ">=0" #SEA
problem63.constraints[1] = "==0" #Susunan 2
problem63.constraints[2] = "==1" #Susunan 3
problem63.constraints[3] = "==0" #Susunan 4
problem63.constraints[4] = "==0" #Susunan 5
problem63.constraints[5] = "==0" #Susunan 6
problem63.constraints[6] = "==0" #Weldox 700E
problem63.constraints[7] = "==0" #Hardox 400
problem63.constraints[8] = "==1" #Domex Protect 500
problem63.constraints[9] = "==0" #Armox 560T
problem63.constraints[10] = "==0" #Al7075-T651
problem63.constraints[11] = "==0" #Kevlar-29
problem63.constraints[12] = "==1" #S2 Glass/SC15
problem63.constraints[13] = "==0" #CFRP
problem63.constraints[14] = ">=0.5" #thickness
problem63.constraints[15] = ">=1" #lapisan
problem63.directions[0] = Problem.MAXIMIZE
problem63.function = Opt

#Case64: Susunan 3 Domex Protect 500 CFRP
problem64 = Problem(15, 1, 16)
problem64.types[:] = variable_types
problem64.constraints[0] = ">=0" #SEA
problem64.constraints[1] = "==0" #Susunan 2
problem64.constraints[2] = "==1" #Susunan 3
problem64.constraints[3] = "==0" #Susunan 4
problem64.constraints[4] = "==0" #Susunan 5
problem64.constraints[5] = "==0" #Susunan 6
problem64.constraints[6] = "==0" #Weldox 700E
problem64.constraints[7] = "==0" #Hardox 400
problem64.constraints[8] = "==1" #Domex Protect 500
problem64.constraints[9] = "==0" #Armox 560T
problem64.constraints[10] = "==0" #Al7075-T651
problem64.constraints[11] = "==0" #Kevlar-29
problem64.constraints[12] = "==0" #S2 Glass/SC15
problem64.constraints[13] = "==1" #CFRP
problem64.constraints[14] = ">=0.5" #thickness
problem64.constraints[15] = ">=1" #lapisan
problem64.directions[0] = Problem.MAXIMIZE
problem64.function = Opt

#Case65: Susunan 3 Armox 560T CNC
problem65 = Problem(15, 1, 16)
problem65.types[:] = variable_types
problem65.constraints[0] = ">=0" #SEA
problem65.constraints[1] = "==0" #Susunan 2
problem65.constraints[2] = "==1" #Susunan 3
problem65.constraints[3] = "==0" #Susunan 4
problem65.constraints[4] = "==0" #Susunan 5
problem65.constraints[5] = "==0" #Susunan 6
problem65.constraints[6] = "==0" #Weldox 700E
problem65.constraints[7] = "==0" #Hardox 400
problem65.constraints[8] = "==0" #Domex Protect 500
problem65.constraints[9] = "==1" #Armox 560T
problem65.constraints[10] = "==0" #Al7075-T651
problem65.constraints[11] = "==0" #Kevlar-29
problem65.constraints[12] = "==0" #S2 Glass/SC15
problem65.constraints[13] = "==0" #CFRP
problem65.constraints[14] = ">=0.5" #thickness
problem65.constraints[15] = ">=1" #lapisan
problem65.directions[0] = Problem.MAXIMIZE
problem65.function = Opt

#Case66: Susunan 3 Armox 560T Kevlar-29
problem66 = Problem(15, 1, 16)
problem66.types[:] = variable_types
problem66.constraints[0] = ">=0" #SEA
problem66.constraints[1] = "==0" #Susunan 2
problem66.constraints[2] = "==1" #Susunan 3
problem66.constraints[3] = "==0" #Susunan 4
problem66.constraints[4] = "==0" #Susunan 5
problem66.constraints[5] = "==0" #Susunan 6
problem66.constraints[6] = "==0" #Weldox 700E
problem66.constraints[7] = "==0" #Hardox 400
problem66.constraints[8] = "==0" #Domex Protect 500
problem66.constraints[9] = "==1" #Armox 560T
problem66.constraints[10] = "==0" #Al7075-T651
problem66.constraints[11] = "==1" #Kevlar-29
problem66.constraints[12] = "==0" #S2 Glass/SC15
problem66.constraints[13] = "==0" #CFRP
problem66.constraints[14] = ">=0.5" #thickness
problem66.constraints[15] = ">=1" #lapisan
problem66.directions[0] = Problem.MAXIMIZE
problem66.function = Opt

#Case67: Susunan 3 Armox 560T S2Glass/SC15
problem67 = Problem(15, 1, 16)
problem67.types[:] = variable_types
problem67.constraints[0] = ">=0" #SEA
problem67.constraints[1] = "==0" #Susunan 2
problem67.constraints[2] = "==1" #Susunan 3
problem67.constraints[3] = "==0" #Susunan 4
problem67.constraints[4] = "==0" #Susunan 5
problem67.constraints[5] = "==0" #Susunan 6
problem67.constraints[6] = "==0" #Weldox 700E
problem67.constraints[7] = "==0" #Hardox 400
problem67.constraints[8] = "==0" #Domex Protect 500
problem67.constraints[9] = "==1" #Armox 560T
problem67.constraints[10] = "==0" #Al7075-T651
problem67.constraints[11] = "==0" #Kevlar-29
problem67.constraints[12] = "==1" #S2 Glass/SC15
problem67.constraints[13] = "==0" #CFRP
problem67.constraints[14] = ">=0.5" #thickness
problem67.constraints[15] = ">=1" #lapisan
problem67.directions[0] = Problem.MAXIMIZE
problem67.function = Opt

 #Case68: Susunan 3 Armox 560T CFRP
problem68 = Problem(15, 1, 16)
problem68.types[:] = variable_types
problem68.constraints[0] = ">=0" #SEA
problem68.constraints[1] = "==0" #Susunan 2
problem68.constraints[2] = "==1" #Susunan 3
problem68.constraints[3] = "==0" #Susunan 4
problem68.constraints[4] = "==0" #Susunan 5
problem68.constraints[5] = "==0" #Susunan 6
problem68.constraints[6] = "==0" #Weldox 700E
problem68.constraints[7] = "==0" #Hardox 400
problem68.constraints[8] = "==0" #Domex Protect 500
problem68.constraints[9] = "==1" #Armox 560T
problem68.constraints[10] = "==0" #Al7075-T651
problem68.constraints[11] = "==0" #Kevlar-29
problem68.constraints[12] = "==0" #S2 Glass/SC15
problem68.constraints[13] = "==1" #CFRP
problem68.constraints[14] = ">=0.5" #thickness
problem68.constraints[15] = ">=1" #lapisan
problem68.directions[0] = Problem.MAXIMIZE
problem68.function = Opt

#Case69: Susunan 3 Al7075-T651 CNC
problem69 = Problem(15, 1, 16)
problem69.types[:] = variable_types
problem69.constraints[0] = ">=0" #SEA
problem69.constraints[1] = "==0" #Susunan 2
problem69.constraints[2] = "==1" #Susunan 3
problem69.constraints[3] = "==0" #Susunan 4
problem69.constraints[4] = "==0" #Susunan 5
problem69.constraints[5] = "==0" #Susunan 6
problem69.constraints[6] = "==0" #Weldox 700E
problem69.constraints[7] = "==0" #Hardox 400
problem69.constraints[8] = "==0" #Domex Protect 500
problem69.constraints[9] = "==0" #Armox 560T
problem69.constraints[10] = "==1" #Al7075-T651
problem69.constraints[11] = "==0" #Kevlar-29
problem69.constraints[12] = "==0" #S2 Glass/SC15
problem69.constraints[13] = "==0" #CFRP
problem69.constraints[14] = ">=0.5" #thickness
problem69.constraints[15] = ">=1" #lapisan
problem69.directions[0] = Problem.MAXIMIZE
problem69.function = Opt

#Case70: Susunan 3 Al7075-T651 Kevlar-29
problem70 = Problem(15, 1, 16)
problem70.types[:] = variable_types
problem70.constraints[0] = ">=0" #SEA
problem70.constraints[1] = "==0" #Susunan 2
problem70.constraints[2] = "==1" #Susunan 3
problem70.constraints[3] = "==0" #Susunan 4
problem70.constraints[4] = "==0" #Susunan 5
problem70.constraints[5] = "==0" #Susunan 6
problem70.constraints[6] = "==0" #Weldox 700E
problem70.constraints[7] = "==0" #Hardox 400
problem70.constraints[8] = "==0" #Domex Protect 500
problem70.constraints[9] = "==0" #Armox 560T
problem70.constraints[10] = "==1" #Al7075-T651
problem70.constraints[11] = "==1" #Kevlar-29
problem70.constraints[12] = "==0" #S2 Glass/SC15
problem70.constraints[13] = "==0" #CFRP
problem70.constraints[14] = ">=0.5" #thickness
problem70.constraints[15] = ">=1" #lapisan
problem70.directions[0] = Problem.MAXIMIZE
problem70.function = Opt

#Case71: Susunan 3 Al7075-T651 S2Glass/SC15
problem71 = Problem(15, 1, 16)
problem71.types[:] = variable_types
problem71.constraints[0] = ">=0" #SEA
problem71.constraints[1] = "==0" #Susunan 2
problem71.constraints[2] = "==1" #Susunan 3
problem71.constraints[3] = "==0" #Susunan 4
problem71.constraints[4] = "==0" #Susunan 5
problem71.constraints[5] = "==0" #Susunan 6
problem71.constraints[6] = "==0" #Weldox 700E
problem71.constraints[7] = "==0" #Hardox 400
problem71.constraints[8] = "==0" #Domex Protect 500
problem71.constraints[9] = "==0" #Armox 560T
problem71.constraints[10] = "==1" #Al7075-T651
problem71.constraints[11] = "==0" #Kevlar-29
problem71.constraints[12] = "==1" #S2 Glass/SC15
problem71.constraints[13] = "==0" #CFRP
problem71.constraints[14] = ">=0.5" #thickness
problem71.constraints[15] = ">=1" #lapisan
problem71.directions[0] = Problem.MAXIMIZE
problem71.function = Opt

#Case72: Susunan 3 Al7075-T651 CFRP
problem72 = Problem(15, 1, 16)
problem72.types[:] = variable_types
problem72.constraints[0] = ">=0" #SEA
problem72.constraints[1] = "==0" #Susunan 2
problem72.constraints[2] = "==1" #Susunan 3
problem72.constraints[3] = "==0" #Susunan 4
problem72.constraints[4] = "==0" #Susunan 5
problem72.constraints[5] = "==0" #Susunan 6
problem72.constraints[6] = "==0" #Weldox 700E
problem72.constraints[7] = "==0" #Hardox 400
problem72.constraints[8] = "==0" #Domex Protect 500
problem72.constraints[9] = "==0" #Armox 560T
problem72.constraints[10] = "==1" #Al7075-T651
problem72.constraints[11] = "==0" #Kevlar-29
problem72.constraints[12] = "==0" #S2 Glass/SC15
problem72.constraints[13] = "==1" #CFRP
problem72.constraints[14] = ">=0.5" #thickness
problem72.constraints[15] = ">=1" #lapisan
problem72.directions[0] = Problem.MAXIMIZE
problem72.function = Opt

#Case73: Susunan 4 Weldox 500E CNC
problem73 = Problem(15, 1, 16)
problem73.types[:] = variable_types
problem73.constraints[0] = ">=0" #SEA
problem73.constraints[1] = "==0" #Susunan 2
problem73.constraints[2] = "==0" #Susunan 3
problem73.constraints[3] = "==1" #Susunan 4
problem73.constraints[4] = "==0" #Susunan 5
problem73.constraints[5] = "==0" #Susunan 6
problem73.constraints[6] = "==0" #Weldox 700E
problem73.constraints[7] = "==0" #Hardox 400
problem73.constraints[8] = "==0" #Domex Protect 500
problem73.constraints[9] = "==0" #Armox 560T
problem73.constraints[10] = "==0" #Al7075-T651
problem73.constraints[11] = "==0" #Kevlar-29
problem73.constraints[12] = "==0" #S2 Glass/SC15
problem73.constraints[13] = "==0" #CFRP
problem73.constraints[14] = ">=0.5" #thickness
problem73.constraints[15] = ">=1" #lapisan
problem73.directions[0] = Problem.MAXIMIZE
problem73.function = Opt

#Case74: Susunan 4 Weldox 500E Kevlar-29
problem74 = Problem(15, 1, 16)
problem74.types[:] = variable_types
problem74.constraints[0] = ">=0" #SEA
problem74.constraints[1] = "==0" #Susunan 2
problem74.constraints[2] = "==0" #Susunan 3
problem74.constraints[3] = "==1" #Susunan 4
problem74.constraints[4] = "==0" #Susunan 5
problem74.constraints[5] = "==0" #Susunan 6
problem74.constraints[6] = "==0" #Weldox 700E
problem74.constraints[7] = "==0" #Hardox 400
problem74.constraints[8] = "==0" #Domex Protect 500
problem74.constraints[9] = "==0" #Armox 560T
problem74.constraints[10] = "==0" #Al7075-T651
problem74.constraints[11] = "==1" #Kevlar-29
problem74.constraints[12] = "==0" #S2 Glass/SC15
problem74.constraints[13] = "==0" #CFRP
problem74.constraints[14] = ">=0.5" #thickness
problem74.constraints[15] = ">=1" #lapisan
problem74.directions[0] = Problem.MAXIMIZE
problem74.function = Opt

#Case75: Susunan 4 Weldox 500E S2Glass/SC15
problem75 = Problem(15, 1, 16)
problem75.types[:] = variable_types
problem75.constraints[0] = ">=0" #SEA
problem75.constraints[1] = "==0" #Susunan 2
problem75.constraints[2] = "==0" #Susunan 3
problem75.constraints[3] = "==1" #Susunan 4
problem75.constraints[4] = "==0" #Susunan 5
problem75.constraints[5] = "==0" #Susunan 6
problem75.constraints[6] = "==0" #Weldox 700E
problem75.constraints[7] = "==0" #Hardox 400
problem75.constraints[8] = "==0" #Domex Protect 500
problem75.constraints[9] = "==0" #Armox 560T
problem75.constraints[10] = "==0" #Al7075-T651
problem75.constraints[11] = "==0" #Kevlar-29
problem75.constraints[12] = "==1" #S2 Glass/SC15
problem75.constraints[13] = "==0" #CFRP
problem75.constraints[14] = ">=0.5" #thickness
problem75.constraints[15] = ">=1" #lapisan
problem75.directions[0] = Problem.MAXIMIZE
problem75.function = Opt

#Case76: Susunan 4 Weldox 500E CFRP
problem76 = Problem(15, 1, 16)
problem76.types[:] = variable_types
problem76.constraints[0] = ">=0" #SEA
problem76.constraints[1] = "==0" #Susunan 2
problem76.constraints[2] = "==0" #Susunan 3
problem76.constraints[3] = "==1" #Susunan 4
problem76.constraints[4] = "==0" #Susunan 5
problem76.constraints[5] = "==0" #Susunan 6
problem76.constraints[6] = "==0" #Weldox 700E
problem76.constraints[7] = "==0" #Hardox 400
problem76.constraints[8] = "==0" #Domex Protect 500
problem76.constraints[9] = "==0" #Armox 560T
problem76.constraints[10] = "==0" #Al7075-T651
problem76.constraints[11] = "==0" #Kevlar-29
problem76.constraints[12] = "==0" #S2 Glass/SC15
problem76.constraints[13] = "==1" #CFRP
problem76.constraints[14] = ">=0.5" #thickness
problem76.constraints[15] = ">=1" #lapisan
problem76.directions[0] = Problem.MAXIMIZE
problem76.function = Opt

#Case77: Susunan 4 Weldox 700E CNC
problem77 = Problem(15, 1, 16)
problem77.types[:] = variable_types
problem77.constraints[0] = ">=0" #SEA
problem77.constraints[1] = "==0" #Susunan 2
problem77.constraints[2] = "==0" #Susunan 3
problem77.constraints[3] = "==1" #Susunan 4
problem77.constraints[4] = "==0" #Susunan 5
problem77.constraints[5] = "==0" #Susunan 6
problem77.constraints[6] = "==1" #Weldox 700E
problem77.constraints[7] = "==0" #Hardox 400
problem77.constraints[8] = "==0" #Domex Protect 500
problem77.constraints[9] = "==0" #Armox 560T
problem77.constraints[10] = "==0" #Al7075-T651
problem77.constraints[11] = "==0" #Kevlar-29
problem77.constraints[12] = "==0" #S2 Glass/SC15
problem77.constraints[13] = "==0" #CFRP
problem77.constraints[14] = ">=0.5" #thickness
problem77.constraints[15] = ">=1" #lapisan
problem77.directions[0] = Problem.MAXIMIZE
problem77.function = Opt

#Case78: Susunan 4 Weldox 700E Kevlar-29
problem78 = Problem(15, 1, 16)
problem78.types[:] = variable_types
problem78.constraints[0] = ">=0" #SEA
problem78.constraints[1] = "==0" #Susunan 2
problem78.constraints[2] = "==0" #Susunan 3
problem78.constraints[3] = "==1" #Susunan 4
problem78.constraints[4] = "==0" #Susunan 5
problem78.constraints[5] = "==0" #Susunan 6
problem78.constraints[6] = "==1" #Weldox 700E
problem78.constraints[7] = "==0" #Hardox 400
problem78.constraints[8] = "==0" #Domex Protect 500
problem78.constraints[9] = "==0" #Armox 560T
problem78.constraints[10] = "==0" #Al7075-T651
problem78.constraints[11] = "==1" #Kevlar-29
problem78.constraints[12] = "==0" #S2 Glass/SC15
problem78.constraints[13] = "==0" #CFRP
problem78.constraints[14] = ">=0.5" #thickness
problem78.constraints[15] = ">=1" #lapisan
problem78.directions[0] = Problem.MAXIMIZE
problem78.function = Opt

#Case79: Susunan 4 Weldox 700E S2Glass/SC15
problem79 = Problem(15, 1, 16)
problem79.types[:] = variable_types
problem79.constraints[0] = ">=0" #SEA
problem79.constraints[1] = "==0" #Susunan 2
problem79.constraints[2] = "==0" #Susunan 3
problem79.constraints[3] = "==1" #Susunan 4
problem79.constraints[4] = "==0" #Susunan 5
problem79.constraints[5] = "==0" #Susunan 6
problem79.constraints[6] = "==1" #Weldox 700E
problem79.constraints[7] = "==0" #Hardox 400
problem79.constraints[8] = "==0" #Domex Protect 500
problem79.constraints[9] = "==0" #Armox 560T
problem79.constraints[10] = "==0" #Al7075-T651
problem79.constraints[11] = "==0" #Kevlar-29
problem79.constraints[12] = "==1" #S2 Glass/SC15
problem79.constraints[13] = "==0" #CFRP
problem79.constraints[14] = ">=0.5" #thickness
problem79.constraints[15] = ">=1" #lapisan
problem79.directions[0] = Problem.MAXIMIZE
problem79.function = Opt

#Case80: Susunan 4 Weldox 700E CFRP
problem80 = Problem(15, 1, 16)
problem80.types[:] = variable_types
problem80.constraints[0] = ">=0" #SEA
problem80.constraints[1] = "==0" #Susunan 2
problem80.constraints[2] = "==0" #Susunan 3
problem80.constraints[3] = "==1" #Susunan 4
problem80.constraints[4] = "==0" #Susunan 5
problem80.constraints[5] = "==0" #Susunan 6
problem80.constraints[6] = "==1" #Weldox 700E
problem80.constraints[7] = "==0" #Hardox 400
problem80.constraints[8] = "==0" #Domex Protect 500
problem80.constraints[9] = "==0" #Armox 560T
problem80.constraints[10] = "==0" #Al7075-T651
problem80.constraints[11] = "==0" #Kevlar-29
problem80.constraints[12] = "==0" #S2 Glass/SC15
problem80.constraints[13] = "==1" #CFRP
problem80.constraints[14] = ">=0.5" #thickness
problem80.constraints[15] = ">=1" #lapisan
problem80.directions[0] = Problem.MAXIMIZE
problem80.function = Opt

#Case81: Susunan 4 Hardox 400 CNC
problem81 = Problem(15, 1, 16)
problem81.types[:] = variable_types
problem81.constraints[0] = ">=0" #SEA
problem81.constraints[1] = "==0" #Susunan 2
problem81.constraints[2] = "==0" #Susunan 3
problem81.constraints[3] = "==1" #Susunan 4
problem81.constraints[4] = "==0" #Susunan 5
problem81.constraints[5] = "==0" #Susunan 6
problem81.constraints[6] = "==0" #Weldox 700E
problem81.constraints[7] = "==1" #Hardox 400
problem81.constraints[8] = "==0" #Domex Protect 500
problem81.constraints[9] = "==0" #Armox 560T
problem81.constraints[10] = "==0" #Al7075-T651
problem81.constraints[11] = "==0" #Kevlar-29
problem81.constraints[12] = "==0" #S2 Glass/SC15
problem81.constraints[13] = "==0" #CFRP
problem81.constraints[14] = ">=0.5" #thickness
problem81.constraints[15] = ">=1" #lapisan
problem81.directions[0] = Problem.MAXIMIZE
problem81.function = Opt

#Case82: Susunan 4 Hardox 400 Kevlar-29
problem82 = Problem(15, 1, 16)
problem82.types[:] = variable_types
problem82.constraints[0] = ">=0" #SEA
problem82.constraints[1] = "==0" #Susunan 2
problem82.constraints[2] = "==0" #Susunan 3
problem82.constraints[3] = "==1" #Susunan 4
problem82.constraints[4] = "==0" #Susunan 5
problem82.constraints[5] = "==0" #Susunan 6
problem82.constraints[6] = "==0" #Weldox 700E
problem82.constraints[7] = "==1" #Hardox 400
problem82.constraints[8] = "==0" #Domex Protect 500
problem82.constraints[9] = "==0" #Armox 560T
problem82.constraints[10] = "==0" #Al7075-T651
problem82.constraints[11] = "==1" #Kevlar-29
problem82.constraints[12] = "==0" #S2 Glass/SC15
problem82.constraints[13] = "==0" #CFRP
problem82.constraints[14] = ">=0.5" #thickness
problem82.constraints[15] = ">=1" #lapisan
problem82.directions[0] = Problem.MAXIMIZE
problem82.function = Opt

#Case83: Susunan 4 Hardox 400 S2Glass/SC15
problem83 = Problem(15, 1, 16)
problem83.types[:] = variable_types
problem83.constraints[0] = ">=0" #SEA
problem83.constraints[1] = "==0" #Susunan 2
problem83.constraints[2] = "==0" #Susunan 3
problem83.constraints[3] = "==1" #Susunan 4
problem83.constraints[4] = "==0" #Susunan 5
problem83.constraints[5] = "==0" #Susunan 6
problem83.constraints[6] = "==0" #Weldox 700E
problem83.constraints[7] = "==1" #Hardox 400
problem83.constraints[8] = "==0" #Domex Protect 500
problem83.constraints[9] = "==0" #Armox 560T
problem83.constraints[10] = "==0" #Al7075-T651
problem83.constraints[11] = "==0" #Kevlar-29
problem83.constraints[12] = "==1" #S2 Glass/SC15
problem83.constraints[13] = "==0" #CFRP
problem83.constraints[14] = ">=0.5" #thickness
problem83.constraints[15] = ">=1" #lapisan
problem83.directions[0] = Problem.MAXIMIZE
problem83.function = Opt

#Case84: Susunan 4 Hardox 400 CFRP
problem84 = Problem(15, 1, 16)
problem84.types[:] = variable_types
problem84.constraints[0] = ">=0" #SEA
problem84.constraints[1] = "==0" #Susunan 2
problem84.constraints[2] = "==0" #Susunan 3
problem84.constraints[3] = "==1" #Susunan 4
problem84.constraints[4] = "==0" #Susunan 5
problem84.constraints[5] = "==0" #Susunan 6
problem84.constraints[6] = "==0" #Weldox 700E
problem84.constraints[7] = "==1" #Hardox 400
problem84.constraints[8] = "==0" #Domex Protect 500
problem84.constraints[9] = "==0" #Armox 560T
problem84.constraints[10] = "==0" #Al7075-T651
problem84.constraints[11] = "==0" #Kevlar-29
problem84.constraints[12] = "==0" #S2 Glass/SC15
problem84.constraints[13] = "==1" #CFRP
problem84.constraints[14] = ">=0.5" #thickness
problem84.constraints[15] = ">=1" #lapisan
problem84.directions[0] = Problem.MAXIMIZE
problem84.function = Opt

#Case85: Susunan 4 Domex Protect 500 CNC
problem85 = Problem(15, 1, 16)
problem85.types[:] = variable_types
problem85.constraints[0] = ">=0" #SEA
problem85.constraints[1] = "==0" #Susunan 2
problem85.constraints[2] = "==0" #Susunan 3
problem85.constraints[3] = "==1" #Susunan 4
problem85.constraints[4] = "==0" #Susunan 5
problem85.constraints[5] = "==0" #Susunan 6
problem85.constraints[6] = "==0" #Weldox 700E
problem85.constraints[7] = "==0" #Hardox 400
problem85.constraints[8] = "==1" #Domex Protect 500
problem85.constraints[9] = "==0" #Armox 560T
problem85.constraints[10] = "==0" #Al7075-T651
problem85.constraints[11] = "==0" #Kevlar-29
problem85.constraints[12] = "==0" #S2 Glass/SC15
problem85.constraints[13] = "==0" #CFRP
problem85.constraints[14] = ">=0.5" #thickness
problem85.constraints[15] = ">=1" #lapisan
problem85.directions[0] = Problem.MAXIMIZE
problem85.function = Opt

#Case86: Susunan 4 Domex Protect 500 Kevlar-29
problem86 = Problem(15, 1, 16)
problem86.types[:] = variable_types
problem86.constraints[0] = ">=0" #SEA
problem86.constraints[1] = "==0" #Susunan 2
problem86.constraints[2] = "==0" #Susunan 3
problem86.constraints[3] = "==1" #Susunan 4
problem86.constraints[4] = "==0" #Susunan 5
problem86.constraints[5] = "==0" #Susunan 6
problem86.constraints[6] = "==0" #Weldox 700E
problem86.constraints[7] = "==0" #Hardox 400
problem86.constraints[8] = "==1" #Domex Protect 500
problem86.constraints[9] = "==0" #Armox 560T
problem86.constraints[10] = "==0" #Al7075-T651
problem86.constraints[11] = "==1" #Kevlar-29
problem86.constraints[12] = "==0" #S2 Glass/SC15
problem86.constraints[13] = "==0" #CFRP
problem86.constraints[14] = ">=0.5" #thickness
problem86.constraints[15] = ">=1" #lapisan
problem86.directions[0] = Problem.MAXIMIZE
problem86.function = Opt

#Case87: Susunan 4 Domex Protect 500 S2Glass/SC15
problem87 = Problem(15, 1, 16)
problem87.types[:] = variable_types
problem87.constraints[0] = ">=0" #SEA
problem87.constraints[1] = "==0" #Susunan 2
problem87.constraints[2] = "==0" #Susunan 3
problem87.constraints[3] = "==1" #Susunan 4
problem87.constraints[4] = "==0" #Susunan 5
problem87.constraints[5] = "==0" #Susunan 6
problem87.constraints[6] = "==0" #Weldox 700E
problem87.constraints[7] = "==0" #Hardox 400
problem87.constraints[8] = "==1" #Domex Protect 500
problem87.constraints[9] = "==0" #Armox 560T
problem87.constraints[10] = "==0" #Al7075-T651
problem87.constraints[11] = "==0" #Kevlar-29
problem87.constraints[12] = "==1" #S2 Glass/SC15
problem87.constraints[13] = "==0" #CFRP
problem87.constraints[14] = ">=0.5" #thickness
problem87.constraints[15] = ">=1" #lapisan
problem87.directions[0] = Problem.MAXIMIZE
problem87.function = Opt

#Case88: Susunan 4 Domex Protect 500 CFRP
problem88 = Problem(15, 1, 16)
problem88.types[:] = variable_types
problem88.constraints[0] = ">=0" #SEA
problem88.constraints[1] = "==0" #Susunan 2
problem88.constraints[2] = "==0" #Susunan 3
problem88.constraints[3] = "==1" #Susunan 4
problem88.constraints[4] = "==0" #Susunan 5
problem88.constraints[5] = "==0" #Susunan 6
problem88.constraints[6] = "==0" #Weldox 700E
problem88.constraints[7] = "==0" #Hardox 400
problem88.constraints[8] = "==1" #Domex Protect 500
problem88.constraints[9] = "==0" #Armox 560T
problem88.constraints[10] = "==0" #Al7075-T651
problem88.constraints[11] = "==0" #Kevlar-29
problem88.constraints[12] = "==0" #S2 Glass/SC15
problem88.constraints[13] = "==1" #CFRP
problem88.constraints[14] = ">=0.5" #thickness
problem88.constraints[15] = ">=1" #lapisan
problem88.directions[0] = Problem.MAXIMIZE
problem88.function = Opt

#Case89: Susunan 4 Armox 560T CNC
problem89 = Problem(15, 1, 16)
problem89.types[:] = variable_types
problem89.constraints[0] = ">=0" #SEA
problem89.constraints[1] = "==0" #Susunan 2
problem89.constraints[2] = "==0" #Susunan 3
problem89.constraints[3] = "==1" #Susunan 4
problem89.constraints[4] = "==0" #Susunan 5
problem89.constraints[5] = "==0" #Susunan 6
problem89.constraints[6] = "==0" #Weldox 700E
problem89.constraints[7] = "==0" #Hardox 400
problem89.constraints[8] = "==0" #Domex Protect 500
problem89.constraints[9] = "==1" #Armox 560T
problem89.constraints[10] = "==0" #Al7075-T651
problem89.constraints[11] = "==0" #Kevlar-29
problem89.constraints[12] = "==0" #S2 Glass/SC15
problem89.constraints[13] = "==0" #CFRP
problem89.constraints[14] = ">=0.5" #thickness
problem89.constraints[15] = ">=1" #lapisan
problem89.directions[0] = Problem.MAXIMIZE
problem89.function = Opt

#Case90: Susunan 4 Armox 560T Kevlar-29
problem90 = Problem(15, 1, 16)
problem90.types[:] = variable_types
problem90.constraints[0] = ">=0" #SEA
problem90.constraints[1] = "==0" #Susunan 2
problem90.constraints[2] = "==0" #Susunan 3
problem90.constraints[3] = "==1" #Susunan 4
problem90.constraints[4] = "==0" #Susunan 5
problem90.constraints[5] = "==0" #Susunan 6
problem90.constraints[6] = "==0" #Weldox 700E
problem90.constraints[7] = "==0" #Hardox 400
problem90.constraints[8] = "==0" #Domex Protect 500
problem90.constraints[9] = "==1" #Armox 560T
problem90.constraints[10] = "==0" #Al7075-T651
problem90.constraints[11] = "==1" #Kevlar-29
problem90.constraints[12] = "==0" #S2 Glass/SC15
problem90.constraints[13] = "==0" #CFRP
problem90.constraints[14] = ">=0.5" #thickness
problem90.constraints[15] = ">=1" #lapisan
problem90.directions[0] = Problem.MAXIMIZE
problem90.function = Opt

#Case91: Susunan 4 Armox 560T S2Glass/SC15
problem91 = Problem(15, 1, 16)
problem91.types[:] = variable_types
problem91.constraints[0] = ">=0" #SEA
problem91.constraints[1] = "==0" #Susunan 2
problem91.constraints[2] = "==0" #Susunan 3
problem91.constraints[3] = "==1" #Susunan 4
problem91.constraints[4] = "==0" #Susunan 5
problem91.constraints[5] = "==0" #Susunan 6
problem91.constraints[6] = "==0" #Weldox 700E
problem91.constraints[7] = "==0" #Hardox 400
problem91.constraints[8] = "==0" #Domex Protect 500
problem91.constraints[9] = "==1" #Armox 560T
problem91.constraints[10] = "==0" #Al7075-T651
problem91.constraints[11] = "==0" #Kevlar-29
problem91.constraints[12] = "==1" #S2 Glass/SC15
problem91.constraints[13] = "==0" #CFRP
problem91.constraints[14] = ">=0.5" #thickness
problem91.constraints[15] = ">=1" #lapisan
problem91.directions[0] = Problem.MAXIMIZE
problem91.function = Opt

#Case92: Susunan 4 Armox 560T CFRP
problem92 = Problem(15, 1, 16)
problem92.types[:] = variable_types
problem92.constraints[0] = ">=0" #SEA
problem92.constraints[1] = "==0" #Susunan 2
problem92.constraints[2] = "==0" #Susunan 3
problem92.constraints[3] = "==1" #Susunan 4
problem92.constraints[4] = "==0" #Susunan 5
problem92.constraints[5] = "==0" #Susunan 6
problem92.constraints[6] = "==0" #Weldox 700E
problem92.constraints[7] = "==0" #Hardox 400
problem92.constraints[8] = "==0" #Domex Protect 500
problem92.constraints[9] = "==1" #Armox 560T
problem92.constraints[10] = "==0" #Al7075-T651
problem92.constraints[11] = "==0" #Kevlar-29
problem92.constraints[12] = "==0" #S2 Glass/SC15
problem92.constraints[13] = "==1" #CFRP
problem92.constraints[14] = ">=0.5" #thickness
problem92.constraints[15] = ">=1" #lapisan
problem92.directions[0] = Problem.MAXIMIZE
problem92.function = Opt

#Case93: Susunan 4 Al7075-T651 CNC
problem93 = Problem(15, 1, 16)
problem93.types[:] = variable_types
problem93.constraints[0] = ">=0" #SEA
problem93.constraints[1] = "==0" #Susunan 2
problem93.constraints[2] = "==0" #Susunan 3
problem93.constraints[3] = "==1" #Susunan 4
problem93.constraints[4] = "==0" #Susunan 5
problem93.constraints[5] = "==0" #Susunan 6
problem93.constraints[6] = "==0" #Weldox 700E
problem93.constraints[7] = "==0" #Hardox 400
problem93.constraints[8] = "==0" #Domex Protect 500
problem93.constraints[9] = "==0" #Armox 560T
problem93.constraints[10] = "==1" #Al7075-T651
problem93.constraints[11] = "==0" #Kevlar-29
problem93.constraints[12] = "==0" #S2 Glass/SC15
problem93.constraints[13] = "==0" #CFRP
problem93.constraints[14] = ">=0.5" #thickness
problem93.constraints[15] = ">=1" #lapisan
problem93.directions[0] = Problem.MAXIMIZE
problem93.function = Opt

#Case94: Susunan 4 Al7075-T651 Kevlar-29
problem94 = Problem(15, 1, 16)
problem94.types[:] = variable_types
problem94.constraints[0] = ">=0" #SEA
problem94.constraints[1] = "==0" #Susunan 2
problem94.constraints[2] = "==0" #Susunan 3
problem94.constraints[3] = "==1" #Susunan 4
problem94.constraints[4] = "==0" #Susunan 5
problem94.constraints[5] = "==0" #Susunan 6
problem94.constraints[6] = "==0" #Weldox 700E
problem94.constraints[7] = "==0" #Hardox 400
problem94.constraints[8] = "==0" #Domex Protect 500
problem94.constraints[9] = "==0" #Armox 560T
problem94.constraints[10] = "==1" #Al7075-T651
problem94.constraints[11] = "==1" #Kevlar-29
problem94.constraints[12] = "==0" #S2 Glass/SC15
problem94.constraints[13] = "==0" #CFRP
problem94.constraints[14] = ">=0.5" #thickness
problem94.constraints[15] = ">=1" #lapisan
problem94.directions[0] = Problem.MAXIMIZE
problem94.function = Opt

#Case95: Susunan 4 Al7075-T651 S2Glass/SC15
problem95 = Problem(15, 1, 16)
problem95.types[:] = variable_types
problem95.constraints[0] = ">=0" #SEA
problem95.constraints[1] = "==0" #Susunan 2
problem95.constraints[2] = "==0" #Susunan 3
problem95.constraints[3] = "==1" #Susunan 4
problem95.constraints[4] = "==0" #Susunan 5
problem95.constraints[5] = "==0" #Susunan 6
problem95.constraints[6] = "==0" #Weldox 700E
problem95.constraints[7] = "==0" #Hardox 400
problem95.constraints[8] = "==0" #Domex Protect 500
problem95.constraints[9] = "==0" #Armox 560T
problem95.constraints[10] = "==1" #Al7075-T651
problem95.constraints[11] = "==0" #Kevlar-29
problem95.constraints[12] = "==1" #S2 Glass/SC15
problem95.constraints[13] = "==0" #CFRP
problem95.constraints[14] = ">=0.5" #thickness
problem95.constraints[15] = ">=1" #lapisan
problem95.directions[0] = Problem.MAXIMIZE
problem95.function = Opt

#Case96: Susunan 4 Al7075-T651 CFRP
problem96 = Problem(15, 1, 16)
problem96.types[:] = variable_types
problem96.constraints[0] = ">=0" #SEA
problem96.constraints[1] = "==0" #Susunan 2
problem96.constraints[2] = "==0" #Susunan 3
problem96.constraints[3] = "==1" #Susunan 4
problem96.constraints[4] = "==0" #Susunan 5
problem96.constraints[5] = "==0" #Susunan 6
problem96.constraints[6] = "==0" #Weldox 700E
problem96.constraints[7] = "==0" #Hardox 400
problem96.constraints[8] = "==0" #Domex Protect 500
problem96.constraints[9] = "==0" #Armox 560T
problem96.constraints[10] = "==1" #Al7075-T651
problem96.constraints[11] = "==0" #Kevlar-29
problem96.constraints[12] = "==0" #S2 Glass/SC15
problem96.constraints[13] = "==1" #CFRP
problem96.constraints[14] = ">=0.5" #thickness
problem96.constraints[15] = ">=1" #lapisan
problem96.directions[0] = Problem.MAXIMIZE
problem96.function = Opt

#Case97: Susunan 5 Weldox 500E CNC
problem97 = Problem(15, 1, 16)
problem97.types[:] = variable_types
problem97.constraints[0] = ">=0" #SEA
problem97.constraints[1] = "==0" #Susunan 2
problem97.constraints[2] = "==0" #Susunan 3
problem97.constraints[3] = "==0" #Susunan 4
problem97.constraints[4] = "==1" #Susunan 5
problem97.constraints[5] = "==0" #Susunan 6
problem97.constraints[6] = "==0" #Weldox 700E
problem97.constraints[7] = "==0" #Hardox 400
problem97.constraints[8] = "==0" #Domex Protect 500
problem97.constraints[9] = "==0" #Armox 560T
problem97.constraints[10] = "==0" #Al7075-T651
problem97.constraints[11] = "==0" #Kevlar-29
problem97.constraints[12] = "==0" #S2 Glass/SC15
problem97.constraints[13] = "==0" #CFRP
problem97.constraints[14] = ">=0.5" #thickness
problem97.constraints[15] = ">=1" #lapisan
problem97.directions[0] = Problem.MAXIMIZE
problem97.function = Opt

#Case98: Susunan 5 Weldox 500E Kevlar-29
problem98 = Problem(15, 1, 16)
problem98.types[:] = variable_types
problem98.constraints[0] = ">=0" #SEA
problem98.constraints[1] = "==0" #Susunan 2
problem98.constraints[2] = "==0" #Susunan 3
problem98.constraints[3] = "==0" #Susunan 4
problem98.constraints[4] = "==1" #Susunan 5
problem98.constraints[5] = "==0" #Susunan 6
problem98.constraints[6] = "==0" #Weldox 700E
problem98.constraints[7] = "==0" #Hardox 400
problem98.constraints[8] = "==0" #Domex Protect 500
problem98.constraints[9] = "==0" #Armox 560T
problem98.constraints[10] = "==0" #Al7075-T651
problem98.constraints[11] = "==1" #Kevlar-29
problem98.constraints[12] = "==0" #S2 Glass/SC15
problem98.constraints[13] = "==0" #CFRP
problem98.constraints[14] = ">=0.5" #thickness
problem98.constraints[15] = ">=1" #lapisan
problem98.directions[0] = Problem.MAXIMIZE
problem98.function = Opt

#Case99: Susunan 5 Weldox 500E S2Glass/SC15
problem99 = Problem(15, 1, 16)
problem99.types[:] = variable_types
problem99.constraints[0] = ">=0" #SEA
problem99.constraints[1] = "==0" #Susunan 2
problem99.constraints[2] = "==0" #Susunan 3
problem99.constraints[3] = "==0" #Susunan 4
problem99.constraints[4] = "==1" #Susunan 5
problem99.constraints[5] = "==0" #Susunan 6
problem99.constraints[6] = "==0" #Weldox 700E
problem99.constraints[7] = "==0" #Hardox 400
problem99.constraints[8] = "==0" #Domex Protect 500
problem99.constraints[9] = "==0" #Armox 560T
problem99.constraints[10] = "==0" #Al7075-T651
problem99.constraints[11] = "==0" #Kevlar-29
problem99.constraints[12] = "==1" #S2 Glass/SC15
problem99.constraints[13] = "==0" #CFRP
problem99.constraints[14] = ">=0.5" #thickness
problem99.constraints[15] = ">=1" #lapisan
problem99.directions[0] = Problem.MAXIMIZE
problem99.function = Opt

#Case100: Susunan 5 Weldox 500E CFRP
problem100 = Problem(15, 1, 16)
problem100.types[:] = variable_types
problem100.constraints[0] = ">=0" #SEA
problem100.constraints[1] = "==0" #Susunan 2
problem100.constraints[2] = "==0" #Susunan 3
problem100.constraints[3] = "==0" #Susunan 4
problem100.constraints[4] = "==1" #Susunan 5
problem100.constraints[5] = "==0" #Susunan 6
problem100.constraints[6] = "==0" #Weldox 700E
problem100.constraints[7] = "==0" #Hardox 400
problem100.constraints[8] = "==0" #Domex Protect 500
problem100.constraints[9] = "==0" #Armox 560T
problem100.constraints[10] = "==0" #Al7075-T651
problem100.constraints[11] = "==0" #Kevlar-29
problem100.constraints[12] = "==0" #S2 Glass/SC15
problem100.constraints[13] = "==1" #CFRP
problem100.constraints[14] = ">=0.5" #thickness
problem100.constraints[15] = ">=1" #lapisan
problem100.directions[0] = Problem.MAXIMIZE
problem100.function = Opt

#Case101: Susunan 5 Weldox 700E CNC
problem101 = Problem(15, 1, 16)
problem101.types[:] = variable_types
problem101.constraints[0] = ">=0" #SEA
problem101.constraints[1] = "==0" #Susunan 2
problem101.constraints[2] = "==0" #Susunan 3
problem101.constraints[3] = "==0" #Susunan 4
problem101.constraints[4] = "==1" #Susunan 5
problem101.constraints[5] = "==0" #Susunan 6
problem101.constraints[6] = "==1" #Weldox 700E
problem101.constraints[7] = "==0" #Hardox 400
problem101.constraints[8] = "==0" #Domex Protect 500
problem101.constraints[9] = "==0" #Armox 560T
problem101.constraints[10] = "==0" #Al7075-T651
problem101.constraints[11] = "==0" #Kevlar-29
problem101.constraints[12] = "==0" #S2 Glass/SC15
problem101.constraints[13] = "==0" #CFRP
problem101.constraints[14] = ">=0.5" #thickness
problem101.constraints[15] = ">=1" #lapisan
problem101.directions[0] = Problem.MAXIMIZE
problem101.function = Opt

#Case102: Susunan 5 Weldox 700E Kevlar-29
problem102 = Problem(15, 1, 16)
problem102.types[:] = variable_types
problem102.constraints[0] = ">=0" #SEA
problem102.constraints[1] = "==0" #Susunan 2
problem102.constraints[2] = "==0" #Susunan 3
problem102.constraints[3] = "==0" #Susunan 4
problem102.constraints[4] = "==1" #Susunan 5
problem102.constraints[5] = "==0" #Susunan 6
problem102.constraints[6] = "==1" #Weldox 700E
problem102.constraints[7] = "==0" #Hardox 400
problem102.constraints[8] = "==0" #Domex Protect 500
problem102.constraints[9] = "==0" #Armox 560T
problem102.constraints[10] = "==0" #Al7075-T651
problem102.constraints[11] = "==1" #Kevlar-29
problem102.constraints[12] = "==0" #S2 Glass/SC15
problem102.constraints[13] = "==0" #CFRP
problem102.constraints[14] = ">=0.5" #thickness
problem102.constraints[15] = ">=1" #lapisan
problem102.directions[0] = Problem.MAXIMIZE
problem102.function = Opt

#Case103: Susunan 5 Weldox 700E S2Glass/SC15
problem103 = Problem(15, 1, 16)
problem103.types[:] = variable_types
problem103.constraints[0] = ">=0" #SEA
problem103.constraints[1] = "==0" #Susunan 2
problem103.constraints[2] = "==0" #Susunan 3
problem103.constraints[3] = "==0" #Susunan 4
problem103.constraints[4] = "==1" #Susunan 5
problem103.constraints[5] = "==0" #Susunan 6
problem103.constraints[6] = "==1" #Weldox 700E
problem103.constraints[7] = "==0" #Hardox 400
problem103.constraints[8] = "==0" #Domex Protect 500
problem103.constraints[9] = "==0" #Armox 560T
problem103.constraints[10] = "==0" #Al7075-T651
problem103.constraints[11] = "==0" #Kevlar-29
problem103.constraints[12] = "==1" #S2 Glass/SC15
problem103.constraints[13] = "==0" #CFRP
problem103.constraints[14] = ">=0.5" #thickness
problem103.constraints[15] = ">=1" #lapisan
problem103.directions[0] = Problem.MAXIMIZE
problem103.function = Opt

#Case104: Susunan 5 Weldox 700E CFRP
problem104 = Problem(15, 1, 16)
problem104.types[:] = variable_types
problem104.constraints[0] = ">=0" #SEA
problem104.constraints[1] = "==0" #Susunan 2
problem104.constraints[2] = "==0" #Susunan 3
problem104.constraints[3] = "==0" #Susunan 4
problem104.constraints[4] = "==1" #Susunan 5
problem104.constraints[5] = "==0" #Susunan 6
problem104.constraints[6] = "==1" #Weldox 700E
problem104.constraints[7] = "==0" #Hardox 400
problem104.constraints[8] = "==0" #Domex Protect 500
problem104.constraints[9] = "==0" #Armox 560T
problem104.constraints[10] = "==0" #Al7075-T651
problem104.constraints[11] = "==0" #Kevlar-29
problem104.constraints[12] = "==0" #S2 Glass/SC15
problem104.constraints[13] = "==1" #CFRP
problem104.constraints[14] = ">=0.5" #thickness
problem104.constraints[15] = ">=1" #lapisan
problem104.directions[0] = Problem.MAXIMIZE
problem104.function = Opt

#Case105: Susunan 5 Hardox 400 CNC
problem105 = Problem(15, 1, 16)
problem105.types[:] = variable_types
problem105.constraints[0] = ">=0" #SEA
problem105.constraints[1] = "==0" #Susunan 2
problem105.constraints[2] = "==0" #Susunan 3
problem105.constraints[3] = "==0" #Susunan 4
problem105.constraints[4] = "==1" #Susunan 5
problem105.constraints[5] = "==0" #Susunan 6
problem105.constraints[6] = "==0" #Weldox 700E
problem105.constraints[7] = "==1" #Hardox 400
problem105.constraints[8] = "==0" #Domex Protect 500
problem105.constraints[9] = "==0" #Armox 560T
problem105.constraints[10] = "==0" #Al7075-T651
problem105.constraints[11] = "==0" #Kevlar-29
problem105.constraints[12] = "==0" #S2 Glass/SC15
problem105.constraints[13] = "==0" #CFRP
problem105.constraints[14] = ">=0.5" #thickness
problem105.constraints[15] = ">=1" #lapisan
problem105.directions[0] = Problem.MAXIMIZE
problem105.function = Opt

#Case106: Susunan 5 Hardox 400 Kevlar-29
problem106 = Problem(15, 1, 16)
problem106.types[:] = variable_types
problem106.constraints[0] = ">=0" #SEA
problem106.constraints[1] = "==0" #Susunan 2
problem106.constraints[2] = "==0" #Susunan 3
problem106.constraints[3] = "==0" #Susunan 4
problem106.constraints[4] = "==1" #Susunan 5
problem106.constraints[5] = "==0" #Susunan 6
problem106.constraints[6] = "==0" #Weldox 700E
problem106.constraints[7] = "==1" #Hardox 400
problem106.constraints[8] = "==0" #Domex Protect 500
problem106.constraints[9] = "==0" #Armox 560T
problem106.constraints[10] = "==0" #Al7075-T651
problem106.constraints[11] = "==1" #Kevlar-29
problem106.constraints[12] = "==0" #S2 Glass/SC15
problem106.constraints[13] = "==0" #CFRP
problem106.constraints[14] = ">=0.5" #thickness
problem106.constraints[15] = ">=1" #lapisan
problem106.directions[0] = Problem.MAXIMIZE
problem106.function = Opt

#Case107: Susunan 5 Hardox 400 S2Glass/SC15
problem107 = Problem(15, 1, 16)
problem107.types[:] = variable_types
problem107.constraints[0] = ">=0" #SEA
problem107.constraints[1] = "==0" #Susunan 2
problem107.constraints[2] = "==0" #Susunan 3
problem107.constraints[3] = "==0" #Susunan 4
problem107.constraints[4] = "==1" #Susunan 5
problem107.constraints[5] = "==0" #Susunan 6
problem107.constraints[6] = "==0" #Weldox 700E
problem107.constraints[7] = "==1" #Hardox 400
problem107.constraints[8] = "==0" #Domex Protect 500
problem107.constraints[9] = "==0" #Armox 560T
problem107.constraints[10] = "==0" #Al7075-T651
problem107.constraints[11] = "==0" #Kevlar-29
problem107.constraints[12] = "==1" #S2 Glass/SC15
problem107.constraints[13] = "==0" #CFRP
problem107.constraints[14] = ">=0.5" #thickness
problem107.constraints[15] = ">=1" #lapisan
problem107.directions[0] = Problem.MAXIMIZE
problem107.function = Opt

#Case108: Susunan 5 Hardox 400 CFRP
problem108 = Problem(15, 1, 16)
problem108.types[:] = variable_types
problem108.constraints[0] = ">=0" #SEA
problem108.constraints[1] = "==0" #Susunan 2
problem108.constraints[2] = "==0" #Susunan 3
problem108.constraints[3] = "==0" #Susunan 4
problem108.constraints[4] = "==1" #Susunan 5
problem108.constraints[5] = "==0" #Susunan 6
problem108.constraints[6] = "==0" #Weldox 700E
problem108.constraints[7] = "==1" #Hardox 400
problem108.constraints[8] = "==0" #Domex Protect 500
problem108.constraints[9] = "==0" #Armox 560T
problem108.constraints[10] = "==0" #Al7075-T651
problem108.constraints[11] = "==0" #Kevlar-29
problem108.constraints[12] = "==0" #S2 Glass/SC15
problem108.constraints[13] = "==1" #CFRP
problem108.constraints[14] = ">=0.5" #thickness
problem108.constraints[15] = ">=1" #lapisan
problem108.directions[0] = Problem.MAXIMIZE
problem108.function = Opt
#Create problem-n definition for 51 other cases

#Iterasi setiap case
iteration =1000
i = 37
algorithm37 = NSGAII(problem37, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm37.run(iteration)
i = 38
algorithm38 = NSGAII(problem38, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm38.run(iteration)
i = 39
algorithm39 = NSGAII(problem39, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm39.run(iteration)
i = 40
algorithm40 = NSGAII(problem40, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm40.run(iteration)
i = 41
algorithm41 = NSGAII(problem41, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm41.run(iteration)
i = 42
algorithm42 = NSGAII(problem42, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm42.run(iteration)
i = 43
algorithm43 = NSGAII(problem43, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm43.run(iteration)
i = 44
algorithm44 = NSGAII(problem44, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm44.run(iteration)
i = 45
algorithm45 = NSGAII(problem45, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm45.run(iteration)
i = 46
algorithm46 = NSGAII(problem46, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm46.run(iteration)
i = 47
algorithm47 = NSGAII(problem47, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm47.run(iteration)
i = 48
algorithm48 = NSGAII(problem48, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm48.run(iteration)
i = 49
algorithm49 = NSGAII(problem49, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm49.run(iteration)
i = 50
algorithm50 = NSGAII(problem50, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm50.run(iteration)
i = 51
algorithm51 = NSGAII(problem51, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm51.run(iteration)
i = 52
algorithm52 = NSGAII(problem52, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm52.run(iteration)
i = 53
algorithm53 = NSGAII(problem53, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm53.run(iteration)
i = 54
algorithm54 = NSGAII(problem54, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm54.run(iteration)
i = 55
algorithm55 = NSGAII(problem55, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm55.run(iteration)
i = 56
algorithm56 = NSGAII(problem56, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm56.run(iteration)
i = 57
algorithm57 = NSGAII(problem57, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm57.run(iteration)
i = 58
algorithm58 = NSGAII(problem58, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm58.run(iteration)
i = 59
algorithm59 = NSGAII(problem59, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm59.run(iteration)
i = 60
algorithm60 = NSGAII(problem60, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm60.run(iteration)
i = 61
algorithm61 = NSGAII(problem61, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm61.run(iteration)
i = 62
algorithm62 = NSGAII(problem62, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm62.run(iteration)
i = 63
algorithm63 = NSGAII(problem63, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm63.run(iteration)
i = 64
algorithm64 = NSGAII(problem64, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm64.run(iteration)
i = 65
algorithm65 = NSGAII(problem65, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm65.run(iteration)
i = 66
algorithm66 = NSGAII(problem66, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm66.run(iteration)
i = 67
algorithm67 = NSGAII(problem67, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm67.run(iteration)
i = 68
algorithm68 = NSGAII(problem68, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm68.run(iteration)
i = 69
algorithm69 = NSGAII(problem69, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm69.run(iteration)
i = 70
algorithm70 = NSGAII(problem70, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm70.run(iteration)
i = 71
algorithm71 = NSGAII(problem71, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm71.run(iteration)
i = 72
algorithm72 = NSGAII(problem72, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm72.run(iteration)
i = 73
algorithm73 = NSGAII(problem73, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm73.run(iteration)
i = 74
algorithm74 = NSGAII(problem74, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm74.run(iteration)
i = 75
algorithm75 = NSGAII(problem75, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm75.run(iteration)
i = 76
algorithm76 = NSGAII(problem76, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm76.run(iteration)
i = 77
algorithm77 = NSGAII(problem77, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm77.run(iteration)
i = 78
algorithm78 = NSGAII(problem78, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm78.run(iteration)
i = 79
algorithm79 = NSGAII(problem79, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm79.run(iteration)
i = 80
algorithm80 = NSGAII(problem80, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm80.run(iteration)
i = 81
algorithm81 = NSGAII(problem81, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm81.run(iteration)
i = 82
algorithm82 = NSGAII(problem82, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm82.run(iteration)
i = 83
algorithm83 = NSGAII(problem83, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm83.run(iteration)
i = 84
algorithm84 = NSGAII(problem84, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm84.run(iteration)
i = 85
algorithm85 = NSGAII(problem85, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm85.run(iteration)
i = 86
algorithm86 = NSGAII(problem86, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm86.run(iteration)
i = 87
algorithm87 = NSGAII(problem87, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm87.run(iteration)
i = 88
algorithm88 = NSGAII(problem88, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm88.run(iteration)
i = 89
algorithm89 = NSGAII(problem89, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm89.run(iteration)
i = 90
algorithm90 = NSGAII(problem90, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm90.run(iteration)
i = 91
algorithm91 = NSGAII(problem91, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm91.run(iteration)
i = 92
algorithm92 = NSGAII(problem92, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm92.run(iteration)
i = 93
algorithm93 = NSGAII(problem93, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm93.run(iteration)
i = 94
algorithm94 = NSGAII(problem94, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm94.run(iteration)
i = 95
algorithm95 = NSGAII(problem95, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm95.run(iteration)
i = 96
algorithm96 = NSGAII(problem96, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm96.run(iteration)
i = 97
algorithm97 = NSGAII(problem97, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm97.run(iteration)
i = 98
algorithm98 = NSGAII(problem98, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm98.run(iteration)
i = 99
algorithm99 = NSGAII(problem99, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm99.run(iteration)
i = 100
algorithm100 = NSGAII(problem100, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm100.run(iteration)
i = 101
algorithm101 = NSGAII(problem101, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm101.run(iteration)
i = 102
algorithm102 = NSGAII(problem102, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm102.run(iteration)
i = 103
algorithm103 = NSGAII(problem103, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm103.run(iteration)
i = 104
algorithm104 = NSGAII(problem104, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm104.run(iteration)
i = 105
algorithm105 = NSGAII(problem105, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm105.run(iteration)
i = 106
algorithm106 = NSGAII(problem106, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm106.run(iteration)
i = 107
algorithm107 = NSGAII(problem107, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm107.run(iteration)
i = 108
algorithm108 = NSGAII(problem108, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm108.run(iteration)
#Create problem-i definition for 51 other cases
result37 = algorithm37.result
result38 = algorithm38.result
result39 = algorithm39.result
result40 = algorithm40.result
result41 = algorithm41.result
result42 = algorithm42.result
result43 = algorithm43.result
result44 = algorithm44.result
result45 = algorithm45.result
result46 = algorithm46.result
result47 = algorithm47.result
result48 = algorithm48.result
result49 = algorithm49.result
result50 = algorithm50.result
result51 = algorithm51.result
result52 = algorithm52.result
result53 = algorithm53.result
result54 = algorithm54.result
result55 = algorithm55.result
result56 = algorithm56.result
result57 = algorithm57.result
result58 = algorithm58.result
result59 = algorithm59.result
result60 = algorithm60.result
result61 = algorithm61.result
result62 = algorithm62.result
result63 = algorithm63.result
result64 = algorithm64.result
result65 = algorithm65.result
result66 = algorithm66.result
result67 = algorithm67.result
result68 = algorithm68.result
result69 = algorithm69.result
result70 = algorithm70.result
result71 = algorithm71.result
result72 = algorithm72.result
result73 = algorithm73.result
result74 = algorithm74.result
result75 = algorithm75.result
result76 = algorithm76.result
result77 = algorithm77.result
result78 = algorithm78.result
result79 = algorithm79.result
result80 = algorithm80.result
result81 = algorithm81.result
result82 = algorithm82.result
result83 = algorithm83.result
result84 = algorithm84.result
result85 = algorithm85.result
result86 = algorithm86.result
result87 = algorithm87.result
result88 = algorithm88.result
result89 = algorithm89.result
result90 = algorithm90.result
result91 = algorithm91.result
result92 = algorithm92.result
result93 = algorithm93.result
result94 = algorithm94.result
result95 = algorithm95.result
result96 = algorithm96.result
result97 = algorithm97.result
result98 = algorithm98.result
result99 = algorithm99.result
result100 = algorithm100.result
result101 = algorithm101.result
result102 = algorithm102.result
result103 = algorithm103.result
result104 = algorithm104.result
result105 = algorithm105.result
result106 = algorithm106.result
result107 = algorithm107.result
result108 = algorithm108.result
#Create result-i definition for 51 other cases

feasible_solutions = [s for s in result37 if s.feasible]
for solution in feasible_solutions:
    print(solution.objectives)
SEA_hasil37 = [s.constraints[0] for s in algorithm37.result if s.feasible]
SEA_hasil38 = [s.constraints[0] for s in algorithm38.result if s.feasible]
SEA_hasil39 = [s.constraints[0] for s in algorithm39.result if s.feasible]
SEA_hasil40 = [s.constraints[0] for s in algorithm40.result if s.feasible]
SEA_hasil41 = [s.constraints[0] for s in algorithm41.result if s.feasible]
SEA_hasil42 = [s.constraints[0] for s in algorithm42.result if s.feasible]
SEA_hasil43 = [s.constraints[0] for s in algorithm43.result if s.feasible]
SEA_hasil44 = [s.constraints[0] for s in algorithm44.result if s.feasible]
SEA_hasil45 = [s.constraints[0] for s in algorithm45.result if s.feasible]
SEA_hasil46 = [s.constraints[0] for s in algorithm46.result if s.feasible]
SEA_hasil47 = [s.constraints[0] for s in algorithm47.result if s.feasible]
SEA_hasil48 = [s.constraints[0] for s in algorithm48.result if s.feasible]
SEA_hasil49 = [s.constraints[0] for s in algorithm49.result if s.feasible]
SEA_hasil50 = [s.constraints[0] for s in algorithm50.result if s.feasible]
SEA_hasil51 = [s.constraints[0] for s in algorithm51.result if s.feasible]
SEA_hasil52 = [s.constraints[0] for s in algorithm52.result if s.feasible]
SEA_hasil53 = [s.constraints[0] for s in algorithm53.result if s.feasible]
SEA_hasil54 = [s.constraints[0] for s in algorithm54.result if s.feasible]
SEA_hasil55 = [s.constraints[0] for s in algorithm55.result if s.feasible]
SEA_hasil56 = [s.constraints[0] for s in algorithm56.result if s.feasible]
SEA_hasil57 = [s.constraints[0] for s in algorithm57.result if s.feasible]
SEA_hasil58 = [s.constraints[0] for s in algorithm58.result if s.feasible]
SEA_hasil59 = [s.constraints[0] for s in algorithm59.result if s.feasible]
SEA_hasil60 = [s.constraints[0] for s in algorithm60.result if s.feasible]
SEA_hasil61 = [s.constraints[0] for s in algorithm61.result if s.feasible]
SEA_hasil62 = [s.constraints[0] for s in algorithm62.result if s.feasible]
SEA_hasil63 = [s.constraints[0] for s in algorithm63.result if s.feasible]
SEA_hasil64 = [s.constraints[0] for s in algorithm64.result if s.feasible]
SEA_hasil65 = [s.constraints[0] for s in algorithm65.result if s.feasible]
SEA_hasil66 = [s.constraints[0] for s in algorithm66.result if s.feasible]
SEA_hasil67 = [s.constraints[0] for s in algorithm67.result if s.feasible]
SEA_hasil68 = [s.constraints[0] for s in algorithm68.result if s.feasible]
SEA_hasil69 = [s.constraints[0] for s in algorithm69.result if s.feasible]
SEA_hasil70 = [s.constraints[0] for s in algorithm70.result if s.feasible]
SEA_hasil71 = [s.constraints[0] for s in algorithm71.result if s.feasible]
SEA_hasil72 = [s.constraints[0] for s in algorithm72.result if s.feasible]
SEA_hasil73 = [s.constraints[0] for s in algorithm73.result if s.feasible]
SEA_hasil74 = [s.constraints[0] for s in algorithm74.result if s.feasible]
SEA_hasil75 = [s.constraints[0] for s in algorithm75.result if s.feasible]
SEA_hasil76 = [s.constraints[0] for s in algorithm76.result if s.feasible]
SEA_hasil77 = [s.constraints[0] for s in algorithm77.result if s.feasible]
SEA_hasil78 = [s.constraints[0] for s in algorithm78.result if s.feasible]
SEA_hasil79 = [s.constraints[0] for s in algorithm79.result if s.feasible]
SEA_hasil80 = [s.constraints[0] for s in algorithm80.result if s.feasible]
SEA_hasil81 = [s.constraints[0] for s in algorithm81.result if s.feasible]
SEA_hasil82 = [s.constraints[0] for s in algorithm82.result if s.feasible]
SEA_hasil83 = [s.constraints[0] for s in algorithm83.result if s.feasible]
SEA_hasil84 = [s.constraints[0] for s in algorithm84.result if s.feasible]
SEA_hasil85 = [s.constraints[0] for s in algorithm85.result if s.feasible]
SEA_hasil86 = [s.constraints[0] for s in algorithm86.result if s.feasible]
SEA_hasil87 = [s.constraints[0] for s in algorithm87.result if s.feasible]
SEA_hasil88 = [s.constraints[0] for s in algorithm88.result if s.feasible]
SEA_hasil89 = [s.constraints[0] for s in algorithm89.result if s.feasible]
SEA_hasil90 = [s.constraints[0] for s in algorithm90.result if s.feasible]
SEA_hasil91 = [s.constraints[0] for s in algorithm91.result if s.feasible]
SEA_hasil92 = [s.constraints[0] for s in algorithm92.result if s.feasible]
SEA_hasil93 = [s.constraints[0] for s in algorithm93.result if s.feasible]
SEA_hasil94 = [s.constraints[0] for s in algorithm94.result if s.feasible]
SEA_hasil95 = [s.constraints[0] for s in algorithm95.result if s.feasible]
SEA_hasil96 = [s.constraints[0] for s in algorithm96.result if s.feasible]
SEA_hasil97 = [s.constraints[0] for s in algorithm97.result if s.feasible]
SEA_hasil98 = [s.constraints[0] for s in algorithm98.result if s.feasible]
SEA_hasil99 = [s.constraints[0] for s in algorithm99.result if s.feasible]
SEA_hasil100 = [s.constraints[0] for s in algorithm100.result if s.feasible]
SEA_hasil101 = [s.constraints[0] for s in algorithm101.result if s.feasible]
SEA_hasil102 = [s.constraints[0] for s in algorithm102.result if s.feasible]
SEA_hasil103 = [s.constraints[0] for s in algorithm103.result if s.feasible]
SEA_hasil104 = [s.constraints[0] for s in algorithm104.result if s.feasible]
SEA_hasil105 = [s.constraints[0] for s in algorithm105.result if s.feasible]
SEA_hasil106 = [s.constraints[0] for s in algorithm106.result if s.feasible]
SEA_hasil107 = [s.constraints[0] for s in algorithm107.result if s.feasible]
SEA_hasil108 = [s.constraints[0] for s in algorithm108.result if s.feasible]

#Create SEA_hasil-i definition for 51 other cases

SEA_hasill=np.hstack((SEA_hasil37, 	SEA_hasil38, 	SEA_hasil39, 	SEA_hasil40, 	SEA_hasil41, 	SEA_hasil42, 	SEA_hasil43, 	SEA_hasil44, 	SEA_hasil45, 	SEA_hasil46, 	SEA_hasil47, 	SEA_hasil48, 	SEA_hasil49, 	SEA_hasil50, 	SEA_hasil51, 	SEA_hasil52, 	SEA_hasil53, 	SEA_hasil54, 	SEA_hasil55, 	SEA_hasil56, 	SEA_hasil57, 	SEA_hasil58, 	SEA_hasil59, 	SEA_hasil60, 	SEA_hasil61, 	SEA_hasil62, 	SEA_hasil63, 	SEA_hasil64, 	SEA_hasil65, 	SEA_hasil66, 	SEA_hasil67, 	SEA_hasil68, 	SEA_hasil69, 	SEA_hasil70, 	SEA_hasil71, 	SEA_hasil72, 	SEA_hasil73, 	SEA_hasil74, 	SEA_hasil75, 	SEA_hasil76, 	SEA_hasil77, 	SEA_hasil78, 	SEA_hasil79, 	SEA_hasil80, 	SEA_hasil81, 	SEA_hasil82, 	SEA_hasil83, 	SEA_hasil84, 	SEA_hasil85, 	SEA_hasil86, 	SEA_hasil87, 	SEA_hasil88, 	SEA_hasil89, 	SEA_hasil90, 	SEA_hasil91, 	SEA_hasil92, 	SEA_hasil93, 	SEA_hasil94, 	SEA_hasil95, 	SEA_hasil96, 	SEA_hasil97, 	SEA_hasil98, 	SEA_hasil99, 	SEA_hasil100, 	SEA_hasil101, 	SEA_hasil102, 	SEA_hasil103, 	SEA_hasil104, 	SEA_hasil105, 	SEA_hasil106, 	SEA_hasil107, 	SEA_hasil108))
#Create SEA_hasil for 51 other cases

print("SEA_hasill = ")
print(SEA_hasill)
Variable37 = np.reshape(np.array([s.variables for s in algorithm37.result if s.feasible]), (-1,15))
Variable38 = np.reshape(np.array([s.variables for s in algorithm38.result if s.feasible]), (-1,15))
Variable39 = np.reshape(np.array([s.variables for s in algorithm39.result if s.feasible]), (-1,15))
Variable40 = np.reshape(np.array([s.variables for s in algorithm40.result if s.feasible]), (-1,15))
Variable41 = np.reshape(np.array([s.variables for s in algorithm41.result if s.feasible]), (-1,15))
Variable42 = np.reshape(np.array([s.variables for s in algorithm42.result if s.feasible]), (-1,15))
Variable43 = np.reshape(np.array([s.variables for s in algorithm43.result if s.feasible]), (-1,15))
Variable44 = np.reshape(np.array([s.variables for s in algorithm44.result if s.feasible]), (-1,15))
Variable45 = np.reshape(np.array([s.variables for s in algorithm45.result if s.feasible]), (-1,15))
Variable46 = np.reshape(np.array([s.variables for s in algorithm46.result if s.feasible]), (-1,15))
Variable47 = np.reshape(np.array([s.variables for s in algorithm47.result if s.feasible]), (-1,15))
Variable48 = np.reshape(np.array([s.variables for s in algorithm48.result if s.feasible]), (-1,15))
Variable49 = np.reshape(np.array([s.variables for s in algorithm49.result if s.feasible]), (-1,15))
Variable50 = np.reshape(np.array([s.variables for s in algorithm50.result if s.feasible]), (-1,15))
Variable51 = np.reshape(np.array([s.variables for s in algorithm51.result if s.feasible]), (-1,15))
Variable52 = np.reshape(np.array([s.variables for s in algorithm52.result if s.feasible]), (-1,15))
Variable53 = np.reshape(np.array([s.variables for s in algorithm53.result if s.feasible]), (-1,15))
Variable54 = np.reshape(np.array([s.variables for s in algorithm54.result if s.feasible]), (-1,15))
Variable55 = np.reshape(np.array([s.variables for s in algorithm55.result if s.feasible]), (-1,15))
Variable56 = np.reshape(np.array([s.variables for s in algorithm56.result if s.feasible]), (-1,15))
Variable57 = np.reshape(np.array([s.variables for s in algorithm57.result if s.feasible]), (-1,15))
Variable58 = np.reshape(np.array([s.variables for s in algorithm58.result if s.feasible]), (-1,15))
Variable59 = np.reshape(np.array([s.variables for s in algorithm59.result if s.feasible]), (-1,15))
Variable60 = np.reshape(np.array([s.variables for s in algorithm60.result if s.feasible]), (-1,15))
Variable61 = np.reshape(np.array([s.variables for s in algorithm61.result if s.feasible]), (-1,15))
Variable62 = np.reshape(np.array([s.variables for s in algorithm62.result if s.feasible]), (-1,15))
Variable63 = np.reshape(np.array([s.variables for s in algorithm63.result if s.feasible]), (-1,15))
Variable64 = np.reshape(np.array([s.variables for s in algorithm64.result if s.feasible]), (-1,15))
Variable65 = np.reshape(np.array([s.variables for s in algorithm65.result if s.feasible]), (-1,15))
Variable66 = np.reshape(np.array([s.variables for s in algorithm66.result if s.feasible]), (-1,15))
Variable67 = np.reshape(np.array([s.variables for s in algorithm67.result if s.feasible]), (-1,15))
Variable68 = np.reshape(np.array([s.variables for s in algorithm68.result if s.feasible]), (-1,15))
Variable69 = np.reshape(np.array([s.variables for s in algorithm69.result if s.feasible]), (-1,15))
Variable70 = np.reshape(np.array([s.variables for s in algorithm70.result if s.feasible]), (-1,15))
Variable71 = np.reshape(np.array([s.variables for s in algorithm71.result if s.feasible]), (-1,15))
Variable72 = np.reshape(np.array([s.variables for s in algorithm72.result if s.feasible]), (-1,15))
Variable73 = np.reshape(np.array([s.variables for s in algorithm73.result if s.feasible]), (-1,15))
Variable74 = np.reshape(np.array([s.variables for s in algorithm74.result if s.feasible]), (-1,15))
Variable75 = np.reshape(np.array([s.variables for s in algorithm75.result if s.feasible]), (-1,15))
Variable76 = np.reshape(np.array([s.variables for s in algorithm76.result if s.feasible]), (-1,15))
Variable77 = np.reshape(np.array([s.variables for s in algorithm77.result if s.feasible]), (-1,15))
Variable78 = np.reshape(np.array([s.variables for s in algorithm78.result if s.feasible]), (-1,15))
Variable79 = np.reshape(np.array([s.variables for s in algorithm79.result if s.feasible]), (-1,15))
Variable80 = np.reshape(np.array([s.variables for s in algorithm80.result if s.feasible]), (-1,15))
Variable81 = np.reshape(np.array([s.variables for s in algorithm81.result if s.feasible]), (-1,15))
Variable82 = np.reshape(np.array([s.variables for s in algorithm82.result if s.feasible]), (-1,15))
Variable83 = np.reshape(np.array([s.variables for s in algorithm83.result if s.feasible]), (-1,15))
Variable84 = np.reshape(np.array([s.variables for s in algorithm84.result if s.feasible]), (-1,15))
Variable85 = np.reshape(np.array([s.variables for s in algorithm85.result if s.feasible]), (-1,15))
Variable86 = np.reshape(np.array([s.variables for s in algorithm86.result if s.feasible]), (-1,15))
Variable87 = np.reshape(np.array([s.variables for s in algorithm87.result if s.feasible]), (-1,15))
Variable88 = np.reshape(np.array([s.variables for s in algorithm88.result if s.feasible]), (-1,15))
Variable89 = np.reshape(np.array([s.variables for s in algorithm89.result if s.feasible]), (-1,15))
Variable90 = np.reshape(np.array([s.variables for s in algorithm90.result if s.feasible]), (-1,15))
Variable91 = np.reshape(np.array([s.variables for s in algorithm91.result if s.feasible]), (-1,15))
Variable92 = np.reshape(np.array([s.variables for s in algorithm92.result if s.feasible]), (-1,15))
Variable93 = np.reshape(np.array([s.variables for s in algorithm93.result if s.feasible]), (-1,15))
Variable94 = np.reshape(np.array([s.variables for s in algorithm94.result if s.feasible]), (-1,15))
Variable95 = np.reshape(np.array([s.variables for s in algorithm95.result if s.feasible]), (-1,15))
Variable96 = np.reshape(np.array([s.variables for s in algorithm96.result if s.feasible]), (-1,15))
Variable97 = np.reshape(np.array([s.variables for s in algorithm97.result if s.feasible]), (-1,15))
Variable98 = np.reshape(np.array([s.variables for s in algorithm98.result if s.feasible]), (-1,15))
Variable99 = np.reshape(np.array([s.variables for s in algorithm99.result if s.feasible]), (-1,15))
Variable100 = np.reshape(np.array([s.variables for s in algorithm100.result if s.feasible]), (-1,15))
Variable101 = np.reshape(np.array([s.variables for s in algorithm101.result if s.feasible]), (-1,15))
Variable102 = np.reshape(np.array([s.variables for s in algorithm102.result if s.feasible]), (-1,15))
Variable103 = np.reshape(np.array([s.variables for s in algorithm103.result if s.feasible]), (-1,15))
Variable104 = np.reshape(np.array([s.variables for s in algorithm104.result if s.feasible]), (-1,15))
Variable105 = np.reshape(np.array([s.variables for s in algorithm105.result if s.feasible]), (-1,15))
Variable106 = np.reshape(np.array([s.variables for s in algorithm106.result if s.feasible]), (-1,15))
Variable107 = np.reshape(np.array([s.variables for s in algorithm107.result if s.feasible]), (-1,15))
Variable108 = np.reshape(np.array([s.variables for s in algorithm108.result if s.feasible]), (-1,15))


#Create Variable-i definition for 51 other cases

Variables = np.vstack((Variable37, 	Variable38, 	Variable39, 	Variable40, 	Variable41, 	Variable42, 	Variable43, 	Variable44, 	Variable45, 	Variable46, 	Variable47, 	Variable48, 	Variable49, 	Variable50, 	Variable51, 	Variable52, 	Variable53, 	Variable54, 	Variable55, 	Variable56, 	Variable57, 	Variable58, 	Variable59, 	Variable60, 	Variable61, 	Variable62, 	Variable63, 	Variable64, 	Variable65, 	Variable66, 	Variable67, 	Variable68, 	Variable69, 	Variable70, 	Variable71, 	Variable72, 	Variable73, 	Variable74, 	Variable75, 	Variable76, 	Variable77, 	Variable78, 	Variable79, 	Variable80, 	Variable81, 	Variable82, 	Variable83, 	Variable84, 	Variable85, 	Variable86, 	Variable87, 	Variable88, 	Variable89, 	Variable90, 	Variable91, 	Variable92, 	Variable93, 	Variable94, 	Variable95, 	Variable96, 	Variable97, 	Variable98, 	Variable99, 	Variable100, 	Variable101, 	Variable102, 	Variable103, 	Variable104, 	Variable105, 	Variable106, 	Variable107, 	Variable108))
#Create Variable-i definition for 51 other cases

print("Variables = ")
print(Variables)
solutionwrite = pd.ExcelWriter('NSGA Solutions2.xlsx', engine='openpyxl')
pd.DataFrame(SEA_hasill).to_excel(solutionwrite, sheet_name="SEA")
pd.DataFrame(Variables).to_excel(solutionwrite, sheet_name="Variables")
solutionwrite._save()

x = SEA_hasill
y = SEA_hasill
SEA_Hasil = np.array([SEA_hasill]).reshape((-1,1))+np.array(0.006)
SEA_Hasil_inverse = min_max_scaler.inverse_transform(SEA_Hasil)
plt.figure(1)
plt.scatter(SEA_Hasil_inverse,SEA_Hasil_inverse)
plt.xlabel('$f_1(SEA)$')
plt.ylabel('$f_2(SEA)$')
plt.show()















