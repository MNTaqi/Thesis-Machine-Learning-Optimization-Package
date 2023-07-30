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

#Case109: Susunan 5 Domex Protect 500 CNC
problem109 = Problem(15, 1, 16)
problem109.types[:] = variable_types
problem109.constraints[0] = ">=0" #SEA
problem109.constraints[1] = "==0" #Susunan 2
problem109.constraints[2] = "==0" #Susunan 3
problem109.constraints[3] = "==0" #Susunan 4
problem109.constraints[4] = "==1" #Susunan 5
problem109.constraints[5] = "==0" #Susunan 6
problem109.constraints[6] = "==0" #Weldox 700E
problem109.constraints[7] = "==0" #Hardox 400
problem109.constraints[8] = "==1" #Domex Protect 500
problem109.constraints[9] = "==0" #Armox 560T
problem109.constraints[10] = "==0" #Al7075-T651
problem109.constraints[11] = "==0" #Kevlar-29
problem109.constraints[12] = "==0" #S2 Glass/SC15
problem109.constraints[13] = "==0" #CFRP
problem109.constraints[14] = ">=0.5" #thickness
problem109.constraints[15] = ">=1" #lapisan
problem109.directions[0] = Problem.MAXIMIZE
problem109.function = Opt

#Case110: Susunan 5 Domex Protect 500 Kevlar-29
problem110 = Problem(15, 1, 16)
problem110.types[:] = variable_types
problem110.constraints[0] = ">=0" #SEA
problem110.constraints[1] = "==0" #Susunan 2
problem110.constraints[2] = "==0" #Susunan 3
problem110.constraints[3] = "==0" #Susunan 4
problem110.constraints[4] = "==1" #Susunan 5
problem110.constraints[5] = "==0" #Susunan 6
problem110.constraints[6] = "==0" #Weldox 700E
problem110.constraints[7] = "==0" #Hardox 400
problem110.constraints[8] = "==1" #Domex Protect 500
problem110.constraints[9] = "==0" #Armox 560T
problem110.constraints[10] = "==0" #Al7075-T651
problem110.constraints[11] = "==1" #Kevlar-29
problem110.constraints[12] = "==0" #S2 Glass/SC15
problem110.constraints[13] = "==0" #CFRP
problem110.constraints[14] = ">=0.5" #thickness
problem110.constraints[15] = ">=1" #lapisan
problem110.directions[0] = Problem.MAXIMIZE
problem110.function = Opt

#Case111: Susunan 5 Domex Protect 500 S2Glass/SC15
problem111 = Problem(15, 1, 16)
problem111.types[:] = variable_types
problem111.constraints[0] = ">=0" #SEA
problem111.constraints[1] = "==0" #Susunan 2
problem111.constraints[2] = "==0" #Susunan 3
problem111.constraints[3] = "==0" #Susunan 4
problem111.constraints[4] = "==1" #Susunan 5
problem111.constraints[5] = "==0" #Susunan 6
problem111.constraints[6] = "==0" #Weldox 700E
problem111.constraints[7] = "==0" #Hardox 400
problem111.constraints[8] = "==1" #Domex Protect 500
problem111.constraints[9] = "==0" #Armox 560T
problem111.constraints[10] = "==0" #Al7075-T651
problem111.constraints[11] = "==0" #Kevlar-29
problem111.constraints[12] = "==1" #S2 Glass/SC15
problem111.constraints[13] = "==0" #CFRP
problem111.constraints[14] = ">=0.5" #thickness
problem111.constraints[15] = ">=1" #lapisan
problem111.directions[0] = Problem.MAXIMIZE
problem111.function = Opt

#Case112: Susunan 5 Domex Protect 500 CFRP
problem112 = Problem(15, 1, 16)
problem112.types[:] = variable_types
problem112.constraints[0] = ">=0" #SEA
problem112.constraints[1] = "==0" #Susunan 2
problem112.constraints[2] = "==0" #Susunan 3
problem112.constraints[3] = "==0" #Susunan 4
problem112.constraints[4] = "==1" #Susunan 5
problem112.constraints[5] = "==0" #Susunan 6
problem112.constraints[6] = "==0" #Weldox 700E
problem112.constraints[7] = "==0" #Hardox 400
problem112.constraints[8] = "==1" #Domex Protect 500
problem112.constraints[9] = "==0" #Armox 560T
problem112.constraints[10] = "==0" #Al7075-T651
problem112.constraints[11] = "==0" #Kevlar-29
problem112.constraints[12] = "==0" #S2 Glass/SC15
problem112.constraints[13] = "==1" #CFRP
problem112.constraints[14] = ">=0.5" #thickness
problem112.constraints[15] = ">=1" #lapisan
problem112.directions[0] = Problem.MAXIMIZE
problem112.function = Opt

#Case113: Susunan 5 Armox 560T CNC
problem113 = Problem(15, 1, 16)
problem113.types[:] = variable_types
problem113.constraints[0] = ">=0" #SEA
problem113.constraints[1] = "==0" #Susunan 2
problem113.constraints[2] = "==0" #Susunan 3
problem113.constraints[3] = "==0" #Susunan 4
problem113.constraints[4] = "==1" #Susunan 5
problem113.constraints[5] = "==0" #Susunan 6
problem113.constraints[6] = "==0" #Weldox 700E
problem113.constraints[7] = "==0" #Hardox 400
problem113.constraints[8] = "==0" #Domex Protect 500
problem113.constraints[9] = "==1" #Armox 560T
problem113.constraints[10] = "==0" #Al7075-T651
problem113.constraints[11] = "==0" #Kevlar-29
problem113.constraints[12] = "==0" #S2 Glass/SC15
problem113.constraints[13] = "==0" #CFRP
problem113.constraints[14] = ">=0.5" #thickness
problem113.constraints[15] = ">=1" #lapisan
problem113.directions[0] = Problem.MAXIMIZE
problem113.function = Opt

#Case114: Susunan 5 Armox 560T Kevlar-29
problem114 = Problem(15, 1, 16)
problem114.types[:] = variable_types
problem114.constraints[0] = ">=0" #SEA
problem114.constraints[1] = "==0" #Susunan 2
problem114.constraints[2] = "==0" #Susunan 3
problem114.constraints[3] = "==0" #Susunan 4
problem114.constraints[4] = "==1" #Susunan 5
problem114.constraints[5] = "==0" #Susunan 6
problem114.constraints[6] = "==0" #Weldox 700E
problem114.constraints[7] = "==0" #Hardox 400
problem114.constraints[8] = "==0" #Domex Protect 500
problem114.constraints[9] = "==1" #Armox 560T
problem114.constraints[10] = "==0" #Al7075-T651
problem114.constraints[11] = "==1" #Kevlar-29
problem114.constraints[12] = "==0" #S2 Glass/SC15
problem114.constraints[13] = "==0" #CFRP
problem114.constraints[14] = ">=0.5" #thickness
problem114.constraints[15] = ">=1" #lapisan
problem114.directions[0] = Problem.MAXIMIZE
problem114.function = Opt

#Case115: Susunan 5 Armox 560T S2Glass/SC15
problem115 = Problem(15, 1, 16)
problem115.types[:] = variable_types
problem115.constraints[0] = ">=0" #SEA
problem115.constraints[1] = "==0" #Susunan 2
problem115.constraints[2] = "==0" #Susunan 3
problem115.constraints[3] = "==0" #Susunan 4
problem115.constraints[4] = "==1" #Susunan 5
problem115.constraints[5] = "==0" #Susunan 6
problem115.constraints[6] = "==0" #Weldox 700E
problem115.constraints[7] = "==0" #Hardox 400
problem115.constraints[8] = "==0" #Domex Protect 500
problem115.constraints[9] = "==1" #Armox 560T
problem115.constraints[10] = "==0" #Al7075-T651
problem115.constraints[11] = "==0" #Kevlar-29
problem115.constraints[12] = "==1" #S2 Glass/SC15
problem115.constraints[13] = "==0" #CFRP
problem115.constraints[14] = ">=0.5" #thickness
problem115.constraints[15] = ">=1" #lapisan
problem115.directions[0] = Problem.MAXIMIZE
problem115.function = Opt

#Case116: Susunan 5 Armox 560T CFRP
problem116 = Problem(15, 1, 16)
problem116.types[:] = variable_types
problem116.constraints[0] = ">=0" #SEA
problem116.constraints[1] = "==0" #Susunan 2
problem116.constraints[2] = "==0" #Susunan 3
problem116.constraints[3] = "==0" #Susunan 4
problem116.constraints[4] = "==1" #Susunan 5
problem116.constraints[5] = "==0" #Susunan 6
problem116.constraints[6] = "==0" #Weldox 700E
problem116.constraints[7] = "==0" #Hardox 400
problem116.constraints[8] = "==0" #Domex Protect 500
problem116.constraints[9] = "==1" #Armox 560T
problem116.constraints[10] = "==0" #Al7075-T651
problem116.constraints[11] = "==0" #Kevlar-29
problem116.constraints[12] = "==0" #S2 Glass/SC15
problem116.constraints[13] = "==1" #CFRP
problem116.constraints[14] = ">=0.5" #thickness
problem116.constraints[15] = ">=1" #lapisan
problem116.directions[0] = Problem.MAXIMIZE
problem116.function = Opt

#Case117: Susunan 5 Al7075-T651 CNC
problem117 = Problem(15, 1, 16)
problem117.types[:] = variable_types
problem117.constraints[0] = ">=0" #SEA
problem117.constraints[1] = "==0" #Susunan 2
problem117.constraints[2] = "==0" #Susunan 3
problem117.constraints[3] = "==0" #Susunan 4
problem117.constraints[4] = "==1" #Susunan 5
problem117.constraints[5] = "==0" #Susunan 6
problem117.constraints[6] = "==0" #Weldox 700E
problem117.constraints[7] = "==0" #Hardox 400
problem117.constraints[8] = "==0" #Domex Protect 500
problem117.constraints[9] = "==0" #Armox 560T
problem117.constraints[10] = "==1" #Al7075-T651
problem117.constraints[11] = "==0" #Kevlar-29
problem117.constraints[12] = "==0" #S2 Glass/SC15
problem117.constraints[13] = "==0" #CFRP
problem117.constraints[14] = ">=0.5" #thickness
problem117.constraints[15] = ">=1" #lapisan
problem117.directions[0] = Problem.MAXIMIZE
problem117.function = Opt

#Case118: Susunan 5 Al7075-T651 Kevlar-29
problem118 = Problem(15, 1, 16)
problem118.types[:] = variable_types
problem118.constraints[0] = ">=0" #SEA
problem118.constraints[1] = "==0" #Susunan 2
problem118.constraints[2] = "==0" #Susunan 3
problem118.constraints[3] = "==0" #Susunan 4
problem118.constraints[4] = "==1" #Susunan 5
problem118.constraints[5] = "==0" #Susunan 6
problem118.constraints[6] = "==0" #Weldox 700E
problem118.constraints[7] = "==0" #Hardox 400
problem118.constraints[8] = "==0" #Domex Protect 500
problem118.constraints[9] = "==0" #Armox 560T
problem118.constraints[10] = "==1" #Al7075-T651
problem118.constraints[11] = "==1" #Kevlar-29
problem118.constraints[12] = "==0" #S2 Glass/SC15
problem118.constraints[13] = "==0" #CFRP
problem118.constraints[14] = ">=0.5" #thickness
problem118.constraints[15] = ">=1" #lapisan
problem118.directions[0] = Problem.MAXIMIZE
problem118.function = Opt

#Case119: Susunan 5 Al7075-T651 S2Glass/SC15
problem119 = Problem(15, 1, 16)
problem119.types[:] = variable_types
problem119.constraints[0] = ">=0" #SEA
problem119.constraints[1] = "==0" #Susunan 2
problem119.constraints[2] = "==0" #Susunan 3
problem119.constraints[3] = "==0" #Susunan 4
problem119.constraints[4] = "==1" #Susunan 5
problem119.constraints[5] = "==0" #Susunan 6
problem119.constraints[6] = "==0" #Weldox 700E
problem119.constraints[7] = "==0" #Hardox 400
problem119.constraints[8] = "==0" #Domex Protect 500
problem119.constraints[9] = "==0" #Armox 560T
problem119.constraints[10] = "==1" #Al7075-T651
problem119.constraints[11] = "==0" #Kevlar-29
problem119.constraints[12] = "==1" #S2 Glass/SC15
problem119.constraints[13] = "==0" #CFRP
problem119.constraints[14] = ">=0.5" #thickness
problem119.constraints[15] = ">=1" #lapisan
problem119.directions[0] = Problem.MAXIMIZE
problem119.function = Opt

#Case120: Susunan 5 Al7075-T651 CFRP
problem120 = Problem(15, 1, 16)
problem120.types[:] = variable_types
problem120.constraints[0] = ">=0" #SEA
problem120.constraints[1] = "==0" #Susunan 2
problem120.constraints[2] = "==0" #Susunan 3
problem120.constraints[3] = "==0" #Susunan 4
problem120.constraints[4] = "==1" #Susunan 5
problem120.constraints[5] = "==0" #Susunan 6
problem120.constraints[6] = "==0" #Weldox 700E
problem120.constraints[7] = "==0" #Hardox 400
problem120.constraints[8] = "==0" #Domex Protect 500
problem120.constraints[9] = "==0" #Armox 560T
problem120.constraints[10] = "==1" #Al7075-T651
problem120.constraints[11] = "==0" #Kevlar-29
problem120.constraints[12] = "==0" #S2 Glass/SC15
problem120.constraints[13] = "==1" #CFRP
problem120.constraints[14] = ">=0.5" #thickness
problem120.constraints[15] = ">=1" #lapisan
problem120.directions[0] = Problem.MAXIMIZE
problem120.function = Opt

#Case121: Susunan 6 Weldox 500E CNC
problem121 = Problem(15, 1, 16)
problem121.types[:] = variable_types
problem121.constraints[0] = ">=0" #SEA
problem121.constraints[1] = "==0" #Susunan 2
problem121.constraints[2] = "==0" #Susunan 3
problem121.constraints[3] = "==0" #Susunan 4
problem121.constraints[4] = "==0" #Susunan 5
problem121.constraints[5] = "==1" #Susunan 6
problem121.constraints[6] = "==0" #Weldox 700E
problem121.constraints[7] = "==0" #Hardox 400
problem121.constraints[8] = "==0" #Domex Protect 500
problem121.constraints[9] = "==0" #Armox 560T
problem121.constraints[10] = "==0" #Al7075-T651
problem121.constraints[11] = "==0" #Kevlar-29
problem121.constraints[12] = "==0" #S2 Glass/SC15
problem121.constraints[13] = "==0" #CFRP
problem121.constraints[14] = ">=0.5" #thickness
problem121.constraints[15] = ">=1" #lapisan
problem121.directions[0] = Problem.MAXIMIZE
problem121.function = Opt

#Case122: Susunan 6 Weldox 500E Kevlar-29
problem122 = Problem(15, 1, 16)
problem122.types[:] = variable_types
problem122.constraints[0] = ">=0" #SEA
problem122.constraints[1] = "==0" #Susunan 2
problem122.constraints[2] = "==0" #Susunan 3
problem122.constraints[3] = "==0" #Susunan 4
problem122.constraints[4] = "==0" #Susunan 5
problem122.constraints[5] = "==1" #Susunan 6
problem122.constraints[6] = "==0" #Weldox 700E
problem122.constraints[7] = "==0" #Hardox 400
problem122.constraints[8] = "==0" #Domex Protect 500
problem122.constraints[9] = "==0" #Armox 560T
problem122.constraints[10] = "==0" #Al7075-T651
problem122.constraints[11] = "==1" #Kevlar-29
problem122.constraints[12] = "==0" #S2 Glass/SC15
problem122.constraints[13] = "==0" #CFRP
problem122.constraints[14] = ">=0.5" #thickness
problem122.constraints[15] = ">=1" #lapisan
problem122.directions[0] = Problem.MAXIMIZE
problem122.function = Opt

#Case123: Susunan 6 Weldox 500E S2Glass/SC15
problem123 = Problem(15, 1, 16)
problem123.types[:] = variable_types
problem123.constraints[0] = ">=0" #SEA
problem123.constraints[1] = "==0" #Susunan 2
problem123.constraints[2] = "==0" #Susunan 3
problem123.constraints[3] = "==0" #Susunan 4
problem123.constraints[4] = "==0" #Susunan 5
problem123.constraints[5] = "==1" #Susunan 6
problem123.constraints[6] = "==0" #Weldox 700E
problem123.constraints[7] = "==0" #Hardox 400
problem123.constraints[8] = "==0" #Domex Protect 500
problem123.constraints[9] = "==0" #Armox 560T
problem123.constraints[10] = "==0" #Al7075-T651
problem123.constraints[11] = "==0" #Kevlar-29
problem123.constraints[12] = "==1" #S2 Glass/SC15
problem123.constraints[13] = "==0" #CFRP
problem123.constraints[14] = ">=0.5" #thickness
problem123.constraints[15] = ">=1" #lapisan
problem123.directions[0] = Problem.MAXIMIZE
problem123.function = Opt

#Case124: Susunan 6 Weldox 500E CFRP
problem124 = Problem(15, 1, 16)
problem124.types[:] = variable_types
problem124.constraints[0] = ">=0" #SEA
problem124.constraints[1] = "==0" #Susunan 2
problem124.constraints[2] = "==0" #Susunan 3
problem124.constraints[3] = "==0" #Susunan 4
problem124.constraints[4] = "==0" #Susunan 5
problem124.constraints[5] = "==1" #Susunan 6
problem124.constraints[6] = "==0" #Weldox 700E
problem124.constraints[7] = "==0" #Hardox 400
problem124.constraints[8] = "==0" #Domex Protect 500
problem124.constraints[9] = "==0" #Armox 560T
problem124.constraints[10] = "==0" #Al7075-T651
problem124.constraints[11] = "==0" #Kevlar-29
problem124.constraints[12] = "==0" #S2 Glass/SC15
problem124.constraints[13] = "==1" #CFRP
problem124.constraints[14] = ">=0.5" #thickness
problem124.constraints[15] = ">=1" #lapisan
problem124.directions[0] = Problem.MAXIMIZE
problem124.function = Opt

#Case125: Susunan 6 Weldox 700E CNC
problem125 = Problem(15, 1, 16)
problem125.types[:] = variable_types
problem125.constraints[0] = ">=0" #SEA
problem125.constraints[1] = "==0" #Susunan 2
problem125.constraints[2] = "==0" #Susunan 3
problem125.constraints[3] = "==0" #Susunan 4
problem125.constraints[4] = "==0" #Susunan 5
problem125.constraints[5] = "==1" #Susunan 6
problem125.constraints[6] = "==1" #Weldox 700E
problem125.constraints[7] = "==0" #Hardox 400
problem125.constraints[8] = "==0" #Domex Protect 500
problem125.constraints[9] = "==0" #Armox 560T
problem125.constraints[10] = "==0" #Al7075-T651
problem125.constraints[11] = "==0" #Kevlar-29
problem125.constraints[12] = "==0" #S2 Glass/SC15
problem125.constraints[13] = "==0" #CFRP
problem125.constraints[14] = ">=0.5" #thickness
problem125.constraints[15] = ">=1" #lapisan
problem125.directions[0] = Problem.MAXIMIZE
problem125.function = Opt

#Case126: Susunan 6 Weldox 700E Kevlar-29
problem126 = Problem(15, 1, 16)
problem126.types[:] = variable_types
problem126.constraints[0] = ">=0" #SEA
problem126.constraints[1] = "==0" #Susunan 2
problem126.constraints[2] = "==0" #Susunan 3
problem126.constraints[3] = "==0" #Susunan 4
problem126.constraints[4] = "==0" #Susunan 5
problem126.constraints[5] = "==1" #Susunan 6
problem126.constraints[6] = "==1" #Weldox 700E
problem126.constraints[7] = "==0" #Hardox 400
problem126.constraints[8] = "==0" #Domex Protect 500
problem126.constraints[9] = "==0" #Armox 560T
problem126.constraints[10] = "==0" #Al7075-T651
problem126.constraints[11] = "==1" #Kevlar-29
problem126.constraints[12] = "==0" #S2 Glass/SC15
problem126.constraints[13] = "==0" #CFRP
problem126.constraints[14] = ">=0.5" #thickness
problem126.constraints[15] = ">=1" #lapisan
problem126.directions[0] = Problem.MAXIMIZE
problem126.function = Opt

#Case127: Susunan 6 Weldox 700E S2Glass/SC15
problem127 = Problem(15, 1, 16)
problem127.types[:] = variable_types
problem127.constraints[0] = ">=0" #SEA
problem127.constraints[1] = "==0" #Susunan 2
problem127.constraints[2] = "==0" #Susunan 3
problem127.constraints[3] = "==0" #Susunan 4
problem127.constraints[4] = "==0" #Susunan 5
problem127.constraints[5] = "==1" #Susunan 6
problem127.constraints[6] = "==1" #Weldox 700E
problem127.constraints[7] = "==0" #Hardox 400
problem127.constraints[8] = "==0" #Domex Protect 500
problem127.constraints[9] = "==0" #Armox 560T
problem127.constraints[10] = "==0" #Al7075-T651
problem127.constraints[11] = "==0" #Kevlar-29
problem127.constraints[12] = "==1" #S2 Glass/SC15
problem127.constraints[13] = "==0" #CFRP
problem127.constraints[14] = ">=0.5" #thickness
problem127.constraints[15] = ">=1" #lapisan
problem127.directions[0] = Problem.MAXIMIZE
problem127.function = Opt

#Case128: Susunan 6 Weldox 700E CFRP
problem128 = Problem(15, 1, 16)
problem128.types[:] = variable_types
problem128.constraints[0] = ">=0" #SEA
problem128.constraints[1] = "==0" #Susunan 2
problem128.constraints[2] = "==0" #Susunan 3
problem128.constraints[3] = "==0" #Susunan 4
problem128.constraints[4] = "==0" #Susunan 5
problem128.constraints[5] = "==1" #Susunan 6
problem128.constraints[6] = "==1" #Weldox 700E
problem128.constraints[7] = "==0" #Hardox 400
problem128.constraints[8] = "==0" #Domex Protect 500
problem128.constraints[9] = "==0" #Armox 560T
problem128.constraints[10] = "==0" #Al7075-T651
problem128.constraints[11] = "==0" #Kevlar-29
problem128.constraints[12] = "==0" #S2 Glass/SC15
problem128.constraints[13] = "==1" #CFRP
problem128.constraints[14] = ">=0.5" #thickness
problem128.constraints[15] = ">=1" #lapisan
problem128.directions[0] = Problem.MAXIMIZE
problem128.function = Opt

#Case129: Susunan 6 Hardox 400 CNC
problem129 = Problem(15, 1, 16)
problem129.types[:] = variable_types
problem129.constraints[0] = ">=0" #SEA
problem129.constraints[1] = "==0" #Susunan 2
problem129.constraints[2] = "==0" #Susunan 3
problem129.constraints[3] = "==0" #Susunan 4
problem129.constraints[4] = "==0" #Susunan 5
problem129.constraints[5] = "==1" #Susunan 6
problem129.constraints[6] = "==0" #Weldox 700E
problem129.constraints[7] = "==1" #Hardox 400
problem129.constraints[8] = "==0" #Domex Protect 500
problem129.constraints[9] = "==0" #Armox 560T
problem129.constraints[10] = "==0" #Al7075-T651
problem129.constraints[11] = "==0" #Kevlar-29
problem129.constraints[12] = "==0" #S2 Glass/SC15
problem129.constraints[13] = "==0" #CFRP
problem129.constraints[14] = ">=0.5" #thickness
problem129.constraints[15] = ">=1" #lapisan
problem129.directions[0] = Problem.MAXIMIZE
problem129.function = Opt

#Case130: Susunan 6 Hardox 400 Kevlar-29
problem130 = Problem(15, 1, 16)
problem130.types[:] = variable_types
problem130.constraints[0] = ">=0" #SEA
problem130.constraints[1] = "==0" #Susunan 2
problem130.constraints[2] = "==0" #Susunan 3
problem130.constraints[3] = "==0" #Susunan 4
problem130.constraints[4] = "==0" #Susunan 5
problem130.constraints[5] = "==1" #Susunan 6
problem130.constraints[6] = "==0" #Weldox 700E
problem130.constraints[7] = "==1" #Hardox 400
problem130.constraints[8] = "==0" #Domex Protect 500
problem130.constraints[9] = "==0" #Armox 560T
problem130.constraints[10] = "==0" #Al7075-T651
problem130.constraints[11] = "==1" #Kevlar-29
problem130.constraints[12] = "==0" #S2 Glass/SC15
problem130.constraints[13] = "==0" #CFRP
problem130.constraints[14] = ">=0.5" #thickness
problem130.constraints[15] = ">=1" #lapisan
problem130.directions[0] = Problem.MAXIMIZE
problem130.function = Opt

#Case131: Susunan 6 Hardox 400 S2Glass/SC15
problem131 = Problem(15, 1, 16)
problem131.types[:] = variable_types
problem131.constraints[0] = ">=0" #SEA
problem131.constraints[1] = "==0" #Susunan 2
problem131.constraints[2] = "==0" #Susunan 3
problem131.constraints[3] = "==0" #Susunan 4
problem131.constraints[4] = "==0" #Susunan 5
problem131.constraints[5] = "==1" #Susunan 6
problem131.constraints[6] = "==0" #Weldox 700E
problem131.constraints[7] = "==1" #Hardox 400
problem131.constraints[8] = "==0" #Domex Protect 500
problem131.constraints[9] = "==0" #Armox 560T
problem131.constraints[10] = "==0" #Al7075-T651
problem131.constraints[11] = "==0" #Kevlar-29
problem131.constraints[12] = "==1" #S2 Glass/SC15
problem131.constraints[13] = "==0" #CFRP
problem131.constraints[14] = ">=0.5" #thickness
problem131.constraints[15] = ">=1" #lapisan
problem131.directions[0] = Problem.MAXIMIZE
problem131.function = Opt

#Case132: Susunan 6 Hardox 400 CFRP
problem132 = Problem(15, 1, 16)
problem132.types[:] = variable_types
problem132.constraints[0] = ">=0" #SEA
problem132.constraints[1] = "==0" #Susunan 2
problem132.constraints[2] = "==0" #Susunan 3
problem132.constraints[3] = "==0" #Susunan 4
problem132.constraints[4] = "==0" #Susunan 5
problem132.constraints[5] = "==1" #Susunan 6
problem132.constraints[6] = "==0" #Weldox 700E
problem132.constraints[7] = "==1" #Hardox 400
problem132.constraints[8] = "==0" #Domex Protect 500
problem132.constraints[9] = "==0" #Armox 560T
problem132.constraints[10] = "==0" #Al7075-T651
problem132.constraints[11] = "==0" #Kevlar-29
problem132.constraints[12] = "==0" #S2 Glass/SC15
problem132.constraints[13] = "==1" #CFRP
problem132.constraints[14] = ">=0.5" #thickness
problem132.constraints[15] = ">=1" #lapisan
problem132.directions[0] = Problem.MAXIMIZE
problem132.function = Opt

#Case133: Susunan 6 Domex Protect 500 CNC
problem133 = Problem(15, 1, 16)
problem133.types[:] = variable_types
problem133.constraints[0] = ">=0" #SEA
problem133.constraints[1] = "==0" #Susunan 2
problem133.constraints[2] = "==0" #Susunan 3
problem133.constraints[3] = "==0" #Susunan 4
problem133.constraints[4] = "==0" #Susunan 5
problem133.constraints[5] = "==1" #Susunan 6
problem133.constraints[6] = "==0" #Weldox 700E
problem133.constraints[7] = "==0" #Hardox 400
problem133.constraints[8] = "==1" #Domex Protect 500
problem133.constraints[9] = "==0" #Armox 560T
problem133.constraints[10] = "==0" #Al7075-T651
problem133.constraints[11] = "==0" #Kevlar-29
problem133.constraints[12] = "==0" #S2 Glass/SC15
problem133.constraints[13] = "==0" #CFRP
problem133.constraints[14] = ">=0.5" #thickness
problem133.constraints[15] = ">=1" #lapisan
problem133.directions[0] = Problem.MAXIMIZE
problem133.function = Opt

#Case134: Susunan 6 Domex Protect 500 Kevlar-29
problem134 = Problem(15, 1, 16)
problem134.types[:] = variable_types
problem134.constraints[0] = ">=0" #SEA
problem134.constraints[1] = "==0" #Susunan 2
problem134.constraints[2] = "==0" #Susunan 3
problem134.constraints[3] = "==0" #Susunan 4
problem134.constraints[4] = "==0" #Susunan 5
problem134.constraints[5] = "==1" #Susunan 6
problem134.constraints[6] = "==0" #Weldox 700E
problem134.constraints[7] = "==0" #Hardox 400
problem134.constraints[8] = "==1" #Domex Protect 500
problem134.constraints[9] = "==0" #Armox 560T
problem134.constraints[10] = "==0" #Al7075-T651
problem134.constraints[11] = "==1" #Kevlar-29
problem134.constraints[12] = "==0" #S2 Glass/SC15
problem134.constraints[13] = "==0" #CFRP
problem134.constraints[14] = ">=0.5" #thickness
problem134.constraints[15] = ">=1" #lapisan
problem134.directions[0] = Problem.MAXIMIZE
problem134.function = Opt

#Case135: Susunan 6 Domex Protect 500 S2Glass/SC15
problem135 = Problem(15, 1, 16)
problem135.types[:] = variable_types
problem135.constraints[0] = ">=0" #SEA
problem135.constraints[1] = "==0" #Susunan 2
problem135.constraints[2] = "==0" #Susunan 3
problem135.constraints[3] = "==0" #Susunan 4
problem135.constraints[4] = "==0" #Susunan 5
problem135.constraints[5] = "==1" #Susunan 6
problem135.constraints[6] = "==0" #Weldox 700E
problem135.constraints[7] = "==0" #Hardox 400
problem135.constraints[8] = "==1" #Domex Protect 500
problem135.constraints[9] = "==0" #Armox 560T
problem135.constraints[10] = "==0" #Al7075-T651
problem135.constraints[11] = "==0" #Kevlar-29
problem135.constraints[12] = "==1" #S2 Glass/SC15
problem135.constraints[13] = "==0" #CFRP
problem135.constraints[14] = ">=0.5" #thickness
problem135.constraints[15] = ">=1" #lapisan
problem135.directions[0] = Problem.MAXIMIZE
problem135.function = Opt

#Case136: Susunan 6 Domex Protect 500 CFRP
problem136 = Problem(15, 1, 16)
problem136.types[:] = variable_types
problem136.constraints[0] = ">=0" #SEA
problem136.constraints[1] = "==0" #Susunan 2
problem136.constraints[2] = "==0" #Susunan 3
problem136.constraints[3] = "==0" #Susunan 4
problem136.constraints[4] = "==0" #Susunan 5
problem136.constraints[5] = "==1" #Susunan 6
problem136.constraints[6] = "==0" #Weldox 700E
problem136.constraints[7] = "==0" #Hardox 400
problem136.constraints[8] = "==1" #Domex Protect 500
problem136.constraints[9] = "==0" #Armox 560T
problem136.constraints[10] = "==0" #Al7075-T651
problem136.constraints[11] = "==0" #Kevlar-29
problem136.constraints[12] = "==0" #S2 Glass/SC15
problem136.constraints[13] = "==1" #CFRP
problem136.constraints[14] = ">=0.5" #thickness
problem136.constraints[15] = ">=1" #lapisan
problem136.directions[0] = Problem.MAXIMIZE
problem136.function = Opt

#Case137: Susunan 6 Armox 560T CNC
problem137 = Problem(15, 1, 16)
problem137.types[:] = variable_types
problem137.constraints[0] = ">=0" #SEA
problem137.constraints[1] = "==0" #Susunan 2
problem137.constraints[2] = "==0" #Susunan 3
problem137.constraints[3] = "==0" #Susunan 4
problem137.constraints[4] = "==0" #Susunan 5
problem137.constraints[5] = "==1" #Susunan 6
problem137.constraints[6] = "==0" #Weldox 700E
problem137.constraints[7] = "==0" #Hardox 400
problem137.constraints[8] = "==0" #Domex Protect 500
problem137.constraints[9] = "==1" #Armox 560T
problem137.constraints[10] = "==0" #Al7075-T651
problem137.constraints[11] = "==0" #Kevlar-29
problem137.constraints[12] = "==0" #S2 Glass/SC15
problem137.constraints[13] = "==0" #CFRP
problem137.constraints[14] = ">=0.5" #thickness
problem137.constraints[15] = ">=1" #lapisan
problem137.directions[0] = Problem.MAXIMIZE
problem137.function = Opt

#Case138: Susunan 6 Armox 560T Kevlar-29
problem138 = Problem(15, 1, 16)
problem138.types[:] = variable_types
problem138.constraints[0] = ">=0" #SEA
problem138.constraints[1] = "==0" #Susunan 2
problem138.constraints[2] = "==0" #Susunan 3
problem138.constraints[3] = "==0" #Susunan 4
problem138.constraints[4] = "==0" #Susunan 5
problem138.constraints[5] = "==1" #Susunan 6
problem138.constraints[6] = "==0" #Weldox 700E
problem138.constraints[7] = "==0" #Hardox 400
problem138.constraints[8] = "==0" #Domex Protect 500
problem138.constraints[9] = "==1" #Armox 560T
problem138.constraints[10] = "==0" #Al7075-T651
problem138.constraints[11] = "==1" #Kevlar-29
problem138.constraints[12] = "==0" #S2 Glass/SC15
problem138.constraints[13] = "==0" #CFRP
problem138.constraints[14] = ">=0.5" #thickness
problem138.constraints[15] = ">=1" #lapisan
problem138.directions[0] = Problem.MAXIMIZE
problem138.function = Opt

#Case139: Susunan 6 Armox 560T S2Glass/SC15
problem139 = Problem(15, 1, 16)
problem139.types[:] = variable_types
problem139.constraints[0] = ">=0" #SEA
problem139.constraints[1] = "==0" #Susunan 2
problem139.constraints[2] = "==0" #Susunan 3
problem139.constraints[3] = "==0" #Susunan 4
problem139.constraints[4] = "==0" #Susunan 5
problem139.constraints[5] = "==1" #Susunan 6
problem139.constraints[6] = "==0" #Weldox 700E
problem139.constraints[7] = "==0" #Hardox 400
problem139.constraints[8] = "==0" #Domex Protect 500
problem139.constraints[9] = "==1" #Armox 560T
problem139.constraints[10] = "==0" #Al7075-T651
problem139.constraints[11] = "==0" #Kevlar-29
problem139.constraints[12] = "==1" #S2 Glass/SC15
problem139.constraints[13] = "==0" #CFRP
problem139.constraints[14] = ">=0.5" #thickness
problem139.constraints[15] = ">=1" #lapisan
problem139.directions[0] = Problem.MAXIMIZE
problem139.function = Opt

#Case140: Susunan 6 Armox 560T CFRP
problem140 = Problem(15, 1, 16)
problem140.types[:] = variable_types
problem140.constraints[0] = ">=0" #SEA
problem140.constraints[1] = "==0" #Susunan 2
problem140.constraints[2] = "==0" #Susunan 3
problem140.constraints[3] = "==0" #Susunan 4
problem140.constraints[4] = "==0" #Susunan 5
problem140.constraints[5] = "==1" #Susunan 6
problem140.constraints[6] = "==0" #Weldox 700E
problem140.constraints[7] = "==0" #Hardox 400
problem140.constraints[8] = "==0" #Domex Protect 500
problem140.constraints[9] = "==1" #Armox 560T
problem140.constraints[10] = "==0" #Al7075-T651
problem140.constraints[11] = "==0" #Kevlar-29
problem140.constraints[12] = "==0" #S2 Glass/SC15
problem140.constraints[13] = "==1" #CFRP
problem140.constraints[14] = ">=0.5" #thickness
problem140.constraints[15] = ">=1" #lapisan
problem140.directions[0] = Problem.MAXIMIZE
problem140.function = Opt

#Case141: Susunan 6 Al7075-T651 CNC
problem141 = Problem(15, 1, 16)
problem141.types[:] = variable_types
problem141.constraints[0] = ">=0" #SEA
problem141.constraints[1] = "==0" #Susunan 2
problem141.constraints[2] = "==0" #Susunan 3
problem141.constraints[3] = "==0" #Susunan 4
problem141.constraints[4] = "==0" #Susunan 5
problem141.constraints[5] = "==1" #Susunan 6
problem141.constraints[6] = "==0" #Weldox 700E
problem141.constraints[7] = "==0" #Hardox 400
problem141.constraints[8] = "==0" #Domex Protect 500
problem141.constraints[9] = "==0" #Armox 560T
problem141.constraints[10] = "==1" #Al7075-T651
problem141.constraints[11] = "==0" #Kevlar-29
problem141.constraints[12] = "==0" #S2 Glass/SC15
problem141.constraints[13] = "==0" #CFRP
problem141.constraints[14] = ">=0.5" #thickness
problem141.constraints[15] = ">=1" #lapisan
problem141.directions[0] = Problem.MAXIMIZE
problem141.function = Opt

#Case142: Susunan 6 Al7075-T651 Kevlar-29
problem142 = Problem(15, 1, 16)
problem142.types[:] = variable_types
problem142.constraints[0] = ">=0" #SEA
problem142.constraints[1] = "==0" #Susunan 2
problem142.constraints[2] = "==0" #Susunan 3
problem142.constraints[3] = "==0" #Susunan 4
problem142.constraints[4] = "==0" #Susunan 5
problem142.constraints[5] = "==1" #Susunan 6
problem142.constraints[6] = "==0" #Weldox 700E
problem142.constraints[7] = "==0" #Hardox 400
problem142.constraints[8] = "==0" #Domex Protect 500
problem142.constraints[9] = "==0" #Armox 560T
problem142.constraints[10] = "==1" #Al7075-T651
problem142.constraints[11] = "==1" #Kevlar-29
problem142.constraints[12] = "==0" #S2 Glass/SC15
problem142.constraints[13] = "==0" #CFRP
problem142.constraints[14] = ">=0.5" #thickness
problem142.constraints[15] = ">=1" #lapisan
problem142.directions[0] = Problem.MAXIMIZE
problem142.function = Opt

#Case143: Susunan 6 Al7075-T651 S2Glass/SC15
problem143 = Problem(15, 1, 16)
problem143.types[:] = variable_types
problem143.constraints[0] = ">=0" #SEA
problem143.constraints[1] = "==0" #Susunan 2
problem143.constraints[2] = "==0" #Susunan 3
problem143.constraints[3] = "==0" #Susunan 4
problem143.constraints[4] = "==0" #Susunan 5
problem143.constraints[5] = "==1" #Susunan 6
problem143.constraints[6] = "==0" #Weldox 700E
problem143.constraints[7] = "==0" #Hardox 400
problem143.constraints[8] = "==0" #Domex Protect 500
problem143.constraints[9] = "==0" #Armox 560T
problem143.constraints[10] = "==1" #Al7075-T651
problem143.constraints[11] = "==0" #Kevlar-29
problem143.constraints[12] = "==1" #S2 Glass/SC15
problem143.constraints[13] = "==0" #CFRP
problem143.constraints[14] = ">=0.5" #thickness
problem143.constraints[15] = ">=1" #lapisan
problem143.directions[0] = Problem.MAXIMIZE
problem143.function = Opt

#Case144: Susunan 6 Al7075-T651 CFRP
problem144 = Problem(15, 1, 16)
problem144.types[:] = variable_types
problem144.constraints[0] = ">=0" #SEA
problem144.constraints[1] = "==0" #Susunan 2
problem144.constraints[2] = "==0" #Susunan 3
problem144.constraints[3] = "==0" #Susunan 4
problem144.constraints[4] = "==0" #Susunan 5
problem144.constraints[5] = "==1" #Susunan 6
problem144.constraints[6] = "==0" #Weldox 700E
problem144.constraints[7] = "==0" #Hardox 400
problem144.constraints[8] = "==0" #Domex Protect 500
problem144.constraints[9] = "==0" #Armox 560T
problem144.constraints[10] = "==1" #Al7075-T651
problem144.constraints[11] = "==0" #Kevlar-29
problem144.constraints[12] = "==0" #S2 Glass/SC15
problem144.constraints[13] = "==1" #CFRP
problem144.constraints[14] = ">=0.5" #thickness
problem144.constraints[15] = ">=1" #lapisan
problem144.directions[0] = Problem.MAXIMIZE
problem144.function = Opt
#Create problem-n definition for 51 other cases

#Iterasi setiap case
iteration =1000

i = 109
algorithm109 = NSGAII(problem109, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm109.run(iteration)
i = 110
algorithm110 = NSGAII(problem110, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm110.run(iteration)
i = 111
algorithm111 = NSGAII(problem111, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm111.run(iteration)
i = 112
algorithm112 = NSGAII(problem112, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm112.run(iteration)
i = 113
algorithm113 = NSGAII(problem113, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm113.run(iteration)
i = 114
algorithm114 = NSGAII(problem114, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm114.run(iteration)
i = 115
algorithm115 = NSGAII(problem115, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm115.run(iteration)
i = 116
algorithm116 = NSGAII(problem116, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm116.run(iteration)
i = 117
algorithm117 = NSGAII(problem117, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm117.run(iteration)
i = 118
algorithm118 = NSGAII(problem118, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm118.run(iteration)
i = 119
algorithm119 = NSGAII(problem119, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm119.run(iteration)
i = 120
algorithm120 = NSGAII(problem120, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm120.run(iteration)
i = 121
algorithm121 = NSGAII(problem121, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm121.run(iteration)
i = 122
algorithm122 = NSGAII(problem122, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm122.run(iteration)
i = 123
algorithm123 = NSGAII(problem123, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm123.run(iteration)
i = 124
algorithm124 = NSGAII(problem124, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm124.run(iteration)
i = 125
algorithm125 = NSGAII(problem125, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm125.run(iteration)
i = 126
algorithm126 = NSGAII(problem126, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm126.run(iteration)
i = 127
algorithm127 = NSGAII(problem127, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm127.run(iteration)
i = 128
algorithm128 = NSGAII(problem128, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm128.run(iteration)
i = 129
algorithm129 = NSGAII(problem129, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm129.run(iteration)
i = 130
algorithm130 = NSGAII(problem130, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm130.run(iteration)
i = 131
algorithm131 = NSGAII(problem131, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm131.run(iteration)
i = 132
algorithm132 = NSGAII(problem132, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm132.run(iteration)
i = 133
algorithm133 = NSGAII(problem133, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm133.run(iteration)
i = 134
algorithm134 = NSGAII(problem134, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm134.run(iteration)
i = 135
algorithm135 = NSGAII(problem135, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm135.run(iteration)
i = 136
algorithm136 = NSGAII(problem136, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm136.run(iteration)
i = 137
algorithm137 = NSGAII(problem137, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm137.run(iteration)
i = 138
algorithm138 = NSGAII(problem138, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm138.run(iteration)
i = 139
algorithm139 = NSGAII(problem139, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm139.run(iteration)
i = 140
algorithm140 = NSGAII(problem140, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm140.run(iteration)
i = 141
algorithm141 = NSGAII(problem141, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm141.run(iteration)
i = 142
algorithm142 = NSGAII(problem142, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm142.run(iteration)
i = 143
algorithm143 = NSGAII(problem143, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm143.run(iteration)
i = 144
algorithm144 = NSGAII(problem144, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm144.run(iteration)

#Create problem-i definition for 51 other cases

result109 = algorithm109.result
result110 = algorithm110.result
result111 = algorithm111.result
result112 = algorithm112.result
result113 = algorithm113.result
result114 = algorithm114.result
result115 = algorithm115.result
result116 = algorithm116.result
result117 = algorithm117.result
result118 = algorithm118.result
result119 = algorithm119.result
result120 = algorithm120.result
result121 = algorithm121.result
result122 = algorithm122.result
result123 = algorithm123.result
result124 = algorithm124.result
result125 = algorithm125.result
result126 = algorithm126.result
result127 = algorithm127.result
result128 = algorithm128.result
result129 = algorithm129.result
result130 = algorithm130.result
result131 = algorithm131.result
result132 = algorithm132.result
result133 = algorithm133.result
result134 = algorithm134.result
result135 = algorithm135.result
result136 = algorithm136.result
result137 = algorithm137.result
result138 = algorithm138.result
result139 = algorithm139.result
result140 = algorithm140.result
result141 = algorithm141.result
result142 = algorithm142.result
result143 = algorithm143.result
result144 = algorithm144.result

#Create result-i definition for 51 other cases

feasible_solutions = [s for s in result109 if s.feasible]
for solution in feasible_solutions:
    print(solution.objectives)


SEA_hasil109 = [s.constraints[0] for s in algorithm109.result if s.feasible]
SEA_hasil110 = [s.constraints[0] for s in algorithm110.result if s.feasible]
SEA_hasil111 = [s.constraints[0] for s in algorithm111.result if s.feasible]
SEA_hasil112 = [s.constraints[0] for s in algorithm112.result if s.feasible]
SEA_hasil113 = [s.constraints[0] for s in algorithm113.result if s.feasible]
SEA_hasil114 = [s.constraints[0] for s in algorithm114.result if s.feasible]
SEA_hasil115 = [s.constraints[0] for s in algorithm115.result if s.feasible]
SEA_hasil116 = [s.constraints[0] for s in algorithm116.result if s.feasible]
SEA_hasil117 = [s.constraints[0] for s in algorithm117.result if s.feasible]
SEA_hasil118 = [s.constraints[0] for s in algorithm118.result if s.feasible]
SEA_hasil119 = [s.constraints[0] for s in algorithm119.result if s.feasible]
SEA_hasil120 = [s.constraints[0] for s in algorithm120.result if s.feasible]
SEA_hasil121 = [s.constraints[0] for s in algorithm121.result if s.feasible]
SEA_hasil122 = [s.constraints[0] for s in algorithm122.result if s.feasible]
SEA_hasil123 = [s.constraints[0] for s in algorithm123.result if s.feasible]
SEA_hasil124 = [s.constraints[0] for s in algorithm124.result if s.feasible]
SEA_hasil125 = [s.constraints[0] for s in algorithm125.result if s.feasible]
SEA_hasil126 = [s.constraints[0] for s in algorithm126.result if s.feasible]
SEA_hasil127 = [s.constraints[0] for s in algorithm127.result if s.feasible]
SEA_hasil128 = [s.constraints[0] for s in algorithm128.result if s.feasible]
SEA_hasil129 = [s.constraints[0] for s in algorithm129.result if s.feasible]
SEA_hasil130 = [s.constraints[0] for s in algorithm130.result if s.feasible]
SEA_hasil131 = [s.constraints[0] for s in algorithm131.result if s.feasible]
SEA_hasil132 = [s.constraints[0] for s in algorithm132.result if s.feasible]
SEA_hasil133 = [s.constraints[0] for s in algorithm133.result if s.feasible]
SEA_hasil134 = [s.constraints[0] for s in algorithm134.result if s.feasible]
SEA_hasil135 = [s.constraints[0] for s in algorithm135.result if s.feasible]
SEA_hasil136 = [s.constraints[0] for s in algorithm136.result if s.feasible]
SEA_hasil137 = [s.constraints[0] for s in algorithm137.result if s.feasible]
SEA_hasil138 = [s.constraints[0] for s in algorithm138.result if s.feasible]
SEA_hasil139 = [s.constraints[0] for s in algorithm139.result if s.feasible]
SEA_hasil140 = [s.constraints[0] for s in algorithm140.result if s.feasible]
SEA_hasil141 = [s.constraints[0] for s in algorithm141.result if s.feasible]
SEA_hasil142 = [s.constraints[0] for s in algorithm142.result if s.feasible]
SEA_hasil143 = [s.constraints[0] for s in algorithm143.result if s.feasible]
SEA_hasil144 = [s.constraints[0] for s in algorithm144.result if s.feasible]

#Create SEA_hasil-i definition for 51 other cases

SEA_hasill=np.hstack((SEA_hasil109, 	SEA_hasil110, 	SEA_hasil111, 	SEA_hasil112, 	SEA_hasil113, 	SEA_hasil114, 	SEA_hasil115, 	SEA_hasil116, 	SEA_hasil117, 	SEA_hasil118, 	SEA_hasil119, 	SEA_hasil120, 	SEA_hasil121, 	SEA_hasil122, 	SEA_hasil123, 	SEA_hasil124, 	SEA_hasil125, 	SEA_hasil126, 	SEA_hasil127, 	SEA_hasil128, 	SEA_hasil129, 	SEA_hasil130, 	SEA_hasil131, 	SEA_hasil132, 	SEA_hasil133, 	SEA_hasil134, 	SEA_hasil135, 	SEA_hasil136, 	SEA_hasil137, 	SEA_hasil138, 	SEA_hasil139, 	SEA_hasil140, 	SEA_hasil141, 	SEA_hasil142, 	SEA_hasil143, 	SEA_hasil144))
#Create SEA_hasil for 51 other cases

print("SEA_hasill = ")
print(SEA_hasill)

Variable109 = np.reshape(np.array([s.variables for s in algorithm109.result if s.feasible]), (-1,15))
Variable110 = np.reshape(np.array([s.variables for s in algorithm110.result if s.feasible]), (-1,15))
Variable111 = np.reshape(np.array([s.variables for s in algorithm111.result if s.feasible]), (-1,15))
Variable112 = np.reshape(np.array([s.variables for s in algorithm112.result if s.feasible]), (-1,15))
Variable113 = np.reshape(np.array([s.variables for s in algorithm113.result if s.feasible]), (-1,15))
Variable114 = np.reshape(np.array([s.variables for s in algorithm114.result if s.feasible]), (-1,15))
Variable115 = np.reshape(np.array([s.variables for s in algorithm115.result if s.feasible]), (-1,15))
Variable116 = np.reshape(np.array([s.variables for s in algorithm116.result if s.feasible]), (-1,15))
Variable117 = np.reshape(np.array([s.variables for s in algorithm117.result if s.feasible]), (-1,15))
Variable118 = np.reshape(np.array([s.variables for s in algorithm118.result if s.feasible]), (-1,15))
Variable119 = np.reshape(np.array([s.variables for s in algorithm119.result if s.feasible]), (-1,15))
Variable120 = np.reshape(np.array([s.variables for s in algorithm120.result if s.feasible]), (-1,15))
Variable121 = np.reshape(np.array([s.variables for s in algorithm121.result if s.feasible]), (-1,15))
Variable122 = np.reshape(np.array([s.variables for s in algorithm122.result if s.feasible]), (-1,15))
Variable123 = np.reshape(np.array([s.variables for s in algorithm123.result if s.feasible]), (-1,15))
Variable124 = np.reshape(np.array([s.variables for s in algorithm124.result if s.feasible]), (-1,15))
Variable125 = np.reshape(np.array([s.variables for s in algorithm125.result if s.feasible]), (-1,15))
Variable126 = np.reshape(np.array([s.variables for s in algorithm126.result if s.feasible]), (-1,15))
Variable127 = np.reshape(np.array([s.variables for s in algorithm127.result if s.feasible]), (-1,15))
Variable128 = np.reshape(np.array([s.variables for s in algorithm128.result if s.feasible]), (-1,15))
Variable129 = np.reshape(np.array([s.variables for s in algorithm129.result if s.feasible]), (-1,15))
Variable130 = np.reshape(np.array([s.variables for s in algorithm130.result if s.feasible]), (-1,15))
Variable131 = np.reshape(np.array([s.variables for s in algorithm131.result if s.feasible]), (-1,15))
Variable132 = np.reshape(np.array([s.variables for s in algorithm132.result if s.feasible]), (-1,15))
Variable133 = np.reshape(np.array([s.variables for s in algorithm133.result if s.feasible]), (-1,15))
Variable134 = np.reshape(np.array([s.variables for s in algorithm134.result if s.feasible]), (-1,15))
Variable135 = np.reshape(np.array([s.variables for s in algorithm135.result if s.feasible]), (-1,15))
Variable136 = np.reshape(np.array([s.variables for s in algorithm136.result if s.feasible]), (-1,15))
Variable137 = np.reshape(np.array([s.variables for s in algorithm137.result if s.feasible]), (-1,15))
Variable138 = np.reshape(np.array([s.variables for s in algorithm138.result if s.feasible]), (-1,15))
Variable139 = np.reshape(np.array([s.variables for s in algorithm139.result if s.feasible]), (-1,15))
Variable140 = np.reshape(np.array([s.variables for s in algorithm140.result if s.feasible]), (-1,15))
Variable141 = np.reshape(np.array([s.variables for s in algorithm141.result if s.feasible]), (-1,15))
Variable142 = np.reshape(np.array([s.variables for s in algorithm142.result if s.feasible]), (-1,15))
Variable143 = np.reshape(np.array([s.variables for s in algorithm143.result if s.feasible]), (-1,15))
Variable144 = np.reshape(np.array([s.variables for s in algorithm144.result if s.feasible]), (-1,15))


#Create Variable-i definition for 51 other cases

Variables = np.vstack((Variable109, 	Variable110, 	Variable111, 	Variable112, 	Variable113, 	Variable114, 	Variable115, 	Variable116, 	Variable117, 	Variable118, 	Variable119, 	Variable120, 	Variable121, 	Variable122, 	Variable123, 	Variable124, 	Variable125, 	Variable126, 	Variable127, 	Variable128, 	Variable129, 	Variable130, 	Variable131, 	Variable132, 	Variable133, 	Variable134, 	Variable135, 	Variable136, 	Variable137, 	Variable138, 	Variable139, 	Variable140, 	Variable141, 	Variable142, 	Variable143, 	Variable144))
#Create Variable-i definition for 51 other cases

print("Variables = ")
print(Variables)
solutionwrite = pd.ExcelWriter('NSGA Solutions3.xlsx', engine='openpyxl')
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















