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

#Case1: Susunan 1 Weldox 500E CNC
problem1 = Problem(15, 1, 16)
problem1.types[:] = variable_types
problem1.constraints[0] = ">=0" #SEA
problem1.constraints[1] = "==0" #Susunan 2
problem1.constraints[2] = "==0" #Susunan 3
problem1.constraints[3] = "==0" #Susunan 4
problem1.constraints[4] = "==0" #Susunan 5
problem1.constraints[5] = "==0" #Susunan 6
problem1.constraints[6] = "==0" #Weldox 700E
problem1.constraints[7] = "==0" #Hardox 400
problem1.constraints[8] = "==0" #Domex Protect 500
problem1.constraints[9] = "==0" #Armox 560T
problem1.constraints[10] = "==0" #Al7075-T651
problem1.constraints[11] = "==0" #Kevlar-29
problem1.constraints[12] = "==0" #S2 Glass/SC15
problem1.constraints[13] = "==0" #CFRP
problem1.constraints[14] = ">=0.5" #thickness
problem1.constraints[15] = ">=1" #lapisan
problem1.directions[0] = Problem.MAXIMIZE
problem1.function = Opt

#Case2: Susunan 1 Weldox 500E Kevlar-29
problem2 = Problem(15, 1, 16)
problem2.types[:] = variable_types
problem2.constraints[0] = ">=0" #SEA
problem2.constraints[1] = "==0" #Susunan 2
problem2.constraints[2] = "==0" #Susunan 3
problem2.constraints[3] = "==0" #Susunan 4
problem2.constraints[4] = "==0" #Susunan 5
problem2.constraints[5] = "==0" #Susunan 6
problem2.constraints[6] = "==0" #Weldox 700E
problem2.constraints[7] = "==0" #Hardox 400
problem2.constraints[8] = "==0" #Domex Protect 500
problem2.constraints[9] = "==0" #Armox 560T
problem2.constraints[10] = "==0" #Weldox Al7075-T651
problem2.constraints[11] = "==1" #Kevlar-29
problem2.constraints[12] = "==0" #S2 Glass/SC15
problem2.constraints[13] = "==0" #CFRP
problem2.constraints[14] = ">=0.5" #thickness
problem2.constraints[15] = ">=1" #lapisan
problem2.directions[0] = Problem.MAXIMIZE
problem2.function = Opt

#Case3: Susunan 1 Weldox 500E S2Glass/SC15
problem3 = Problem(15, 1, 16)
problem3.types[:] = variable_types
problem3.constraints[0] = ">=0" #SEA
problem3.constraints[1] = "==0" #Susunan 2
problem3.constraints[2] = "==0" #Susunan 3
problem3.constraints[3] = "==0" #Susunan 4
problem3.constraints[4] = "==0" #Susunan 5
problem3.constraints[5] = "==0" #Susunan 6
problem3.constraints[6] = "==0" #Weldox 700E
problem3.constraints[7] = "==0" #Hardox 400
problem3.constraints[8] = "==0" #Domex Protect 500
problem3.constraints[9] = "==0" #Armox 560T
problem3.constraints[10] = "==0" #Al7075-T651
problem3.constraints[11] = "==0" #Kevlar-29
problem3.constraints[12] = "==1" #S2 Glass/SC15
problem3.constraints[13] = "==0" #CFRP
problem3.constraints[14] = ">=0.5" #thickness
problem3.constraints[15] = ">=1" #lapisan
problem3.directions[0] = Problem.MAXIMIZE
problem3.function = Opt

#Case4: Susunan 1 Weldox 500E CFRP
problem4 = Problem(15, 1, 16)
problem4.types[:] = variable_types
problem4.constraints[0] = ">=0" #SEA
problem4.constraints[1] = "==0" #Susunan 2
problem4.constraints[2] = "==0" #Susunan 3
problem4.constraints[3] = "==0" #Susunan 4
problem4.constraints[4] = "==0" #Susunan 5
problem4.constraints[5] = "==0" #Susunan 6
problem4.constraints[6] = "==0" #Weldox 700E
problem4.constraints[7] = "==0" #Hardox 400
problem4.constraints[8] = "==0" #Domex Protect 500
problem4.constraints[9] = "==0" #Armox 560T
problem4.constraints[10] = "==0" #Weldox Al7075-T651
problem4.constraints[11] = "==0" #Kevlar-29
problem4.constraints[12] = "==0" #S2 Glass/SC15
problem4.constraints[13] = "==1" #CFRP
problem4.constraints[14] = ">=0.5" #thickness
problem4.constraints[15] = ">=1" #lapisan
problem4.directions[0] = Problem.MAXIMIZE
problem4.function = Opt

#Case5: Susunan 1 Weldox 700E CNC
problem5 = Problem(15, 1, 16)
problem5.types[:] = variable_types
problem5.constraints[0] = ">=0" #SEA
problem5.constraints[1] = "==0" #Susunan 2
problem5.constraints[2] = "==0" #Susunan 3
problem5.constraints[3] = "==0" #Susunan 4
problem5.constraints[4] = "==0" #Susunan 5
problem5.constraints[5] = "==0" #Susunan 6
problem5.constraints[6] = "==1" #Weldox 700E
problem5.constraints[7] = "==0" #Hardox 400
problem5.constraints[8] = "==0" #Domex Protect 500
problem5.constraints[9] = "==0" #Armox 560T
problem5.constraints[10] = "==0" #Al7075-T651
problem5.constraints[11] = "==0" #Kevlar-29
problem5.constraints[12] = "==0" #S2 Glass/SC15
problem5.constraints[13] = "==0" #CFRP
problem5.constraints[14] = ">=0.5" #thickness
problem5.constraints[15] = ">=1" #lapisan
problem5.directions[0] = Problem.MAXIMIZE
problem5.function = Opt

#Case6: Susunan 1 Weldox 700E Kevlar-29
problem6 = Problem(15, 1, 16)
problem6.types[:] = variable_types
problem6.constraints[0] = ">=0" #SEA
problem6.constraints[1] = "==0" #Susunan 2
problem6.constraints[2] = "==0" #Susunan 3
problem6.constraints[3] = "==0" #Susunan 4
problem6.constraints[4] = "==0" #Susunan 5
problem6.constraints[5] = "==0" #Susunan 6
problem6.constraints[6] = "==1" #Weldox 700E
problem6.constraints[7] = "==0" #Hardox 400
problem6.constraints[8] = "==0" #Domex Protect 500
problem6.constraints[9] = "==0" #Armox 560T
problem6.constraints[10] = "==0" #Al7075-T651
problem6.constraints[11] = "==1" #Kevlar-29
problem6.constraints[12] = "==0" #S2 Glass/SC15
problem6.constraints[13] = "==0" #CFRP
problem6.constraints[14] = ">=0.5" #thickness
problem6.constraints[15] = ">=1" #lapisan
problem6.directions[0] = Problem.MAXIMIZE
problem6.function = Opt

#Case7: Susunan 1 Weldox 700E S2Glass/SC15
problem7 = Problem(15, 1, 16)
problem7.types[:] = variable_types
problem7.constraints[0] = ">=0" #SEA
problem7.constraints[1] = "==0" #Susunan 2
problem7.constraints[2] = "==0" #Susunan 3
problem7.constraints[3] = "==0" #Susunan 4
problem7.constraints[4] = "==0" #Susunan 5
problem7.constraints[5] = "==0" #Susunan 6
problem7.constraints[6] = "==1" #Weldox 700E
problem7.constraints[7] = "==0" #Hardox 400
problem7.constraints[8] = "==0" #Domex Protect 500
problem7.constraints[9] = "==0" #Armox 560T
problem7.constraints[10] = "==0" #Al7075-T651
problem7.constraints[11] = "==0" #Kevlar-29
problem7.constraints[12] = "==1" #S2 Glass/SC15
problem7.constraints[13] = "==0" #CFRP
problem7.constraints[14] = ">=0.5" #thickness
problem7.constraints[15] = ">=1" #lapisan
problem7.directions[0] = Problem.MAXIMIZE
problem7.function = Opt

#Case8: Susunan 1 Weldox 700E CFRP
problem8 = Problem(15, 1, 16)
problem8.types[:] = variable_types
problem8.constraints[0] = ">=0" #SEA
problem8.constraints[1] = "==0" #Susunan 2
problem8.constraints[2] = "==0" #Susunan 3
problem8.constraints[3] = "==0" #Susunan 4
problem8.constraints[4] = "==0" #Susunan 5
problem8.constraints[5] = "==0" #Susunan 6
problem8.constraints[6] = "==1" #Weldox 700E
problem8.constraints[7] = "==0" #Hardox 400
problem8.constraints[8] = "==0" #Domex Protect 500
problem8.constraints[9] = "==0" #Armox 560T
problem8.constraints[10] = "==0" #Al7075-T651
problem8.constraints[11] = "==0" #Kevlar-29
problem8.constraints[12] = "==0" #S2 Glass/SC15
problem8.constraints[13] = "==1" #CFRP
problem8.constraints[14] = ">=0.5" #thickness
problem8.constraints[15] = ">=1" #lapisan
problem8.directions[0] = Problem.MAXIMIZE
problem8.function = Opt

#Case9: Susunan 1 Hardox 400 CNC
problem9 = Problem(15, 1, 16)
problem9.types[:] = variable_types
problem9.constraints[0] = ">=0" #SEA
problem9.constraints[1] = "==0" #Susunan 2
problem9.constraints[2] = "==0" #Susunan 3
problem9.constraints[3] = "==0" #Susunan 4
problem9.constraints[4] = "==0" #Susunan 5
problem9.constraints[5] = "==0" #Susunan 6
problem9.constraints[6] = "==0" #Weldox 700E
problem9.constraints[7] = "==1" #Hardox 400
problem9.constraints[8] = "==0" #Domex Protect 500
problem9.constraints[9] = "==0" #Armox 560T
problem9.constraints[10] = "==0" #Al7075-T651
problem9.constraints[11] = "==0" #Kevlar-29
problem9.constraints[12] = "==0" #S2 Glass/SC15
problem9.constraints[13] = "==0" #CFRP
problem9.constraints[14] = ">=0.5" #thickness
problem9.constraints[15] = ">=1" #lapisan
problem9.directions[0] = Problem.MAXIMIZE
problem9.function = Opt

#Case10: Susunan 1 Hardox 400 Kevlar-29
problem10 = Problem(15, 1, 16)
problem10.types[:] = variable_types
problem10.constraints[0] = ">=0" #SEA
problem10.constraints[1] = "==0" #Susunan 2
problem10.constraints[2] = "==0" #Susunan 3
problem10.constraints[3] = "==0" #Susunan 4
problem10.constraints[4] = "==0" #Susunan 5
problem10.constraints[5] = "==0" #Susunan 6
problem10.constraints[6] = "==0" #Weldox 700E
problem10.constraints[7] = "==1" #Hardox 400
problem10.constraints[8] = "==0" #Domex Protect 500
problem10.constraints[9] = "==0" #Armox 560T
problem10.constraints[10] = "==0" #Al7075-T651
problem10.constraints[11] = "==1" #Kevlar-29
problem10.constraints[12] = "==0" #S2 Glass/SC15
problem10.constraints[13] = "==0" #CFRP
problem10.constraints[14] = ">=0.5" #thickness
problem10.constraints[15] = ">=1" #lapisan
problem10.directions[0] = Problem.MAXIMIZE
problem10.function = Opt

#Case11: Susunan 1 Hardox 400 S2Glass/SC15
problem11 = Problem(15, 1, 16)
problem11.types[:] = variable_types
problem11.constraints[0] = ">=0" #SEA
problem11.constraints[1] = "==0" #Susunan 2
problem11.constraints[2] = "==0" #Susunan 3
problem11.constraints[3] = "==0" #Susunan 4
problem11.constraints[4] = "==0" #Susunan 5
problem11.constraints[5] = "==0" #Susunan 6
problem11.constraints[6] = "==0" #Weldox 700E
problem11.constraints[7] = "==1" #Hardox 400
problem11.constraints[8] = "==0" #Domex Protect 500
problem11.constraints[9] = "==0" #Armox 560T
problem11.constraints[10] = "==0" #Al7075-T651
problem11.constraints[11] = "==0" #Kevlar-29
problem11.constraints[12] = "==1" #S2 Glass/SC15
problem11.constraints[13] = "==0" #CFRP
problem11.constraints[14] = ">=0.5" #thickness
problem11.constraints[15] = ">=1" #lapisan
problem11.directions[0] = Problem.MAXIMIZE
problem11.function = Opt

#Case12: Susunan 1 Hardox 400 CFRP
problem12 = Problem(15, 1, 16)
problem12.types[:] = variable_types
problem12.constraints[0] = ">=0" #SEA
problem12.constraints[1] = "==0" #Susunan 2
problem12.constraints[2] = "==0" #Susunan 3
problem12.constraints[3] = "==0" #Susunan 4
problem12.constraints[4] = "==0" #Susunan 5
problem12.constraints[5] = "==0" #Susunan 6
problem12.constraints[6] = "==0" #Weldox 700E
problem12.constraints[7] = "==1" #Hardox 400
problem12.constraints[8] = "==0" #Domex Protect 500
problem12.constraints[9] = "==0" #Armox 560T
problem12.constraints[10] = "==0" #Al7075-T651
problem12.constraints[11] = "==0" #Kevlar-29
problem12.constraints[12] = "==0" #S2 Glass/SC15
problem12.constraints[13] = "==1" #CFRP
problem12.constraints[14] = ">=0.5" #thickness
problem12.constraints[15] = ">=1" #lapisan
problem12.directions[0] = Problem.MAXIMIZE
problem12.function = Opt

#Case13: Susunan 1 Domex Protect 500 CNC
problem13 = Problem(15, 1, 16)
problem13.types[:] = variable_types
problem13.constraints[0] = ">=0" #SEA
problem13.constraints[1] = "==0" #Susunan 2
problem13.constraints[2] = "==0" #Susunan 3
problem13.constraints[3] = "==0" #Susunan 4
problem13.constraints[4] = "==0" #Susunan 5
problem13.constraints[5] = "==0" #Susunan 6
problem13.constraints[6] = "==0" #Weldox 700E
problem13.constraints[7] = "==0" #Hardox 400
problem13.constraints[8] = "==1" #Domex Protect 500
problem13.constraints[9] = "==0" #Armox 560T
problem13.constraints[10] = "==0" #Al7075-T651
problem13.constraints[11] = "==0" #Kevlar-29
problem13.constraints[12] = "==0" #S2 Glass/SC15
problem13.constraints[13] = "==0" #CFRP
problem13.constraints[14] = ">=0.5" #thickness
problem13.constraints[15] = ">=1" #lapisan
problem13.directions[0] = Problem.MAXIMIZE
problem13.function = Opt

#Case14: Susunan 1 Domex Protect 500 Kevlar-29
problem14 = Problem(15, 1, 16)
problem14.types[:] = variable_types
problem14.constraints[0] = ">=0" #SEA
problem14.constraints[1] = "==0" #Susunan 2
problem14.constraints[2] = "==0" #Susunan 3
problem14.constraints[3] = "==0" #Susunan 4
problem14.constraints[4] = "==0" #Susunan 5
problem14.constraints[5] = "==0" #Susunan 6
problem14.constraints[6] = "==0" #Weldox 700E
problem14.constraints[7] = "==0" #Hardox 400
problem14.constraints[8] = "==1" #Domex Protect 500
problem14.constraints[9] = "==0" #Armox 560T
problem14.constraints[10] = "==0" #Al7075-T651
problem14.constraints[11] = "==1" #Kevlar-29
problem14.constraints[12] = "==0" #S2 Glass/SC15
problem14.constraints[13] = "==0" #CFRP
problem14.constraints[14] = ">=0.5" #thickness
problem14.constraints[15] = ">=1" #lapisan
problem14.directions[0] = Problem.MAXIMIZE
problem14.function = Opt

#Case15: Susunan 1 Domex Protect 500 S2Glass/SC15
problem15 = Problem(15, 1, 16)
problem15.types[:] = variable_types
problem15.constraints[0] = ">=0" #SEA
problem15.constraints[1] = "==0" #Susunan 2
problem15.constraints[2] = "==0" #Susunan 3
problem15.constraints[3] = "==0" #Susunan 4
problem15.constraints[4] = "==0" #Susunan 5
problem15.constraints[5] = "==0" #Susunan 6
problem15.constraints[6] = "==0" #Weldox 700E
problem15.constraints[7] = "==0" #Hardox 400
problem15.constraints[8] = "==1" #Domex Protect 500
problem15.constraints[9] = "==0" #Armox 560T
problem15.constraints[10] = "==0" #Al7075-T651
problem15.constraints[11] = "==0" #Kevlar-29
problem15.constraints[12] = "==1" #S2 Glass/SC15
problem15.constraints[13] = "==0" #CFRP
problem15.constraints[14] = ">=0.5" #thickness
problem15.constraints[15] = ">=1" #lapisan
problem15.directions[0] = Problem.MAXIMIZE
problem15.function = Opt

#Case16: Susunan 1 Domex Protect 500 CFRP
problem16 = Problem(15, 1, 16)
problem16.types[:] = variable_types
problem16.constraints[0] = ">=0" #SEA
problem16.constraints[1] = "==0" #Susunan 2
problem16.constraints[2] = "==0" #Susunan 3
problem16.constraints[3] = "==0" #Susunan 4
problem16.constraints[4] = "==0" #Susunan 5
problem16.constraints[5] = "==0" #Susunan 6
problem16.constraints[6] = "==0" #Weldox 700E
problem16.constraints[7] = "==0" #Hardox 400
problem16.constraints[8] = "==1" #Domex Protect 500
problem16.constraints[9] = "==0" #Armox 560T
problem16.constraints[10] = "==0" #Al7075-T651
problem16.constraints[11] = "==0" #Kevlar-29
problem16.constraints[12] = "==0" #S2 Glass/SC15
problem16.constraints[13] = "==1" #CFRP
problem16.constraints[14] = ">=0.5" #thickness
problem16.constraints[15] = ">=1" #lapisan
problem16.directions[0] = Problem.MAXIMIZE
problem16.function = Opt

#Case17: Susunan 1 Armox 560T CNC
problem17 = Problem(15, 1, 16)
problem17.types[:] = variable_types
problem17.constraints[0] = ">=0" #SEA
problem17.constraints[1] = "==0" #Susunan 2
problem17.constraints[2] = "==0" #Susunan 3
problem17.constraints[3] = "==0" #Susunan 4
problem17.constraints[4] = "==0" #Susunan 5
problem17.constraints[5] = "==0" #Susunan 6
problem17.constraints[6] = "==0" #Weldox 700E
problem17.constraints[7] = "==0" #Hardox 400
problem17.constraints[8] = "==0" #Domex Protect 500
problem17.constraints[9] = "==1" #Armox 560T
problem17.constraints[10] = "==0" #Al7075-T651
problem17.constraints[11] = "==0" #Kevlar-29
problem17.constraints[12] = "==0" #S2 Glass/SC15
problem17.constraints[13] = "==0" #CFRP
problem17.constraints[14] = ">=0.5" #thickness
problem17.constraints[15] = ">=1" #lapisan
problem17.directions[0] = Problem.MAXIMIZE
problem17.function = Opt

#Case18: Susunan 1 Armox 560T Kevlar-29
problem18 = Problem(15, 1, 16)
problem18.types[:] = variable_types
problem18.constraints[0] = ">=0" #SEA
problem18.constraints[1] = "==0" #Susunan 2
problem18.constraints[2] = "==0" #Susunan 3
problem18.constraints[3] = "==0" #Susunan 4
problem18.constraints[4] = "==0" #Susunan 5
problem18.constraints[5] = "==0" #Susunan 6
problem18.constraints[6] = "==0" #Weldox 700E
problem18.constraints[7] = "==0" #Hardox 400
problem18.constraints[8] = "==0" #Domex Protect 500
problem18.constraints[9] = "==1" #Armox 560T
problem18.constraints[10] = "==0" #Al7075-T651
problem18.constraints[11] = "==1" #Kevlar-29
problem18.constraints[12] = "==0" #S2 Glass/SC15
problem18.constraints[13] = "==0" #CFRP
problem18.constraints[14] = ">=0.5" #thickness
problem18.constraints[15] = ">=1" #lapisan
problem18.directions[0] = Problem.MAXIMIZE
problem18.function = Opt

#Case19: Susunan 1 Armox 560T S2Glass/SC15
problem19 = Problem(15, 1, 16)
problem19.types[:] = variable_types
problem19.constraints[0] = ">=0" #SEA
problem19.constraints[1] = "==0" #Susunan 2
problem19.constraints[2] = "==0" #Susunan 3
problem19.constraints[3] = "==0" #Susunan 4
problem19.constraints[4] = "==0" #Susunan 5
problem19.constraints[5] = "==0" #Susunan 6
problem19.constraints[6] = "==0" #Weldox 700E
problem19.constraints[7] = "==0" #Hardox 400
problem19.constraints[8] = "==0" #Domex Protect 500
problem19.constraints[9] = "==1" #Armox 560T
problem19.constraints[10] = "==0" #Al7075-T651
problem19.constraints[11] = "==0" #Kevlar-29
problem19.constraints[12] = "==1" #S2 Glass/SC15
problem19.constraints[13] = "==0" #CFRP
problem19.constraints[14] = ">=0.5" #thickness
problem19.constraints[15] = ">=1" #lapisan
problem19.directions[0] = Problem.MAXIMIZE
problem19.function = Opt

#Case20: Susunan 1 Armox 560T CFRP
problem20 = Problem(15, 1, 16)
problem20.types[:] = variable_types
problem20.constraints[0] = ">=0" #SEA
problem20.constraints[1] = "==0" #Susunan 2
problem20.constraints[2] = "==0" #Susunan 3
problem20.constraints[3] = "==0" #Susunan 4
problem20.constraints[4] = "==0" #Susunan 5
problem20.constraints[5] = "==0" #Susunan 6
problem20.constraints[6] = "==0" #Weldox 700E
problem20.constraints[7] = "==0" #Hardox 400
problem20.constraints[8] = "==0" #Domex Protect 500
problem20.constraints[9] = "==1" #Armox 560T
problem20.constraints[10] = "==0" #Al7075-T651
problem20.constraints[11] = "==0" #Kevlar-29
problem20.constraints[12] = "==0" #S2 Glass/SC15
problem20.constraints[13] = "==1" #CFRP
problem20.constraints[14] = ">=0.5" #thickness
problem20.constraints[15] = ">=1" #lapisan
problem20.directions[0] = Problem.MAXIMIZE
problem20.function = Opt

#Case21: Susunan 1 Al7075-T651 CNC
problem21 = Problem(15, 1, 16)
problem21.types[:] = variable_types
problem21.constraints[0] = ">=0" #SEA
problem21.constraints[1] = "==0" #Susunan 2
problem21.constraints[2] = "==0" #Susunan 3
problem21.constraints[3] = "==0" #Susunan 4
problem21.constraints[4] = "==0" #Susunan 5
problem21.constraints[5] = "==0" #Susunan 6
problem21.constraints[6] = "==0" #Weldox 700E
problem21.constraints[7] = "==0" #Hardox 400
problem21.constraints[8] = "==0" #Domex Protect 500
problem21.constraints[9] = "==0" #Armox 560T
problem21.constraints[10] = "==1" #Al7075-T651
problem21.constraints[11] = "==0" #Kevlar-29
problem21.constraints[12] = "==0" #S2 Glass/SC15
problem21.constraints[13] = "==0" #CFRP
problem21.constraints[14] = ">=0.5" #thickness
problem21.constraints[15] = ">=1" #lapisan
problem21.directions[0] = Problem.MAXIMIZE
problem21.function = Opt

#Case22: Susunan 1 Al7075-T651 Kevlar-29
problem22 = Problem(15, 1, 16)
problem22.types[:] = variable_types
problem22.constraints[0] = ">=0" #SEA
problem22.constraints[1] = "==0" #Susunan 2
problem22.constraints[2] = "==0" #Susunan 3
problem22.constraints[3] = "==0" #Susunan 4
problem22.constraints[4] = "==0" #Susunan 5
problem22.constraints[5] = "==0" #Susunan 6
problem22.constraints[6] = "==0" #Weldox 700E
problem22.constraints[7] = "==0" #Hardox 400
problem22.constraints[8] = "==0" #Domex Protect 500
problem22.constraints[9] = "==0" #Armox 560T
problem22.constraints[10] = "==1" #Al7075-T651
problem22.constraints[11] = "==1" #Kevlar-29
problem22.constraints[12] = "==0" #S2 Glass/SC15
problem22.constraints[13] = "==0" #CFRP
problem22.constraints[14] = ">=0.5" #thickness
problem22.constraints[15] = ">=1" #lapisan
problem22.directions[0] = Problem.MAXIMIZE
problem22.function = Opt

#Case23: Susunan 1 Al7075-T651 S2Glass/SC15
problem23 = Problem(15, 1, 16)
problem23.types[:] = variable_types
problem23.constraints[0] = ">=0" #SEA
problem23.constraints[1] = "==0" #Susunan 2
problem23.constraints[2] = "==0" #Susunan 3
problem23.constraints[3] = "==0" #Susunan 4
problem23.constraints[4] = "==0" #Susunan 5
problem23.constraints[5] = "==0" #Susunan 6
problem23.constraints[6] = "==0" #Weldox 700E
problem23.constraints[7] = "==0" #Hardox 400
problem23.constraints[8] = "==0" #Domex Protect 500
problem23.constraints[9] = "==0" #Armox 560T
problem23.constraints[10] = "==1" #Al7075-T651
problem23.constraints[11] = "==0" #Kevlar-29
problem23.constraints[12] = "==1" #S2 Glass/SC15
problem23.constraints[13] = "==0" #CFRP
problem23.constraints[14] = ">=0.5" #thickness
problem23.constraints[15] = ">=1" #lapisan
problem23.directions[0] = Problem.MAXIMIZE
problem23.function = Opt

#Case24: Susunan 1 Al7075-T651 CFRP
problem24 = Problem(15, 1, 16)
problem24.types[:] = variable_types
problem24.constraints[0] = ">=0" #SEA
problem24.constraints[1] = "==0" #Susunan 2
problem24.constraints[2] = "==0" #Susunan 3
problem24.constraints[3] = "==0" #Susunan 4
problem24.constraints[4] = "==0" #Susunan 5
problem24.constraints[5] = "==0" #Susunan 6
problem24.constraints[6] = "==0" #Weldox 700E
problem24.constraints[7] = "==0" #Hardox 400
problem24.constraints[8] = "==0" #Domex Protect 500
problem24.constraints[9] = "==0" #Armox 560T
problem24.constraints[10] = "==1" #Al7075-T651
problem24.constraints[11] = "==0" #Kevlar-29
problem24.constraints[12] = "==0" #S2 Glass/SC15
problem24.constraints[13] = "==1" #CFRP
problem24.constraints[14] = ">=0.5" #thickness
problem24.constraints[15] = ">=1" #lapisan
problem24.directions[0] = Problem.MAXIMIZE
problem24.function = Opt

#Case25: Susunan 2 Weldox 500E CNC
problem25 = Problem(15, 1, 16)
problem25.types[:] = variable_types
problem25.constraints[0] = ">=0" #SEA
problem25.constraints[1] = "==1" #Susunan 2
problem25.constraints[2] = "==0" #Susunan 3
problem25.constraints[3] = "==0" #Susunan 4
problem25.constraints[4] = "==0" #Susunan 5
problem25.constraints[5] = "==0" #Susunan 6
problem25.constraints[6] = "==0" #Weldox 700E
problem25.constraints[7] = "==0" #Hardox 400
problem25.constraints[8] = "==0" #Domex Protect 500
problem25.constraints[9] = "==0" #Armox 560T
problem25.constraints[10] = "==0" #Al7075-T651
problem25.constraints[11] = "==0" #Kevlar-29
problem25.constraints[12] = "==0" #S2 Glass/SC15
problem25.constraints[13] = "==0" #CFRP
problem25.constraints[14] = ">=0.5" #thickness
problem25.constraints[15] = ">=1" #lapisan
problem25.directions[0] = Problem.MAXIMIZE
problem25.function = Opt

#Case26: Susunan 2 Weldox 500E Kevlar-29
problem26 = Problem(15, 1, 16)
problem26.types[:] = variable_types
problem26.constraints[0] = ">=0" #SEA
problem26.constraints[1] = "==1" #Susunan 2
problem26.constraints[2] = "==0" #Susunan 3
problem26.constraints[3] = "==0" #Susunan 4
problem26.constraints[4] = "==0" #Susunan 5
problem26.constraints[5] = "==0" #Susunan 6
problem26.constraints[6] = "==0" #Weldox 700E
problem26.constraints[7] = "==0" #Hardox 400
problem26.constraints[8] = "==0" #Domex Protect 500
problem26.constraints[9] = "==0" #Armox 560T
problem26.constraints[10] = "==0" #Al7075-T651
problem26.constraints[11] = "==1" #Kevlar-29
problem26.constraints[12] = "==0" #S2 Glass/SC15
problem26.constraints[13] = "==0" #CFRP
problem26.constraints[14] = ">=0.5" #thickness
problem26.constraints[15] = ">=1" #lapisan
problem26.directions[0] = Problem.MAXIMIZE
problem26.function = Opt

#Case27: Susunan 2 Weldox 500E S2Glass/SC15
problem27 = Problem(15, 1, 16)
problem27.types[:] = variable_types
problem27.constraints[0] = ">=0" #SEA
problem27.constraints[1] = "==1" #Susunan 2
problem27.constraints[2] = "==0" #Susunan 3
problem27.constraints[3] = "==0" #Susunan 4
problem27.constraints[4] = "==0" #Susunan 5
problem27.constraints[5] = "==0" #Susunan 6
problem27.constraints[6] = "==0" #Weldox 700E
problem27.constraints[7] = "==0" #Hardox 400
problem27.constraints[8] = "==0" #Domex Protect 500
problem27.constraints[9] = "==0" #Armox 560T
problem27.constraints[10] = "==0" #Al7075-T651
problem27.constraints[11] = "==0" #Kevlar-29
problem27.constraints[12] = "==1" #S2 Glass/SC15
problem27.constraints[13] = "==0" #CFRP
problem27.constraints[14] = ">=0.5" #thickness
problem27.constraints[15] = ">=1" #lapisan
problem27.directions[0] = Problem.MAXIMIZE
problem27.function = Opt

#Case28: Susunan 2 Weldox 500E CFRP
problem28 = Problem(15, 1, 16)
problem28.types[:] = variable_types
problem28.constraints[0] = ">=0" #SEA
problem28.constraints[1] = "==1" #Susunan 2
problem28.constraints[2] = "==0" #Susunan 3
problem28.constraints[3] = "==0" #Susunan 4
problem28.constraints[4] = "==0" #Susunan 5
problem28.constraints[5] = "==0" #Susunan 6
problem28.constraints[6] = "==0" #Weldox 700E
problem28.constraints[7] = "==0" #Hardox 400
problem28.constraints[8] = "==0" #Domex Protect 500
problem28.constraints[9] = "==0" #Armox 560T
problem28.constraints[10] = "==0" #Al7075-T651
problem28.constraints[11] = "==0" #Kevlar-29
problem28.constraints[12] = "==0" #S2 Glass/SC15
problem28.constraints[13] = "==1" #CFRP
problem28.constraints[14] = ">=0.5" #thickness
problem28.constraints[15] = ">=1" #lapisan
problem28.directions[0] = Problem.MAXIMIZE
problem28.function = Opt

#Case29: Susunan 2 Weldox 700E CNC
problem29 = Problem(15, 1, 16)
problem29.types[:] = variable_types
problem29.constraints[0] = ">=0" #SEA
problem29.constraints[1] = "==1" #Susunan 2
problem29.constraints[2] = "==0" #Susunan 3
problem29.constraints[3] = "==0" #Susunan 4
problem29.constraints[4] = "==0" #Susunan 5
problem29.constraints[5] = "==0" #Susunan 6
problem29.constraints[6] = "==1" #Weldox 700E
problem29.constraints[7] = "==0" #Hardox 400
problem29.constraints[8] = "==0" #Domex Protect 500
problem29.constraints[9] = "==0" #Armox 560T
problem29.constraints[10] = "==0" #Al7075-T651
problem29.constraints[11] = "==0" #Kevlar-29
problem29.constraints[12] = "==0" #S2 Glass/SC15
problem29.constraints[13] = "==0" #CFRP
problem29.constraints[14] = ">=0.5" #thickness
problem29.constraints[15] = ">=1" #lapisan
problem29.directions[0] = Problem.MAXIMIZE
problem29.function = Opt

#Case30: Susunan 2 Weldox 700E Kevlar-29
problem30 = Problem(15, 1, 16)
problem30.types[:] = variable_types
problem30.constraints[0] = ">=0" #SEA
problem30.constraints[1] = "==1" #Susunan 2
problem30.constraints[2] = "==0" #Susunan 3
problem30.constraints[3] = "==0" #Susunan 4
problem30.constraints[4] = "==0" #Susunan 5
problem30.constraints[5] = "==0" #Susunan 6
problem30.constraints[6] = "==1" #Weldox 700E
problem30.constraints[7] = "==0" #Hardox 400
problem30.constraints[8] = "==0" #Domex Protect 500
problem30.constraints[9] = "==0" #Armox 560T
problem30.constraints[10] = "==0" #Al7075-T651
problem30.constraints[11] = "==1" #Kevlar-29
problem30.constraints[12] = "==0" #S2 Glass/SC15
problem30.constraints[13] = "==0" #CFRP
problem30.constraints[14] = ">=0.5" #thickness
problem30.constraints[15] = ">=1" #lapisan
problem30.directions[0] = Problem.MAXIMIZE
problem30.function = Opt

#Case31: Susunan 2 Weldox 700E S2Glass/SC15
problem31 = Problem(15, 1, 16)
problem31.types[:] = variable_types
problem31.constraints[0] = ">=0" #SEA
problem31.constraints[1] = "==1" #Susunan 2
problem31.constraints[2] = "==0" #Susunan 3
problem31.constraints[3] = "==0" #Susunan 4
problem31.constraints[4] = "==0" #Susunan 5
problem31.constraints[5] = "==0" #Susunan 6
problem31.constraints[6] = "==1" #Weldox 700E
problem31.constraints[7] = "==0" #Hardox 400
problem31.constraints[8] = "==0" #Domex Protect 500
problem31.constraints[9] = "==0" #Armox 560T
problem31.constraints[10] = "==0" #Al7075-T651
problem31.constraints[11] = "==0" #Kevlar-29
problem31.constraints[12] = "==1" #S2 Glass/SC15
problem31.constraints[13] = "==0" #CFRP
problem31.constraints[14] = ">=0.5" #thickness
problem31.constraints[15] = ">=1" #lapisan
problem31.directions[0] = Problem.MAXIMIZE
problem31.function = Opt

#Case32: Susunan 2 Weldox 700E CFRP
problem32 = Problem(15, 1, 16)
problem32.types[:] = variable_types
problem32.constraints[0] = ">=0" #SEA
problem32.constraints[1] = "==1" #Susunan 2
problem32.constraints[2] = "==0" #Susunan 3
problem32.constraints[3] = "==0" #Susunan 4
problem32.constraints[4] = "==0" #Susunan 5
problem32.constraints[5] = "==0" #Susunan 6
problem32.constraints[6] = "==1" #Weldox 700E
problem32.constraints[7] = "==0" #Hardox 400
problem32.constraints[8] = "==0" #Domex Protect 500
problem32.constraints[9] = "==0" #Armox 560T
problem32.constraints[10] = "==0" #Al7075-T651
problem32.constraints[11] = "==0" #Kevlar-29
problem32.constraints[12] = "==0" #S2 Glass/SC15
problem32.constraints[13] = "==1" #CFRP
problem32.constraints[14] = ">=0.5" #thickness
problem32.constraints[15] = ">=1" #lapisan
problem32.directions[0] = Problem.MAXIMIZE
problem32.function = Opt

#Case33: Susunan 2 Hardox 400 CNC
problem33 = Problem(15, 1, 16)
problem33.types[:] = variable_types
problem33.constraints[0] = ">=0" #SEA
problem33.constraints[1] = "==1" #Susunan 2
problem33.constraints[2] = "==0" #Susunan 3
problem33.constraints[3] = "==0" #Susunan 4
problem33.constraints[4] = "==0" #Susunan 5
problem33.constraints[5] = "==0" #Susunan 6
problem33.constraints[6] = "==0" #Weldox 700E
problem33.constraints[7] = "==1" #Hardox 400
problem33.constraints[8] = "==0" #Domex Protect 500
problem33.constraints[9] = "==0" #Armox 560T
problem33.constraints[10] = "==0" #Al7075-T651
problem33.constraints[11] = "==0" #Kevlar-29
problem33.constraints[12] = "==0" #S2 Glass/SC15
problem33.constraints[13] = "==0" #CFRP
problem33.constraints[14] = ">=0.5" #thickness
problem33.constraints[15] = ">=1" #lapisan
problem33.directions[0] = Problem.MAXIMIZE
problem33.function = Opt

#Case34: Susunan 2 Hardox 400 Kevlar-29
problem34 = Problem(15, 1, 16)
problem34.types[:] = variable_types
problem34.constraints[0] = ">=0" #SEA
problem34.constraints[1] = "==1" #Susunan 2
problem34.constraints[2] = "==0" #Susunan 3
problem34.constraints[3] = "==0" #Susunan 4
problem34.constraints[4] = "==0" #Susunan 5
problem34.constraints[5] = "==0" #Susunan 6
problem34.constraints[6] = "==0" #Weldox 700E
problem34.constraints[7] = "==1" #Hardox 400
problem34.constraints[8] = "==0" #Domex Protect 500
problem34.constraints[9] = "==0" #Armox 560T
problem34.constraints[10] = "==0" #Al7075-T651
problem34.constraints[11] = "==1" #Kevlar-29
problem34.constraints[12] = "==0" #S2 Glass/SC15
problem34.constraints[13] = "==0" #CFRP
problem34.constraints[14] = ">=0.5" #thickness
problem34.constraints[15] = ">=1" #lapisan
problem34.directions[0] = Problem.MAXIMIZE
problem34.function = Opt

#Case35: Susunan 2 Hardox 400 S2Glass/SC15
problem35 = Problem(15, 1, 16)
problem35.types[:] = variable_types
problem35.constraints[0] = ">=0" #SEA
problem35.constraints[1] = "==1" #Susunan 2
problem35.constraints[2] = "==0" #Susunan 3
problem35.constraints[3] = "==0" #Susunan 4
problem35.constraints[4] = "==0" #Susunan 5
problem35.constraints[5] = "==0" #Susunan 6
problem35.constraints[6] = "==0" #Weldox 700E
problem35.constraints[7] = "==1" #Hardox 400
problem35.constraints[8] = "==0" #Domex Protect 500
problem35.constraints[9] = "==0" #Armox 560T
problem35.constraints[10] = "==0" #Al7075-T651
problem35.constraints[11] = "==0" #Kevlar-29
problem35.constraints[12] = "==1" #S2 Glass/SC15
problem35.constraints[13] = "==0" #CFRP
problem35.constraints[14] = ">=0.5" #thickness
problem35.constraints[15] = ">=1" #lapisan
problem35.directions[0] = Problem.MAXIMIZE
problem35.function = Opt

#Case36: Susunan 2 Hardox 400 CFRP
problem36 = Problem(15, 1, 16)
problem36.types[:] = variable_types
problem36.constraints[0] = ">=0" #SEA
problem36.constraints[1] = "==1" #Susunan 2
problem36.constraints[2] = "==0" #Susunan 3
problem36.constraints[3] = "==0" #Susunan 4
problem36.constraints[4] = "==0" #Susunan 5
problem36.constraints[5] = "==0" #Susunan 6
problem36.constraints[6] = "==0" #Weldox 700E
problem36.constraints[7] = "==1" #Hardox 400
problem36.constraints[8] = "==0" #Domex Protect 500
problem36.constraints[9] = "==0" #Armox 560T
problem36.constraints[10] = "==0" #Al7075-T651
problem36.constraints[11] = "==0" #Kevlar-29
problem36.constraints[12] = "==0" #S2 Glass/SC15
problem36.constraints[13] = "==1" #CFRP
problem36.constraints[14] = ">=0.5" #thickness
problem36.constraints[15] = ">=1" #lapisan
problem36.directions[0] = Problem.MAXIMIZE
problem36.function = Opt
#Create problem-n definition for 51 other cases

#Iterasi setiap case
iteration =1000

i = 1
algorithm1 = NSGAII(problem1, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm1.run(iteration)
i = 2
algorithm2 = NSGAII(problem2, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm2.run(iteration)
i = 3
algorithm3 = NSGAII(problem3, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm3.run(iteration)
i = 4
algorithm4 = NSGAII(problem4, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm4.run(iteration)
i = 5
algorithm5 = NSGAII(problem5, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm5.run(iteration)
i = 6
algorithm6 = NSGAII(problem6, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm6.run(iteration)
i = 7
algorithm7 = NSGAII(problem7, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm7.run(iteration)
i = 8
algorithm8 = NSGAII(problem8, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm8.run(iteration)
i = 9
algorithm9 = NSGAII(problem9, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm9.run(iteration)
i = 10
algorithm10 = NSGAII(problem10, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm10.run(iteration)
i = 11
algorithm11 = NSGAII(problem11, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm11.run(iteration)
i = 12
algorithm12 = NSGAII(problem12, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm12.run(iteration)
i = 13
algorithm13 = NSGAII(problem13, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm13.run(iteration)
i = 14
algorithm14 = NSGAII(problem14, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm14.run(iteration)
i = 15
algorithm15 = NSGAII(problem15, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm15.run(iteration)
i = 16
algorithm16 = NSGAII(problem16, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm16.run(iteration)
i = 17
algorithm17 = NSGAII(problem17, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm17.run(iteration)
i = 18
algorithm18 = NSGAII(problem18, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm18.run(iteration)
i = 19
algorithm19 = NSGAII(problem19, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm19.run(iteration)
i = 20
algorithm20 = NSGAII(problem20, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm20.run(iteration)
i = 21
algorithm21 = NSGAII(problem21, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm21.run(iteration)
i = 22
algorithm22 = NSGAII(problem22, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm22.run(iteration)
i = 23
algorithm23 = NSGAII(problem23, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm23.run(iteration)
i = 24
algorithm24 = NSGAII(problem24, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm24.run(iteration)
i = 25
algorithm25 = NSGAII(problem25, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm25.run(iteration)
i = 26
algorithm26 = NSGAII(problem26, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm26.run(iteration)
i = 27
algorithm27 = NSGAII(problem27, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm27.run(iteration)
i = 28
algorithm28 = NSGAII(problem28, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm28.run(iteration)
i = 29
algorithm29 = NSGAII(problem29, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm29.run(iteration)
i = 30
algorithm30 = NSGAII(problem30, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm30.run(iteration)
i = 31
algorithm31 = NSGAII(problem31, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm31.run(iteration)
i = 32
algorithm32 = NSGAII(problem32, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm32.run(iteration)
i = 33
algorithm33 = NSGAII(problem33, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm33.run(iteration)
i = 34
algorithm34 = NSGAII(problem34, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm34.run(iteration)
i = 35
algorithm35 = NSGAII(problem35, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm35.run(iteration)
i = 36
algorithm36 = NSGAII(problem36, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
algorithm36.run(iteration)
#Create problem-i definition for 51 other cases

result1 = algorithm1.result
result2 = algorithm2.result
result3 = algorithm3.result
result4 = algorithm4.result
result5 = algorithm5.result
result6 = algorithm6.result
result7 = algorithm7.result
result8 = algorithm8.result
result9 = algorithm9.result
result10 = algorithm10.result
result11 = algorithm11.result
result12 = algorithm12.result
result13 = algorithm13.result
result14 = algorithm14.result
result15 = algorithm15.result
result16 = algorithm16.result
result17 = algorithm17.result
result18 = algorithm18.result
result19 = algorithm19.result
result20 = algorithm20.result
result21 = algorithm21.result
result22 = algorithm22.result
result23 = algorithm23.result
result24 = algorithm24.result
result25 = algorithm25.result
result26 = algorithm26.result
result27 = algorithm27.result
result28 = algorithm28.result
result29 = algorithm29.result
result30 = algorithm30.result
result31 = algorithm31.result
result32 = algorithm32.result
result33 = algorithm33.result
result34 = algorithm34.result
result35 = algorithm35.result
result36 = algorithm36.result

#Create result-i definition for 51 other cases

feasible_solutions = [s for s in result1 if s.feasible]
for solution in feasible_solutions:
    print(solution.objectives)

SEA_hasil1 = [s.constraints[0] for s in algorithm1.result if s.feasible]
SEA_hasil2 = [s.constraints[0] for s in algorithm2.result if s.feasible]
SEA_hasil3 = [s.constraints[0] for s in algorithm3.result if s.feasible]
SEA_hasil4 = [s.constraints[0] for s in algorithm4.result if s.feasible]
SEA_hasil5 = [s.constraints[0] for s in algorithm5.result if s.feasible]
SEA_hasil6 = [s.constraints[0] for s in algorithm6.result if s.feasible]
SEA_hasil7 = [s.constraints[0] for s in algorithm7.result if s.feasible]
SEA_hasil8 = [s.constraints[0] for s in algorithm8.result if s.feasible]
SEA_hasil9 = [s.constraints[0] for s in algorithm9.result if s.feasible]
SEA_hasil10 = [s.constraints[0] for s in algorithm10.result if s.feasible]
SEA_hasil11 = [s.constraints[0] for s in algorithm11.result if s.feasible]
SEA_hasil12 = [s.constraints[0] for s in algorithm12.result if s.feasible]
SEA_hasil13 = [s.constraints[0] for s in algorithm13.result if s.feasible]
SEA_hasil14 = [s.constraints[0] for s in algorithm14.result if s.feasible]
SEA_hasil15 = [s.constraints[0] for s in algorithm15.result if s.feasible]
SEA_hasil16 = [s.constraints[0] for s in algorithm16.result if s.feasible]
SEA_hasil17 = [s.constraints[0] for s in algorithm17.result if s.feasible]
SEA_hasil18 = [s.constraints[0] for s in algorithm18.result if s.feasible]
SEA_hasil19 = [s.constraints[0] for s in algorithm19.result if s.feasible]
SEA_hasil20 = [s.constraints[0] for s in algorithm20.result if s.feasible]
SEA_hasil21 = [s.constraints[0] for s in algorithm21.result if s.feasible]
SEA_hasil22 = [s.constraints[0] for s in algorithm22.result if s.feasible]
SEA_hasil23 = [s.constraints[0] for s in algorithm23.result if s.feasible]
SEA_hasil24 = [s.constraints[0] for s in algorithm24.result if s.feasible]
SEA_hasil25 = [s.constraints[0] for s in algorithm25.result if s.feasible]
SEA_hasil26 = [s.constraints[0] for s in algorithm26.result if s.feasible]
SEA_hasil27 = [s.constraints[0] for s in algorithm27.result if s.feasible]
SEA_hasil28 = [s.constraints[0] for s in algorithm28.result if s.feasible]
SEA_hasil29 = [s.constraints[0] for s in algorithm29.result if s.feasible]
SEA_hasil30 = [s.constraints[0] for s in algorithm30.result if s.feasible]
SEA_hasil31 = [s.constraints[0] for s in algorithm31.result if s.feasible]
SEA_hasil32 = [s.constraints[0] for s in algorithm32.result if s.feasible]
SEA_hasil33 = [s.constraints[0] for s in algorithm33.result if s.feasible]
SEA_hasil34 = [s.constraints[0] for s in algorithm34.result if s.feasible]
SEA_hasil35 = [s.constraints[0] for s in algorithm35.result if s.feasible]
SEA_hasil36 = [s.constraints[0] for s in algorithm36.result if s.feasible]

#Create SEA_hasil-i definition for 51 other cases

SEA_hasill=np.hstack((SEA_hasil1, 	SEA_hasil2, 	SEA_hasil3, 	SEA_hasil4, 	SEA_hasil5, 	SEA_hasil6, 	SEA_hasil7, 	SEA_hasil8, 	SEA_hasil9, 	SEA_hasil10, 	SEA_hasil11, 	SEA_hasil12, 	SEA_hasil13, 	SEA_hasil14, 	SEA_hasil15, 	SEA_hasil16, 	SEA_hasil17, 	SEA_hasil18, 	SEA_hasil19, 	SEA_hasil20, 	SEA_hasil21, 	SEA_hasil22, 	SEA_hasil23, 	SEA_hasil24, 	SEA_hasil25, 	SEA_hasil26, 	SEA_hasil27, 	SEA_hasil28, 	SEA_hasil29, 	SEA_hasil30, 	SEA_hasil31, 	SEA_hasil32, 	SEA_hasil33, 	SEA_hasil34, 	SEA_hasil35, 	SEA_hasil36))
#Create SEA_hasil for 51 other cases

print("SEA_hasill = ")
print(SEA_hasill)
Variable1 = np.reshape(np.array([s.variables for s in algorithm1.result if s.feasible]), (-1,15))
Variable2 = np.reshape(np.array([s.variables for s in algorithm2.result if s.feasible]), (-1,15))
Variable3 = np.reshape(np.array([s.variables for s in algorithm3.result if s.feasible]), (-1,15))
Variable4 = np.reshape(np.array([s.variables for s in algorithm4.result if s.feasible]), (-1,15))
Variable5 = np.reshape(np.array([s.variables for s in algorithm5.result if s.feasible]), (-1,15))
Variable6 = np.reshape(np.array([s.variables for s in algorithm6.result if s.feasible]), (-1,15))
Variable7 = np.reshape(np.array([s.variables for s in algorithm7.result if s.feasible]), (-1,15))
Variable8 = np.reshape(np.array([s.variables for s in algorithm8.result if s.feasible]), (-1,15))
Variable9 = np.reshape(np.array([s.variables for s in algorithm9.result if s.feasible]), (-1,15))
Variable10 = np.reshape(np.array([s.variables for s in algorithm10.result if s.feasible]), (-1,15))
Variable11 = np.reshape(np.array([s.variables for s in algorithm11.result if s.feasible]), (-1,15))
Variable12 = np.reshape(np.array([s.variables for s in algorithm12.result if s.feasible]), (-1,15))
Variable13 = np.reshape(np.array([s.variables for s in algorithm13.result if s.feasible]), (-1,15))
Variable14 = np.reshape(np.array([s.variables for s in algorithm14.result if s.feasible]), (-1,15))
Variable15 = np.reshape(np.array([s.variables for s in algorithm15.result if s.feasible]), (-1,15))
Variable16 = np.reshape(np.array([s.variables for s in algorithm16.result if s.feasible]), (-1,15))
Variable17 = np.reshape(np.array([s.variables for s in algorithm17.result if s.feasible]), (-1,15))
Variable18 = np.reshape(np.array([s.variables for s in algorithm18.result if s.feasible]), (-1,15))
Variable19 = np.reshape(np.array([s.variables for s in algorithm19.result if s.feasible]), (-1,15))
Variable20 = np.reshape(np.array([s.variables for s in algorithm20.result if s.feasible]), (-1,15))
Variable21 = np.reshape(np.array([s.variables for s in algorithm21.result if s.feasible]), (-1,15))
Variable22 = np.reshape(np.array([s.variables for s in algorithm22.result if s.feasible]), (-1,15))
Variable23 = np.reshape(np.array([s.variables for s in algorithm23.result if s.feasible]), (-1,15))
Variable24 = np.reshape(np.array([s.variables for s in algorithm24.result if s.feasible]), (-1,15))
Variable25 = np.reshape(np.array([s.variables for s in algorithm25.result if s.feasible]), (-1,15))
Variable26 = np.reshape(np.array([s.variables for s in algorithm26.result if s.feasible]), (-1,15))
Variable27 = np.reshape(np.array([s.variables for s in algorithm27.result if s.feasible]), (-1,15))
Variable28 = np.reshape(np.array([s.variables for s in algorithm28.result if s.feasible]), (-1,15))
Variable29 = np.reshape(np.array([s.variables for s in algorithm29.result if s.feasible]), (-1,15))
Variable30 = np.reshape(np.array([s.variables for s in algorithm30.result if s.feasible]), (-1,15))
Variable31 = np.reshape(np.array([s.variables for s in algorithm31.result if s.feasible]), (-1,15))
Variable32 = np.reshape(np.array([s.variables for s in algorithm32.result if s.feasible]), (-1,15))
Variable33 = np.reshape(np.array([s.variables for s in algorithm33.result if s.feasible]), (-1,15))
Variable34 = np.reshape(np.array([s.variables for s in algorithm34.result if s.feasible]), (-1,15))
Variable35 = np.reshape(np.array([s.variables for s in algorithm35.result if s.feasible]), (-1,15))
Variable36 = np.reshape(np.array([s.variables for s in algorithm36.result if s.feasible]), (-1,15))

#Create Variable-i definition for 51 other cases

Variables = np.vstack((Variable1, 	Variable2, 	Variable3, 	Variable4, 	Variable5, 	Variable6, 	Variable7, 	Variable8, 	Variable9, 	Variable10, 	Variable11, 	Variable12, 	Variable13, 	Variable14, 	Variable15, 	Variable16, 	Variable17, 	Variable18, 	Variable19, 	Variable20, 	Variable21, 	Variable22, 	Variable23, 	Variable24, 	Variable25, 	Variable26, 	Variable27, 	Variable28, 	Variable29, 	Variable30, 	Variable31, 	Variable32, 	Variable33, 	Variable34, 	Variable35, 	Variable36))
#Create Variable-i definition for 51 other cases

print("Variables = ")
print(Variables)
solutionwrite = pd.ExcelWriter('NSGA Solutions1.xlsx', engine='openpyxl')
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















