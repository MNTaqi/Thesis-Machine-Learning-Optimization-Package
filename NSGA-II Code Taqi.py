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
os.chdir(r"C:\Users\Taqi\OneDrive - Institut Teknologi Bandung\Desktop\Kuliah\S7\TA\.Phyton\ANN\.Model")
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

SEA_hasill=np.hstack((SEA_hasil1, 	SEA_hasil2, 	SEA_hasil3, 	SEA_hasil4, 	SEA_hasil5, 	SEA_hasil6, 	SEA_hasil7, 	SEA_hasil8, 	SEA_hasil9, 	SEA_hasil10, 	SEA_hasil11, 	SEA_hasil12, 	SEA_hasil13, 	SEA_hasil14, 	SEA_hasil15, 	SEA_hasil16, 	SEA_hasil17, 	SEA_hasil18, 	SEA_hasil19, 	SEA_hasil20, 	SEA_hasil21, 	SEA_hasil22, 	SEA_hasil23, 	SEA_hasil24, 	SEA_hasil25, 	SEA_hasil26, 	SEA_hasil27, 	SEA_hasil28, 	SEA_hasil29, 	SEA_hasil30, 	SEA_hasil31, 	SEA_hasil32, 	SEA_hasil33, 	SEA_hasil34, 	SEA_hasil35, 	SEA_hasil36, 	SEA_hasil37, 	SEA_hasil38, 	SEA_hasil39, 	SEA_hasil40, 	SEA_hasil41, 	SEA_hasil42, 	SEA_hasil43, 	SEA_hasil44, 	SEA_hasil45, 	SEA_hasil46, 	SEA_hasil47, 	SEA_hasil48, 	SEA_hasil49, 	SEA_hasil50, 	SEA_hasil51, 	SEA_hasil52, 	SEA_hasil53, 	SEA_hasil54, 	SEA_hasil55, 	SEA_hasil56, 	SEA_hasil57, 	SEA_hasil58, 	SEA_hasil59, 	SEA_hasil60, 	SEA_hasil61, 	SEA_hasil62, 	SEA_hasil63, 	SEA_hasil64, 	SEA_hasil65, 	SEA_hasil66, 	SEA_hasil67, 	SEA_hasil68, 	SEA_hasil69, 	SEA_hasil70, 	SEA_hasil71, 	SEA_hasil72, 	SEA_hasil73, 	SEA_hasil74, 	SEA_hasil75, 	SEA_hasil76, 	SEA_hasil77, 	SEA_hasil78, 	SEA_hasil79, 	SEA_hasil80, 	SEA_hasil81, 	SEA_hasil82, 	SEA_hasil83, 	SEA_hasil84, 	SEA_hasil85, 	SEA_hasil86, 	SEA_hasil87, 	SEA_hasil88, 	SEA_hasil89, 	SEA_hasil90, 	SEA_hasil91, 	SEA_hasil92, 	SEA_hasil93, 	SEA_hasil94, 	SEA_hasil95, 	SEA_hasil96, 	SEA_hasil97, 	SEA_hasil98, 	SEA_hasil99, 	SEA_hasil100, 	SEA_hasil101, 	SEA_hasil102, 	SEA_hasil103, 	SEA_hasil104, 	SEA_hasil105, 	SEA_hasil106, 	SEA_hasil107, 	SEA_hasil108, 	SEA_hasil109, 	SEA_hasil110, 	SEA_hasil111, 	SEA_hasil112, 	SEA_hasil113, 	SEA_hasil114, 	SEA_hasil115, 	SEA_hasil116, 	SEA_hasil117, 	SEA_hasil118, 	SEA_hasil119, 	SEA_hasil120, 	SEA_hasil121, 	SEA_hasil122, 	SEA_hasil123, 	SEA_hasil124, 	SEA_hasil125, 	SEA_hasil126, 	SEA_hasil127, 	SEA_hasil128, 	SEA_hasil129, 	SEA_hasil130, 	SEA_hasil131, 	SEA_hasil132, 	SEA_hasil133, 	SEA_hasil134, 	SEA_hasil135, 	SEA_hasil136, 	SEA_hasil137, 	SEA_hasil138, 	SEA_hasil139, 	SEA_hasil140, 	SEA_hasil141, 	SEA_hasil142, 	SEA_hasil143, 	SEA_hasil144))
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

Variables = np.vstack((Variable1, 	Variable2, 	Variable3, 	Variable4, 	Variable5, 	Variable6, 	Variable7, 	Variable8, 	Variable9, 	Variable10, 	Variable11, 	Variable12, 	Variable13, 	Variable14, 	Variable15, 	Variable16, 	Variable17, 	Variable18, 	Variable19, 	Variable20, 	Variable21, 	Variable22, 	Variable23, 	Variable24, 	Variable25, 	Variable26, 	Variable27, 	Variable28, 	Variable29, 	Variable30, 	Variable31, 	Variable32, 	Variable33, 	Variable34, 	Variable35, 	Variable36, 	Variable37, 	Variable38, 	Variable39, 	Variable40, 	Variable41, 	Variable42, 	Variable43, 	Variable44, 	Variable45, 	Variable46, 	Variable47, 	Variable48, 	Variable49, 	Variable50, 	Variable51, 	Variable52, 	Variable53, 	Variable54, 	Variable55, 	Variable56, 	Variable57, 	Variable58, 	Variable59, 	Variable60, 	Variable61, 	Variable62, 	Variable63, 	Variable64, 	Variable65, 	Variable66, 	Variable67, 	Variable68, 	Variable69, 	Variable70, 	Variable71, 	Variable72, 	Variable73, 	Variable74, 	Variable75, 	Variable76, 	Variable77, 	Variable78, 	Variable79, 	Variable80, 	Variable81, 	Variable82, 	Variable83, 	Variable84, 	Variable85, 	Variable86, 	Variable87, 	Variable88, 	Variable89, 	Variable90, 	Variable91, 	Variable92, 	Variable93, 	Variable94, 	Variable95, 	Variable96, 	Variable97, 	Variable98, 	Variable99, 	Variable100, 	Variable101, 	Variable102, 	Variable103, 	Variable104, 	Variable105, 	Variable106, 	Variable107, 	Variable108, 	Variable109, 	Variable110, 	Variable111, 	Variable112, 	Variable113, 	Variable114, 	Variable115, 	Variable116, 	Variable117, 	Variable118, 	Variable119, 	Variable120, 	Variable121, 	Variable122, 	Variable123, 	Variable124, 	Variable125, 	Variable126, 	Variable127, 	Variable128, 	Variable129, 	Variable130, 	Variable131, 	Variable132, 	Variable133, 	Variable134, 	Variable135, 	Variable136, 	Variable137, 	Variable138, 	Variable139, 	Variable140, 	Variable141, 	Variable142, 	Variable143, 	Variable144))
#Create Variable-i definition for 51 other cases

print("Variables = ")
print(Variables)
solutionwrite = pd.ExcelWriter('NSGA Solutions.xlsx', engine='openpyxl')
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















