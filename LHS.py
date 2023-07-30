import numpy as np
from math import factorial
import xlsxwriter
import os
from mpl_toolkits.mplot3d  import Axes3D
import matplotlib.pyplot as plt
from random import randrange
import random as rn
from IPython.display import display
import pandas as pd
import math
import streamlit as st
st.cache_data.clear()

rn.seed(0)
os.chdir(r"C:\Users\Taqi\OneDrive - Institut Teknologi Bandung\Desktop\Kuliah\S7\TA\.Phyton")
__all__ = ['lhs']
def lhs(n, samples=None, criterion=None, iterations=None):
    H = None
    if samples is None:
        samples = n
    if criterion is not None:
        assert criterion.lower() in ('center', 'c', 'maximin', 'm','centermaximin', 'cm', 'centermaximin1'), 'Invalid value for "criterion": {}'.format(criterion)
    if criterion is None:
        criterion = 'center'
    if iterations is None:
        iterations = 5
    if H is None:
        if criterion.lower() in ('center', 'c'):
            H = _lhscentered(n, samples)
        elif criterion.lower() in ('maximin','m'):
            H = _lhsmaximin(n, samples, iterations, 'maximin')
        elif criterion.lower() in ('centermaximin', 'cm'):
            H = _lhsmaximin(n, samples, iterations, 'centermaximin')
        elif criterion.lower() in ('centermaximin1', 'cm'):
            H = _lhsmaximin1(n, samples, iterations, 'centermaximin1')
    return H
def _lhscentered(n, samples):
    # Generate the intervals
    cut = np.linspace(0, 1, samples)
    u = np.random.rand(samples, n)
    # Make the random pairings
    H = np.zeros_like(u)
    for j in range(n):
        H[:, j] = np.random.permutation(cut)
    return H
##################################################################
def _lhsmaximin(n, samples, iterations,lhstype):
    maxdist = 0
    # Maximize the minimum distance between points
    for i in range(iterations):
        Hcandidate = _lhscentered(n,samples)
        d = _pdist(Hcandidate)
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = Hcandidate.copy()
    return H
##################################################################
def _pdist(x):
    x = np.atleast_2d(x)
    assert len(x.shape) == 2
    m, n = x.shape
    if m < 2:
        return []
    d = []
    for i in range(m - 1):
        for j in range(i + 1, m):
            d.append((sum((x[j, :] - x[i,:]) ** 2)) ** 0.5)
    return np.array(d)
def round_up_to_even(f):
    return np.ceil(f / 2.) * 2
rn.seed(0)
##################################################################
arrangement = randrange(1, 7) #variate arrangement
x = 0
for x in range(9):
    arrangement = np.vstack([arrangement, randrange(1, 7)])
    arrangement2=randrange(1,7)
    z=0
    for z in range (9):
        arrangement2 = np.vstack([arrangement2, randrange(1, 7)])
        z=z+1
    arrangement = np.vstack([arrangement, arrangement2])
    x = x + 1
###################################
metalmattype = randrange(1, 7) #
x = 0
for x in range(9):
    metalmattype = np.vstack([metalmattype, randrange(1, 7)])
    metalmattype2=randrange(1,7)
    z=0
    for z in range (9):
        metalmattype2 = np.vstack([metalmattype2, randrange(1, 7)])
        z=z+1
    metalmattype = np.vstack([metalmattype, metalmattype2])
    x = x + 1

compmattype = randrange(1, 5) #
x = 0
for x in range(9):
    compmattype = np.vstack([compmattype, randrange(1, 5)])
    compmattype2=randrange(1,5)
    z=0
    for z in range (9):
        compmattype2 = np.vstack([compmattype2, randrange(1, 5)])
        z=z+1
    compmattype = np.vstack([compmattype, compmattype2])
    x = x + 1
###################################
material = lhs(4, samples=100, criterion='centermaximin')
display(material)
material[:,0]=metalmattype[:,0] #metal material type, 1-6
material[:,1]=1+(np.round(material[:,1]*4,1)) #metal thickness, 1-5 mm step 0.1
material[:,2]=compmattype[:,0] #composite material type, 1-4
material[:,3]=4*(1+np.fix(14*material[:,3])) #layerkomposit, 4-60 step 4
###################################
display(arrangement)
display(material)
##################################################################
array=np.hstack([arrangement, material])
##################################################################
title = [["Susunan"],["Metal Material"],["Metal Thickness"],["Composite Material"],["Composite Layer"]]
arraytranspose=array.transpose()
data = arraytranspose[1,:]
data2 = arraytranspose[3,:]
arraytranspose = np.hstack((title, arraytranspose))
metal_material=[]
composite_material=[]
for i in range(100):
    if data[i]==1:
        metal_material.append('Weldox 500E')
    if data[i]==2:
        metal_material.append('Weldox 700E')
    if data[i]==3:
        metal_material.append('Hardox 400')
    if data[i]==4:
        metal_material.append('Domex Pro.500')
    if data[i]==5:
        metal_material.append('Armox 560T')
    if data[i]==6:
        metal_material.append('Al7075-T651')
    i=i+1
for i in range(100):
    if data2[i]==1:
        composite_material.append('CnC')
    if data2[i]==2:
        composite_material.append('Kevlar-29')
    if data2[i]==3:
        composite_material.append('S2-Glass/SC15')
    if data2[i]==4:
        composite_material.append('CFRP')
    i=i+1
wel500e_count = metal_material.count('Weldox 500E')
wel700e_count = metal_material.count('Weldox 700E')
hard400_count = metal_material.count('Hardox 400')
dompro500_count = metal_material.count('Domex Pro.500')
arm560t_count = metal_material.count('Armox 560T')
al7075_count = metal_material.count('Al7075-T651')
cnc_count = composite_material.count('CnC')
kevlar_count = composite_material.count('Kevlar-29')
s2_count = composite_material.count('S2-Glass/SC15')
cfrp_count = composite_material.count('CFRP')
#print('hasil itung')
#display(st37_count)
#display(al5083_count)
#display(cnc_count)
#display(kevlar_count)
#display(gfrp_count)
#display(cfrp_count)
#display(seratrami_count)
metal_mat=['Weldox 500E','Weldox 700E','Hardox 400','Domex Pro.500','Armox 560T','Al7075-T651']
comp_mat=['CnC','Kevlar-29','S2-Glass/SC15','CFRP']
count=[wel500e_count,wel700e_count,hard400_count,dompro500_count,arm560t_count,al7075_count]
count2=[cnc_count,kevlar_count,s2_count,cfrp_count]
#print('material dan jumlah')
#display(metal_mat)
#display(count)
#display(comp_mat)
#display(count2)
#print('array')
#display(array)
from matplotlib.ticker import AutoMinorLocator
plt.figure(1)
fig, ax = plt.subplots(2, 1, figsize=[10,15])
plt.minorticks_on()
ax[0].scatter(x=array[:,0], y=array[:,2])
ax[0].set_xlabel("Susunan")
ax[0].set_ylabel("Metal Thickness (mm)")
ax[1].scatter(x=array[:,0], y=array[:,4])
ax[1].set_xlabel("Susunan")
ax[1].set_ylabel("Composite Layer")
plt.figure(2)
fig, ax = plt.subplots(1, 2, figsize=[15,25])
ax[0].bar(metal_mat, count)
ax[0].set_xlabel("Metal Material")
ax[0].set_ylabel("Value")
ax[1].bar(comp_mat, count2)
ax[1].set_xlabel("Composite Material")
ax[1].set_ylabel("Value")
plt.show()
workbook = xlsxwriter.Workbook('arrays.xlsx',{'strings_to_numbers': True})
worksheet = workbook.add_worksheet()
row = 0
#print('arraytranspose')
#print(arraytranspose)
for col, data in enumerate(arraytranspose):
    worksheet.write_column(row, col, data)
workbook.close()
fig = plt.figure(2, figsize=(8, 6), dpi=80)
ax = fig.add_subplot(111, projection='3d')
n = 100
# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = array[:,0]
    ys = array[:,2]
    zs = array[:,4]
ax.scatter(xs, ys, zs, c=c, marker=m)
ax.set_xlabel('Susunan')
ax.set_ylabel('Metal Thickness')
ax.set_zlabel('Composite Layer')
plt.show()