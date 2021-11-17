import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
import os
import scipy.io as sio
from functools import reduce
import time
from JensenTools import *


tol = .025 #tolerance allowed between measured and calculated energy.
# Emeas = [131,330,370,730] #Boothroyds measured E for PrO2
# Emeas = [255] #Kern first measured CF Level for BaPrO3
# Emeas = convertCMtomeV([2013,3141,5285,6604]) #Four optical measured levels for BaPrO3 from Popova
Emeas = [168, 335,385] #Our measured levels of Sr2PrO4
print('Energies as measured by paper (mev):  ', Emeas)

numlevels = 4

xmin, xmax = -1,1
bpfmin, bpfmax =  -1,1
numx, numbpf = 200, 200

comp = 'Sr2PrO4'
runDir = 'cubic_matrix/'


# save, load, findCoords = True, False, False
save, load, findCoords = False, True, True

# LSList = [100]
# LSList = [80]

if(save):
	if __name__ == '__main__':
		saveMatrixPar(xmin,xmax,numx,bpfmin,bpfmax,numbpf,LSList,runDir,comp,numlevels)


if(load):
	LSNames, EList, data = loadMatrix(runDir)
	for c in range(len(LSNames)):

		#Choose which bands to look for compatibilities.
		#For Boothroyd with 4 reported levels. I use indices 1-4.
	


		#Loading the x,bpf, and LS of each file.
		x = data[c]['X'][0]
		bpf = data[c]['B'][0]
		LS = data[c]['LS'][0][0]

		plotContours(data[c],EList[c],Emeas,LSNames[c]) #Contour plotting for 4 E levels


		if(findCoords):

			index = [1,2,3]
			Eindex = []
			EListindex = []
			for i in index:
				Eindex.append(Emeas[i-1])
				EListindex.append(EList[c][i-1])
			
			coords = paramFinder(data[c],EListindex,Eindex,tol,comp,LSNames[c])


			if len(coords) !=0:
				for j in [coords[0],coords[len(coords)-1]]:
					print('With x = ', x[j[0]], ' and bpf = ', bpf[j[1]])
					count = 1
					for i in EList[c]:
						print('E%i = '%count, data[c][i][j[0]][j[1]], 'meV')
						count += 1
					print()
			else:
				print('No compatibilities found')


			if(len(coords) != 0):
				print('\nFor ', LSNames[c])
				xind,bind = coords[0][0], coords[0][1]

				print('\nFor ', comp, ' at x[%i] = %.4f and bpf[%i] = %.4f'%(xind,x[xind],bind,bpf[bind]))
				print(EList[c][1], ' = ', data[c][EList[c][1]][xind][bind] )
				print('Using these values lets see if degeneracies are protected.\n')
				printPCFEigens(x[xind],bpf[bind],LS)

			# if(len(coords) != 0):
			# 	for i in coords:
			# 		if i[1] < 0:
			# 			print('\nFor ', LSNames[c])
			# 			xind,bind = i[0], i[1]

			# 			print('\nFor ', comp, ' at x[%i] = %.4f and bpf[%i] = %.4f'%(xind,x[xind],bind,bpf[bind]))
			# 			print(EList[c][1], ' = ', data[c][EList[c][1]][xind][bind] )
			# 			print('Using these values lets see if degeneracies are protected.\n')
			# 			printPCFEigens(x[xind],bpf[bind],LS)			

			# #Checking the energy value at the same (x,bpf) for LS = 5, 500
			# print('At x = %.2f, bpf = %.2f excited level %s = %.2f '%(x[25],bpf[25],EList[c][1],data[c][EList[c][1]][25][25]))
			# printPCFEigens(x[25],bpf[25],LS)


# plt.show()
