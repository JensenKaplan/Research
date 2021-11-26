import numpy as np
import matplotlib.pyplot as plt
import PyCrystalField as cef
import os
from lmfit import Model
import scipy.io as sio
from functools import reduce
import time
from sklearn.cluster import KMeans
import multiprocessing as mp
from itertools import product
from functools import partial
import pandas as pd 

def getSaveDir(name = 'm', comp = 'Sr2PrO4', dataType = None):
	if name == 'm':
		dataDir = '/Users/jensenkaplan/Dropbox (GaTech)/Jensen/{}/'.format(comp)		
	elif name =='w':
		dataDir =  "C:/Users/jense/Dropbox (GaTech)/Jensen/{}/".format(comp)
	else:
		"ERROR: Please use 'w' for Windows, 'm' for Mac." 
		return

	if dataType == 'grid':
		dataDir = dataDir + 'CubicGridSearch'
	elif dataType =='MH':
		dataDir = dataDir + 'MVSH/'
	elif dataType == 'MT':
		dataDir = dataDir + 'MVST/'
	else:
		return dataDir
	return dataDir

	

#Self made functions for grid search calculations
#####################################################################################################################################################################
def saveMatrixPar(xmin,xmax,numx,bpfmin,bpfmax,numbpf, runDir,**kwargs):
	x = np.linspace(xmin,xmax,numx)
	bpf = np.linspace(bpfmin,bpfmax,numx)

	print('Xmin = %0.3f to Xmax = %.3f\nBPFmin = %.3f to BPFmax = %.3f \nwith number of steps in X = %s, Bpf = %s'%(xmin,xmax,bpfmin,bpfmax,numx,numbpf))
	savedict = {'X': x, 'B': bpf}

	if (os.path.exists(runDir) == False):
		os.makedirs(runDir)

	if kwargs['LS_on']:
		LSList = kwargs['LSList']
		for j in range(len(LSList)):
			savedict['LS'] = LSList[j]
			with mp.Pool() as P:
				E = P.starmap(partial(energyCalcKPar,LSValue = LSList[j], **kwargs),product(x,bpf))
				E = np.reshape(E,(numx,-1,kwargs['numlvls']))
				# P.close()
				for i in range(np.shape(E)[2]):
					savedict['E%i'%(i+1)] = E[:,:,i]
			sio.savemat(runDir+'LS_%i.mat'%LSList[j], savedict)	
	else:
		with mp.Pool() as P:
			E = P.starmap(partial(energyCalcKPar, **kwargs),product(x,bpf))
			# print(len(E))
			E = np.reshape(E,(numx,numbpf,-1))

		for i in range(np.shape(E)[2]):
			savedict['E%i'%(i+1)] = E[:,:,i]
		sio.savemat(runDir+kwargs['grid'], savedict)	

def energyCalcKPar(x,bpf, **kwargs):
	Stev = {}

	Stev['B40'] = bpf
	Stev['B60'] = x*bpf
	Stev['B44'] = 5*Stev['B40']
	Stev['B64'] = -21*Stev['B60']
	if kwargs['LS_on']:
		kwargs['LSValue']
		Pr = cef.LS_CFLevels.Bdict(Bdict=Stev, L=kwargs['L'], S= kwargs['S'], SpinOrbitCoupling= kwargs['LSValue'])
		Pr.diagonalize()
		e = kmeansSort(Pr.eigenvalues,kwargs['numlvls'])
	else:
		Pr = cef.CFLevels.Bdict(Bdict = Stev,ion = kwargs['ion'])
		Pr.diagonalize()
		e = Pr.eigenvalues
	return e

# This function loads my .mat files for analyzing, plotting, finding compatibilities.
# THIS WORKS since it can properly use my previously generated PrO2 data.
def loadMatrix(runDir, **kwargs):
	matList = os.listdir(runDir) #The different LS calculations
	if kwargs['LS_on']:
		#Where to save and load the energy data for contours
		matList = os.listdir(runDir) #The different LS calculations
		LSNames = [] #Saving the LS names for plotting
		for i in matList:
			s = i.split('.')[0].split('_')
			LSNames.append((s[0]+ ' = ' + s[1] + ' meV'))
		dataList = []
		EList = []

		for i in matList:
			data = sio.loadmat(runDir + i)
			E = []
			for i in data.keys():
				if 'E' in i:
					E.append(i)
			dataList.append(data)
			EList.append(E)

		return LSNames, EList, dataList

	else:
		# print(matList)
		data = sio.loadmat(runDir+matList[2])
		E = []
		# print(data.keys())
		for i in data.keys():
			if 'E' in i:
				E.append(i)
		return E, data		

# Takes a filename and data directory
# Returns the field, magnetic moment, error, and name (temperature) of the run
# field, moment, and error are returned as np arrays
# name is returned as a string
## Note that the name handling is hardcoded for how my lab conventionally names our files.
def getData(magrun, dataDir,**kwargs):
	who = kwargs['who']
	dataType = kwargs['dataType']
	if who == 'Arun':
	    name = magrun.split('_')[-1].split('.')[0]
	    mass = magrun.split('_')[3]
	    mass = mass.replace('P','.')
	    mass = mass[:-2]
	    measType = magrun.split('_')[-1].split('.')[0]
	    

	    f = open(dataDir + magrun)
	    while f.readline().strip() != '[Data]':
	        pass
	    df = pd.read_csv(f)
	    df.dropna(subset = ['Magnetic Field (Oe)','M. Std. Err. (emu)'],inplace = True)
	    T = np.array(df['Temperature (K)'])
	    H = np.array(df['Magnetic Field (Oe)'])
	    E = np.array(df['M. Std. Err. (emu)'])
	    M = np.array(df['Moment (emu)'])
	    mass = getMass(magrun, **kwargs)
	    measType = magrun.split('_')[-1].split('.')[0]
	    # print(measType)
	    if dataType == 'MH':
	    	return H, M, E, name
	    if dataType == 'MT':
	    	return M,H,T,E, mass, measType

	elif who == 'PPMS':
		name = magrun.split('_')[4].split('.')[0]
		name = name.replace('P','.')
		mass = getMass(magrun,**kwargs)
		print(mass)
		df = pd.read_csv(dataDir + magrun)
		df.dropna(inplace = True)
		T = np.array(df['Temperature (K)'])
		H = np.array(df['Magnetic Field (Oe)'])
		E = np.array(df['M. Std. Err. (emu)'])
		M = np.array(df['Moment (emu)'])

		if dataType == 'MH':
			return H, M, E, name
		if dataType == 'MT':
			measType = magrun.split('_')[5].split('.')[0]
			return M,H,T,E, mass, measType


# contour plotting function for all energy bands
def plotContours(data,EList,**kwargs):

	numplots = len(EList)
	# print()
	if(numplots == 2):
		snum = 2
	elif (numplots%2 == 0):
		snum = np.ceil(np.sqrt(numplots))
	else:
		snum = np.sqrt(numplots) + 1

	if kwargs['LS_on']:
		plt.figure()
		for i in range(1,numplots+1):
			ax = plt.subplot(snum,snum,i)
			# print(data['X'])
			mapp = ax.contourf(data['X'][0],data['B'][0],data[EList[i-1]])
			# print(np.shape(data[EList[i-1]]))
			ax.set(xlabel = 'Ratio of B60/B40', ylabel = 'B Prefactor', title = EList[i-1])
			cbar = plt.colorbar(mapp,ax = ax)
			cbar.set_label('Energy (meV)')

		plt.tight_layout(h_pad = -1, w_pad = -2)
		plt.suptitle(kwargs['LSName'])

	else:
		plt.figure()
		for i in range(1,numplots+1):
			ax = plt.subplot(snum,snum,i)
			# print((data[EList[0]]))
			mapp = ax.contourf(data['X'][0],data['B'][0],data[EList[i-1]])
			# print(np.shape(data[EList[i-1]]))
			ax.set(xlabel = 'Ratio of B60/B40', ylabel = 'B Prefactor', title = EList[i-1])
			cbar = plt.colorbar(mapp,ax = ax)
			cbar.set_label('Energy (meV)')
		plt.tight_layout(h_pad = -1, w_pad = -2)
	# plt.show()
	return	

#Function that finds compatibility coordinates within a certain tolerance.
#Works as follows:
#Check if E1 is within tolerance, add (x,bpf) coords to list
#Check the same for E2, etc.
#Then only keep the coordinates that appear in all of the energy bands
def paramFinder(data,band,E,tolerance,comp,**kwargs):

	if kwargs['LS_on']:
		LSName = kwargs['LSName']
		print('\nParameter search for: ',' Compound: ', comp, ' at ', LSName, 'with %0.3f tolerance.' %tolerance)
	else:
		print('\nParameter search for: ',' Compound: ', comp, ' with %0.3f tolerance.' %tolerance)

	coords = []
	#The first part that only care about individual energy band and toleranc 
	for i in range(len(E)):
		for j in range(len(data[band[i]])):
			for k in range(len(data[band[i]][j])):
				if not (np.isnan(data[band[i]][j][k])):
					if (((1-tolerance)*E[i] <= data[band[i]][j][k]) and ((1+tolerance)*E[i] >=  data[band[i]][j][k])):
						temp = [data[band[i]][j][k]]
						if ([j,k] not in coords):
							coords.append([j,k])

	newCoords = []
	#The second part that only leaves the coordinates that fall within all four bands.
	for i in coords:
		allbands = []
		for j in range(len(E)):
			if ((1-tolerance)*E[j] <= data[band[j]][i[0]][i[1]] and ((1+tolerance)*E[j] >= data[band[j]][i[0]][i[1]])):
				allbands.append(0)
				# print(data[band[j]][i[0]][i[1]])
			else:
				allbands.append(1)
		if 1 not in allbands:
			newCoords.append(i)

	return newCoords


def normSusc(M,H,MErr,molweight,samplemass,massErr):
	XNorm = []
	XErrNorm = []
	M = emuToBohr(M,mass = samplemass, molweight = molweight)
	MErr = emuToBohr(MErr,mass = samplemass, molweight = molweight)
	H = oeToTesla(H)
	for i in range(len(M)):
	    XNorm.append(M[i]/H[i]*molweight/samplemass/1)
	    XErrNorm.append(MErr[i]/H[i]*molweight/samplemass + M[i]/H[i]*molweight*massErr/(samplemass**2))

	Xi = 1/np.array(XNorm)
	XiErr = np.array(XErrNorm)/np.array(XNorm)**2
	return XNorm, XErrNorm, Xi, XiErr


def saveEvsLS(E,LS,runDir):
	savedict = {'LS': LS}
	E1 = []
	E2 = []
	E3 = []
	E4 = []
	E5 = [] 
	E6 = []
	E7 = []
	E8 = []
	E9 = []
	E10 = []
	E11 = []
	E12 = []
	E13 = []
	E14 = []
	for i in E:
		E1.append(i[0])
		E2.append(i[1])
		E3.append(i[2])
		E4.append(i[3])
	# print(E1)		
	savedict['E1'] = E1
	savedict['E2'] = E2
	savedict['E3'] = E3
	savedict['E4'] = E4
	if (os.path.exists(runDir) == False):
		os.makedirs(runDir)
	sio.savemat(runDir + '4levels',savedict)
	return

def saveEvsLS14(E,LS,runDir):
	savedict = {'LS': LS}
	E1 = []
	E2 = []
	E3 = []
	E4 = []
	E5 = [] 
	E6 = []
	E7 = []
	E8 = []
	E9 = []
	E10 = []
	E11 = []
	E12 = []
	E13 = []
	E14 = []
	for i in E:
		E1.append(i[0])
		E2.append(i[1])
		E3.append(i[2])
		E4.append(i[3])
		E5.append(i[4])
		E6.append(i[5])
		E7.append(i[6])
		E8.append(i[7])
		E9.append(i[8])
		E10.append(i[9])
		E11.append(i[10])
		E12.append(i[11])
		E13.append(i[12])
		E14.append(i[13])
	# print(E1)		
	savedict['E1'] = E1
	savedict['E2'] = E2
	savedict['E3'] = E3
	savedict['E4'] = E4
	savedict['E5'] = E5
	savedict['E6'] = E6
	savedict['E7'] = E7
	savedict['E8'] = E8
	savedict['E9'] = E9
	savedict['E10'] = E10
	savedict['E11'] = E11
	savedict['E12'] = E12
	savedict['E13'] = E13
	savedict['E14'] = E14

	if (os.path.exists(runDir) == False):
		os.makedirs(runDir)
	sio.savemat(runDir + '14levels',savedict)
	return


def loadEvsLS14(runDir):
	runList = os.listdir(runDir)
	data = sio.loadmat(runDir + '14levels')
	LS = data['LS'][0]
	E1 = data['E1'][0]
	E2 = data['E2'][0]
	E3 = data['E3'][0]
	E4 = data['E4'][0]
	E5 = data['E5'][0]
	E6 = data['E6'][0]
	E7 = data['E7'][0]
	E8 = data['E8'][0]
	E9 = data['E9'][0]
	E10 = data['E10'][0]
	E11 = data['E11'][0]
	E12 = data['E12'][0]
	E13 = data['E13'][0]
	E14 = data['E14'][0]
	return LS,E1,E2,E3,E4,E5,E6,E7,E8,E9,E10,E11,E12,E13,E14

def loadEvsLS(runDir):
	runList = os.listdir(runDir)
	data = sio.loadmat(runDir + '4levels.mat')
	LS = data['LS'][0]
	E1 = data['E1'][0]
	E2 = data['E2'][0]
	E3 = data['E3'][0]
	E4 = data['E4'][0]
	# print(np.shape(E1))
	return LS, E1, E2, E3, E4


def energyCalcKPar14(LS,x = 0.0352 ,bpf = -0.3970, numlevels = 4):
	numlevels = numlevels
	Stev = {}

	Stev['B40'] = bpf
	Stev['B60'] = x*bpf
	Stev['B44'] = 5*Stev['B40']
	Stev['B64'] = -21*Stev['B60']

	Pr = cef.LS_CFLevels.Bdict(Bdict=Stev, L=3, S=0.5, SpinOrbitCoupling=LS)
	Pr.diagonalize()
	e = Pr.eigenvalues
	return e

#New K-Means sorting which uses ML to cluster and track the energy bands.
def kmeansSort(e,numlevels):
	km = KMeans(numlevels+1) #5 clusters. One for each excited energy level (4) and one for the ground state.
	pred_y = km.fit(e.reshape(-1,1))
	centers = pred_y.cluster_centers_
	finalEvalList = []
	for j in centers:
		data_shift = list(np.abs(e-j))
		i = data_shift.index(min(list(data_shift)))
		finalEvalList.append(e[i])
	finalEvalList = np.sort(finalEvalList).tolist()
	return finalEvalList[1:] #This excludes the lowest (0 energy) mode

#For converting Popova's optical measurements to meV
def convertCMtomeV(e):
	converted = []
	for i in e:
		converted.append(i/8.065)
	return converted

# for checking eigenvalues (and hence energies) at a given (x,bpf) coordinate
def printPCFEigens(x,bpf, **kwargs):
	Stev={'B40': bpf, 'B60': x*bpf}
	Stev['B44'] = 5*Stev['B40']
	Stev['B64'] = -21*Stev['B60']
	if kwargs['LS_on']:
		Pr = cef.LS_CFLevels.Bdict(Bdict=Stev, L=3, S=0.5, SpinOrbitCoupling=kwargs['LS'])
	else:
		Pr = cef.CFLevels.Bdict(Bdict = stev, ion = kwargs['ion'])
	Pr.diagonalize()
	Pr.printEigenvectors()
	return


#Takes a list of moments (in emu), sample mass, and molecular weight
#Returns list of moments (in Bohr Magnetons)
def emuToBohr(emuM,mass,molweight):
    avo =6.0221409e+23 #part/mol
    bohr = 9.274e-21 #emu / Bohr magneton = erg/G/Bohrmag
    if (isinstance(emuM,list)):
        bohrM = []
        for i in M:
            bohrM.append(i/(mass/molweight*avo*bohr))
    else:
        bohrM = emuM/(mass/molweight*avo*bohr)
    return bohrM

#Takes in a list of magnetic field (in Oe)
#Returns a list of magnetic fields (in Tesla)
def oeToTesla(H):
    newH = H/10000
    return newH

def getMass(filename,**kwargs):
	if kwargs['who'] == 'Arun':
		mass = filename.split('_')[3]
		mass = mass.replace('P','.')
	else:
		mass = filename.split('_')[2]
		mass = mass.replace('P','.')		
	mass = mass[:-2]
	mass = float(mass)
	mass = mass/1000
	return mass
def getTemp(filename,**kwargs):
	if kwargs['who'] == 'Arun':
	    temp = filename.split('_')[-1].split('.')[0][:-1]
	    temp = float(temp)
	else:
	    temp = filename.split('_')[-1].split('.')[0][:-1].replace('P','.')
	    temp = float(temp)		
	return temp


    
#Deprecated
#####################################################################################################################################################################
# # for checking eigenvalues (and hence energies) at a given (x,bpf) coordinate
# def printPCFEigensJ(x,bpf):
# 	Stev={'B40': bpf, 'B60': x*bpf}
# 	Stev['B44'] = 5*Stev['B40']
# 	Stev['B64'] = -21*Stev['B60']
# 	Pr = cef.CFLevels.Bdict(Bdict=Stev, ion = 'Ce3+')
# 	Pr.diagonalize()
# 	Pr.printEigenvectors()
# 	return

# def energyCalcK(x,bpf,LS):
# 	numlevels = 4
# 	# print('For LS = ', LS)
# 	Stev = {}
# 	e = np.zeros((len(x),len(bpf),numlevels))
# 	for i in range(len(x)):
# 		for j in range(len(bpf)):
# 			Stev['B40'] = -bpf[j]
# 			Stev['B60'] = -x[i]*bpf[j]
# 			Stev['B44'] = 5*Stev['B40']
# 			Stev['B64'] = -21*Stev['B60']
# 			Pr = cef.LS_CFLevels.Bdict(Bdict=Stev, L=3, S=0.5, SpinOrbitCoupling=LS)
# 			Pr.diagonalize()
# 			e[i][j] = kmeansSort(Pr.eigenvalues)
# 		# print(i)
# 	return e



# def energyCalcKPar2(LS,x = 0.0352 ,bpf = -0.3970, numlevels = 4):
# 	numlevels = numlevels
# 	Stev = {}

# 	Stev['B40'] = bpf
# 	Stev['B60'] = x*bpf
# 	Stev['B44'] = 5*Stev['B40']
# 	Stev['B64'] = -21*Stev['B60']

# 	Pr = cef.LS_CFLevels.Bdict(Bdict=Stev, L=3, S=0.5, SpinOrbitCoupling=LS)
# 	Pr.diagonalize()
# 	e = kmeansSort(Pr.eigenvalues,numlevels)
# 	return e

# #Function that finds compatibility coordinates within a certain tolerance.
# #Works as follows:
# #Check if E1 is within tolerance, add (x,bpf) coords to list
# #Check the same for E2, etc.
# #Then only keep the coordinates that appear in all of the energy bands
# def paramFinderJ(data,band,E,tolerance,comp):

# 	print('\nParameter search for: ',' Compound: ', comp, ' with %0.3f tolerance.' %tolerance)
# 	coords = []
# 	#The first part that only care about individual energy band and toleranc 
# 	for i in range(len(E)):
# 		for j in range(len(data[band[i]])):
# 			for k in range(len(data[band[i]][j])):
# 				if not (np.isnan(data[band[i]][j][k])):
# 					if (((1-tolerance)*E[i] <= data[band[i]][j][k]) and ((1+tolerance)*E[i] >=  data[band[i]][j][k])):
# 						temp = [data[band[i]][j][k]]
# 						if ([j,k] not in coords):
# 							coords.append([j,k])

# 	newCoords = []
# 	#The second part that only leaves the coordinates that fall within all four bands.
# 	for i in coords:
# 		allbands = []
# 		for j in range(len(E)):
# 			if ((1-tolerance)*E[j] <= data[band[j]][i[0]][i[1]] and ((1+tolerance)*E[j] >= data[band[j]][i[0]][i[1]])):
# 				allbands.append(0)
# 				# print(data[band[j]][i[0]][i[1]])
# 			else:
# 				allbands.append(1)
# 		if 1 not in allbands:
# 			newCoords.append(i)

# 	return newCoords

# # This function loads my .mat files for analyzing, plotting, finding compatibilities.
# #THIS WORKS since it can properly use my previously generated PrO2 data.
# def loadMatrixJ(runDir,grid, **kwargs):
# 	#Where to save and load the energy data for contours
# 	data = sio.loadmat(runDir+grid)
# 	dataList = []
# 	EList = []
# 	E = []
# 	# print(data.keys())
# 	for i in data.keys():
# 		if 'E' in i:
# 			E.append(i)
# 		dataList.append(data)
# 		EList.append(E)
# 	return E, data






# # contour plotting function for all energy bands
# def plotContoursLS(data,EList,E,LSName):
# 	plt.figure()
# 	numplots = len(EList)
# 	if (numplots%2 == 0):
# 		snum = np.sqrt(numplots)
# 	else:
# 		snum = np.sqrt(numplots) + 1
# 	for i in range(1,numplots+1):
# 		ax = plt.subplot(snum,snum,i)
# 		# print(data['X'])
# 		mapp = ax.contourf(data['X'][0],data['B'][0],data[EList[i-1]])
# 		# print(np.shape(data[EList[i-1]]))
# 		ax.set(xlabel = 'Ratio of B60/B40', ylabel = 'B Prefactor', title = EList[i-1])
# 		cbar = plt.colorbar(mapp,ax = ax)
# 		cbar.set_label('Energy (meV)')

# 	plt.tight_layout(h_pad = -1, w_pad = -2)
# 	plt.suptitle(LSName)
# 	# plt.show()
# 	return	

# def saveMatrixPar(xmin,xmax,numx,bpfmin,bpfmax,numbpf,LSList,runDir,numlevels):
# 	x = np.linspace(xmin,xmax,numx)
# 	bpf = np.linspace(bpfmin,bpfmax,numx)
# 	print('Xmin = %0.3f to Xmax = %.3f\nBPFmin = %.3f to BPFmax = %.3f \nwith number of steps in X = %s, Bpf = %s'%(xmin,xmax,bpfmin,bpfmax,numx,numbpf))

# 	for j in range(len(LSList)):
# 		savedict = {'X': x, 'B': bpf, 'LS': LSList[j]}
# 		# if __name__ == '__main__':
# 		with mp.Pool() as P:
# 			E = P.starmap(partial(energyCalcKPar,LS = LSList[j], numlevels = numlevels),product(x,bpf))
# 			print(E[0])
# 			E = np.reshape(E,(numx,-1,numlevels))
# 			# P.close()

# 		if (os.path.exists(runDir) == False):
# 			os.makedirs(runDir)
# 		for i in range(np.shape(E)[2]):
# 			savedict['E%i'%(i+1)] = E[:,:,i]
# 		sio.savemat(runDir+'LS_%i.mat'%LSList[j], savedict)
# 	return


# def energyCalcKParJ(x,bpf, numlevels):
# 	numlevels = numlevels
# 	Stev = {}

# 	Stev['B40'] = bpf
# 	Stev['B60'] = x*bpf
# 	Stev['B44'] = 5*Stev['B40']
# 	Stev['B64'] = -21*Stev['B60']

# 	Pr = cef.CFLevels.Bdict(Bdict=Stev, ion ='Ce3+')
# 	Pr.diagonalize()
# 	e = kmeansSort(Pr.eigenvalues,numlevels)
# 	return e

# def saveMatrix(xmin,xmax,numx,bpfmin,bpfmax,numbpf,LSList,runDir,comp):
# 	x = np.linspace(xmin,xmax,numx)
# 	bpf = np.linspace(bpfmin,bpfmax,numx)
# 	print('Xmin = %0.3f to Xmax = %.3f\nBPFmin = %.3f to BPFmax = %.3f \nwith number of steps in X = %s, Bpf = %s'%(xmin,xmax,bpfmin,bpfmax,numx,numbpf))
# 	for j in  range(len(LSList)):
# 		E = energyCalcK(x,bpf,LSList[j],comp)
# 		savedict = {'X': x, 'B': bpf, 'LS': LSList[j]}
# 		if (os.path.exists(runDir) == False):
# 			os.makedirs(runDir)
# 		for i in range(np.shape(E)[2]):
# 			savedict['E%i'%(i+1)] = E[:,:,i]
# 		sio.savemat(runDir+'LS_%i.mat'%LSList[j], savedict)
# 	return

# def plotContours(data,EList,E,LSName):
# 	fig, axs = plt.subplots(2, 2)
# 	row = 0
# 	col = 0
# 	for i in range(len(EList)):
# 		mapp = axs[row,col].contourf(data['X'],data['B'],data[EList[i]])
# 		axs[row,col].set(xlabel = 'Ratio of B60/B40', ylabel = 'B Prefactor', title = EList[i]+' = %i meV'%E[i])
# 		cbar = plt.colorbar(mapp,ax = axs[row,col])
# 		cbar.set_label('Energy (meV)')

# 		if (row == 0 and col == 0):
# 			row = 0
# 			col = 1
# 		elif (row == 0 and col == 1):
# 			row = 1
# 			col = 0
# 		elif (row == 1 and col == 0):
# 			row = 1
# 			col = 1

# 	fig.tight_layout(h_pad = -.01)
# 	fig.suptitle(LSName)
# 	# plt.show()
# 	return


# # #energy level calculating function, using old eigensorting NOT PARALLELIZED
# def energyCalc(x,bpf,LS, comp):
# 	numlevels = 4 #number of excited levels expected
# 	print('For LS = ', LS)
# 	PCOLig, Pr = cef.importCIF(comp + '.cif','Pr1', LS_Coupling = LS)
# 	e = np.zeros((np.shape(x)[0],np.shape(bpf)[0],numlevels)) #Creating the 3D structure for energies, the last dimension is the number of levels stored at each point. (x,bpf,[E1,E2,E3,E4])
# 	for i in range(len(x)):#iterate through x
# 		for j in range(len(bpf)):#iterate through bpf
# 			#Implementing cubic symmetry relations
# 			B40 = -1*bpf[i][j]
# 			B60 = -x[i][j]*bpf[i][j]
# 			B44 = 5*B40
# 			B64 = -21*B60
# 			boothroydBs = [0,0,0,B40,0,0,0,B44,B60,0,0,0,B64,0]#assigning the new coefficients
# 			Pr.newCoeff(boothroydBs) #diagonalizing
# 			e[i][j] = eigenSort(Pr.eigenvalues)[0:4] #calling my OLD eigensort function and just keeping the first 4 levels.
# 	return e

# # #energy level calculating function using kmeans sorting NOT PARALLELIZED
# def energyCalcK(x,bpf,LS, comp):
# 	numlevels = 4
# 	print('For LS = ', LS)
# 	PCOLig, Pr = cef.importCIF(comp + '.cif','Pr1', LS_Coupling = LS)
# 	e = np.zeros((np.shape(x)[0],np.shape(bpf)[0],numlevels))
# 	for i in range(np.shape(x)[1]):
# 		for j in range(len(bpf)):
# 			B40 = 1*bpf[i][j]
# 			B60 = -x[i][j]*bpf[i][j]
# 			B44 = 5*B40
# 			B64 = -21*B60
# 			boothroydBs = [0,0,0,B40,0,0,0,B44,B60,0,0,0,B64,0]
# 			Pr.newCoeff(boothroydBs)
# 			e[i][j] = kmeansSort(Pr.eigenvalues)
# 		print(i)
# 	return e

# This is the function used to calculate the energy at each (x,bpf) and save it as a .mat file with the filename representing the LS value
# Since this is where energy calculation happens, the bug is somewhere here.
# def saveMatrix(x,bpf,LSList,runDir,comp):
# 	X,Bpf = np.meshgrid(x,bpf)
# 	for j in  range(len(LSList)):
# 		E = energyCalcK(X,Bpf,LSList[j],comp)
# 		savedict = {'X': X, 'B': Bpf}
# 		if (os.path.exists(runDir) == False):
# 			os.makedirs(runDir)
# 		for i in range(np.shape(E)[2]):
# 			savedict['E%i'%(i+1)] = E[:,:,i]
# 		sio.savemat(runDir+'LS_%i.mat'%LSList[j], savedict)
# 	return

#Deprecated fxn which includes 0 modes. Just for checking I handled 3D data structure correctly.
# def eigenSort0(e):
#     eigens = []
#     for i in (np.unique(e.round(decimals = 6))):
#         if i != 0:
#             eigens.append(i)
#     return eigens

#Original Eigensort function which (somewhat sloppily) chooses the 4 energy levels to track out of the 14 produced by PCF
# def eigenSort(e):
# 	e1 = np.unique(e.round(decimals = 6))
# 	eigens = []
# 	for i in e1:
# 		if i != 0:
# 			eigens.append(i)
# 	return eigens

#####################################################################################################################################################################

