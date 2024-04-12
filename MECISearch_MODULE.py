import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
import sys,os
import time

import MECI as meci
import CONSTANTS as CST

def getStateDerivatives(filename=None,mw=True,freq=False):
    NAtoms,NCoords,atomicNumbers,currentCoordinates=meci.fchk2coordinates(filename) # coordinates in Angstroms
    if freq:
        currentEnergy,currentGradient,currentHessian=meci.fchk2derivatives(filename,mw=mw,freq=freq)[:3]
    else:
        currentEnergy,currentGradient=meci.fchk2derivatives(filename,mw=mw,freq=freq)[:2]
    currentGradient=currentGradient/CST.BOHR_TO_ANGSTROM
    if freq:
        currentHessian=currentHessian/(CST.BOHR_TO_ANGSTROM**2)
    if freq:
        return currentEnergy,currentGradient,currentHessian
    else:
        return currentEnergy,currentGradient

def BFGSUpdate(previousDisplacement,previousGradient,previousHessian,currentGradient):
    deltaGradient=currentGradient-previousGradient
    currentHessian=(
            previousHessian
            +
            np.tensordot(deltaGradient,deltaGradient,axes=0)
            /
            np.dot(deltaGradient,previousDisplacement)
            -
            (np.tensordot(np.dot(previousHessian,previousDisplacement),np.dot(previousHessian,previousDisplacement),axes=0))
            /
            (np.dot(previousDisplacement,np.dot(previousHessian,previousDisplacement)))
            )
    return currentHessian

def writeComFileFromXYZ(comfile,header,xyzfile):
    geometry=np.loadtxt(xyzfile,skiprows=2,dtype=str)
    atomicNumbers=geometry[:,0].astype(int)
    coordinates=geometry[:,1:].astype(float)
    with open(comfile,"w") as f:
        f.write(header)
        f.write("\n")
        for atom in geometry:
            f.write("\t".join(atom))
            f.write("\n")
        f.write("\n")
        f.write("\n")
    return

def writeComFile(comfile,header,nextCoordinates):
    geometry=np.copy(nextCoordinates).astype(str)
    with open(comfile,"w") as f:
        f.write(header)
        f.write("\n")
        for atom in geometry:
            f.write("\t".join(atom))
            f.write("\n")
        f.write("\n")
        f.write("\n")
    return

def writeComFiles(oldcomfilenames,nextCoordinates,c=None):
    filetag="_".join(oldcomfilenames[0].split("_")[:-1])
    roottags=["A","B"]
    rootnumbers=[1,2]
    comfilenames=[]
    for iroot,roottag in enumerate(roottags):
        if type(c) is int:
            comfilename="".join(filetag.split("_")[:-1])+"_{}_{}.com".format(c,roottag)
        else:
            comfilename=filetag+"_GD_{}.com".format(roottag)
        comfilenames.append(comfilename)
        # print("Writing file "+comfilename)
        with open(oldcomfilenames[iroot],"r") as f:
            lines=f.readlines()[:8]
            multiplicityLine=lines[-1]
            if len(lines[4].split())>0:
                lines[-2]+=lines[-1]
                lines[-1]=''
                # print("Error found in the .com file (wrong number of lines), concatenating last two lines")
            lines=lines[:5]
            calculation=np.array(lines[3].split())
            lines[3]=" ".join(calculation)+"\n"
        with open(comfilename,"w") as f:
            for i,line in enumerate(lines):
                if i==0:
                    f.write("%chk="+comfilename.split(".")[0]+".chk\n")
                else:
                    if len(line.split())>0:
                        f.write(line)
            f.write("\n")
            f.write("step\n")
            f.write("\n")
            f.write(multiplicityLine)
            for iatom,atomCoordinates in enumerate(nextCoordinates):
                f.write("\t".join(atomCoordinates)+"\n")
            f.write("\n")
    return oldcomfilenames,comfilenames

def writeTriangularMatrix(matrix_to_print,f,pandas_printing=False):
    if not pandas_printing:
        c=0
        flattened_matrix=np.array([])
        for line in matrix_to_print.astype(str):
            flattened_matrix=np.append(flattened_matrix,line[c:])
            c+=1
        f.write("NElements {}\n".format(flattened_matrix.shape))
        for i in range(len(flattened_matrix)//5):
            f.write(" ".join(flattened_matrix[i*5:(i+1)*5])+"\n")
        if len(flattened_matrix)%5!=0:
            f.write(" ".join(flattened_matrix[5*len(flattened_matrix)//5-1:])+"\n")
    if pandas_printing:
        c=0
        flattened_matrix=np.array([])
        for line in matrix_to_print.astype(str):
            flattened_matrix=np.append(flattened_matrix,line[c:])
            c+=1
        f.write("NElements {}\n".format(flattened_matrix.shape))
        for i in range(len(flattened_matrix)//5):
            to_print=pd.DataFrame([[_] for _ in flattened_matrix[i*5:(i+1)*5]])
            f.write(to_print.to_string(index=False,header=False)+"\n")
        to_print=pd.DataFrame([[_] for _ in flattened_matrix[5*len(flattened_matrix)//5-1:]])
        if len(flattened_matrix)%5!=0:
            f.write(to_print.to_string(index=False,header=False)+"\n")
    return 

