# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:19:35 2021

@author: HB

todo: make outputstring stating the nucleus-nucleus interactions
make it auto detect delimiters
"""

import numpy as np
from scipy import constants as cnst
import sys
import os


class DataSet():

    def __init__(self, data_path, nuc1, nuc2):
        
        # Data import
        self.data = np.genfromtxt(data_path, 
                     dtype=str, delimiter=' ', encoding=None)
        self.data = np.split(self.data, self.find_distinct_indices(), axis=0)
        
        # Isotope information
        isoPath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "IsotopeProperties"
        self.isotopes = self.getIsotopeInfo(isoPath)
        self.ind1 = np.where(self.split_string(nuc1) == np.array(self.isotopes['atomMass']))[0][0]
        self.ind2 = np.where(self.split_string(nuc2) == np.array(self.isotopes['atomMass']))[0][0]
        self.gamma1 = self.isotopes['gamma'][self.ind1] * 1e7
        self.gamma2 = self.isotopes['gamma'][self.ind2] * 1e7
        self.quant_number = self.isotopes['spin'][self.ind2]
        self.abundance = self.isotopes['abundance'][self.ind1]/100
        # self.index2 = np.where(self.nuc2[0] == np.array(self.isotopes['atomMass']))[0][0]
        # self.quant_number = self.isotopes['spin'][self.index[2]]
    
    def split_string(self, s):
        try:
            mass = float(s.rstrip('AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz'))
        except:
            raise ValueError('Nuclei have to be input in the format "mass number" + "atomic symbol", e.g., 11B or 125Te.')
        return mass
        
    def find_distinct_indices(self):
        # Splits data into arrays according to individual crystal sites
        # !This works only for two different sites for now!
        strings = np.unique(self.data[:,0])
        indices = [np.where(self.data[:,0] == strings[x])[0][0] 
                   for x in range(1, len(strings))]
        return indices

    def m2het(self, dist_sum):
        vac_perm = cnst.physical_constants['vacuum mag. permeability'][0]
        return (self.abundance * (4/15) * (vac_perm / (4 * cnst.pi))**2 * 
                self.quant_number * (self.quant_number + 1)*
                self.gamma1**2 * self.gamma2**2 * cnst.hbar**2 *dist_sum)
    
    def second_moment(self):
        result = [self.m2het(self.dist_sum(x))/1e6 for x in range(len(self.data))]
        return result

    def dist_sum(self, x):
        reci_sp_dist = ((np.array(self.data[x][:,3].astype(np.float)) 
                          * 1e-10)**6 )**-1
        multiplicity = np.array(self.data[x][:,2].astype(np.float))
        return np.sum(reci_sp_dist*multiplicity)
    # def second_moment(self, data, isotope, quant_number):
    #     N, vacuum_permeability = 1, 4*sp.pi*1e-7
        
    #     # Make lookup of gammas given the input string - perhaps take it from strings
    #     # mtwo = np.sum( (N*index*(4/15)*(u0/(4*pi))^2*S2*(S2+1)*gamma1^2*gamma2^2*h^2*1/(dist*1e-10)^6) * index )
    #     test1 = [((4/15)*
    #               (vacuum_permeability/(4*sp.pi))**2*
    #               quant_number*
    #               (quant_number+1)*
    #               gamma1**2*
    #               gamma2**2*
    #               cnst.h**2
    #               *1/(r*1e-10)**6) for r in data[0][:,3]]
    #     test2 = N* test1
    
        # return 
    
    def fOrNone(self, inp):
        """Converts a string to a float and dashes to None"""
        if inp == '-':
            return None
        return float(inp)

    def getIsotopeInfo(self, isoPath):
        """
        Loads the isotope table from a given path.
    
        Parameters
        ----------
        isoPath : str
            The path to the file with the isotope properties.
    
        Returns
        -------
        dict
            A dictionary with the isotope properties.
            Unknown or undefined values are set to None.
        """
        if sys.version_info < (3,):
            with open(isoPath) as isoFile:
                isoList = [line.strip().split('\t') for line in isoFile]
        else:
            with open(isoPath, encoding='UTF-8') as isoFile:
                isoList = [line.strip().split('\t') for line in isoFile]
        isoList = isoList[1:] #Cut off header
        nameList = []
        fullNameList = []
        formatNameList = []
        atomNumList = []
        atomMassList = []
        spinList = []
        abundanceList = []
        gammaList = []
        qList = []
        freqRatioList = []
        refSampleList = []
        sampleConditionList = []
        linewidthFactorList = []
        lifetimeList = []
        sensList = []
        for i, _ in enumerate(isoList):
            isoN = isoList[i]
            atomNumList.append(int(isoN[0]))
            nameList.append(isoN[1])
            fullNameList.append(isoN[2])
            atomMassList.append(self.fOrNone(isoN[3]))
            formatNameList.append(nameList[-1])
            if atomMassList[-1] is not None:
                formatNameList[-1] = '%d' % (atomMassList[i]) + formatNameList[-1]
            spinList.append(self.fOrNone(isoN[4]))
            abundanceList.append(self.fOrNone(isoN[5]))
            gammaList.append(self.fOrNone(isoN[6]))
            qList.append(self.fOrNone(isoN[7]))
            freqRatioList.append(self.fOrNone(isoN[8]))
            refSampleList.append(isoN[9])
            sampleConditionList.append(isoN[10])
            if isoN[4] == '0.5' or spinList[i] is None or qList[i] is None:
                linewidthFactorList.append(None)
            else:
                linewidthFactorList.append((2 * spinList[i] + 3) * qList[i]**2 / (spinList[i]**2 * (2 * spinList[i] - 1)))  # Linewidth due to quadrupolar broadening: (2I + 3) * Q /(I^2 * (2I - 1))
            if gammaList[-1] is not None and abundanceList[-1] is not None and spinList[-1] is not None:
                sensList.append(abundanceList[-1] * abs(gammaList[-1])**3 * spinList[-1] * (spinList[-1] + 1))              # Sensitivity: chi * gamma**3 * I * (I + 1)
            else:
                sensList.append(None)
            lifetimeList.append(isoN[11])
        isotopes = {'atomNum':atomNumList, 'name':nameList, 'fullName':fullNameList, 'atomMass':atomMassList,
                    'formatName':formatNameList, 'spin':spinList, 'abundance':abundanceList, 'q':qList, 'freqRatio':freqRatioList,
                    'refSample':refSampleList, 'sampleCondition':sampleConditionList, 'linewidthFactor':linewidthFactorList,
                    'sensitivity':sensList, 'lifetime':lifetimeList, 'gamma':gammaList}
        return isotopes
    
##########################################################



