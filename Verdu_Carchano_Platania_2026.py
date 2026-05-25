# ======================================================================================================================================================

# Code for the article Verdú, Carchano, Platania (2026). On abnormal returns in events associated with equity offerings.
# Estudios de Economía. 53:1. 5-26. (Pending of Publication).

# ======================================================================================================================================================

# To execute this code, you must have the following files available in https://doi.org/10.5281/zenodo.20378127.
#   Verdu_Carchano_Platania_2026_Data.csv

# Once the data is available in the same directory, you only need to execute the code to obtain the results shown in the article.

# The article can be found at: http://dx.doi.org/10.4067/S0718-52862026000100005 (Pending of Publication)

# ======================================================================================================================================================

#################### LIBRARIES TO USE ####################

import os
import sys
import warnings
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from contextlib import redirect_stdout

from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.discrete.discrete_model import Probit
from scipy.stats import norm, chi2, f as f_dist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

#################### START OF AUXILIAR PROGRAMS ####################

def sample_des(DATA, DATAC):

    L1 = len(DATA)
    F1 = len(DATAC)

    return L1, F1

def run_all_tests(DATA, Series, label):
    """
    Run all tests and return results as a dictionary.
    """
    DAT = DATA[DATA[Series].notna() & (DATA[Series] != 0)]
    AR = DAT[Series].values
    N = len(AR)
    
    if N < 2:
        return None
    
    results = {'label': label, 'series': Series, 'N': N, 'AVE': np.mean(AR)}
    
    # t-test
    AVE = np.mean(AR)
    STD = np.std(AR, ddof=1)
    tst = (AVE / STD) * np.sqrt(N)
    results['t_pval'] = 2 * stats.t.sf(abs(tst), N-1)
    
    # Patell
    SAR = AR / STD
    Z_patell = np.sum(SAR) / np.sqrt(N)
    results['patell_pval'] = 2 * (1 - stats.norm.cdf(abs(Z_patell)))
    
    # BMP
    SAR_mean = np.mean(SAR)
    S_SAR = np.std(SAR, ddof=1)
    Z_bmp = np.sqrt(N) * SAR_mean / S_SAR
    results['bmp_pval'] = 2 * (1 - stats.norm.cdf(abs(Z_bmp)))
    
    # Generalized Sign
    w = np.sum(AR > 0)
    Z_gs = (w - N * 0.5) / np.sqrt(N * 0.25)
    results['gsign_pval'] = 2 * (1 - stats.norm.cdf(abs(Z_gs)))
    results['pct_pos'] = w / N
    
    return results

def model(DATA, Series):
    """
    """
    if Series not in DATA.columns:
        print(f"Column {Series} not found in DATA.")
        return

    DAT = DATA[DATA[Series].notna() & (DATA[Series] != 0)]
    D = len(DAT)

    if D < 12:
        print(f"Not enough data for {Series}: {D} rows after filtering.")
        return

    if DAT[Series].nunique() == 1:
        print(f"Dependent variable {Series} is constant after filtering.")
        return

    Models = ['3', '4']
    
    # Formulas
    formulas = [Series + ' ~ DIL + RAT + INS + RGT + IDX + BTM + ILIQ + PER2 + PER3',
        Series + ' ~ DIL + RAT + INS + RGT + IDX + BTM + ILIQ + PER2 + PER3 + EXR'
    ]

    for i in range(len(formulas)):
        print(f'\n################ MODEL [{Models[i]}] ####################')
        
        try:
            
            # Run model
            formula = formulas[i]
            res = smf.ols(formula, DAT).fit(cov_type='HC1')
            print(res.summary())
            
        except Exception as e:
            print(f"Error fitting model {Models[i]} for {Series}: {e}")

#################### END OF AUXILIAR PROGRAMS ####################

#################### START OF THE CODE ####################

os.chdir('/Users/manuelverduhenares/Library/CloudStorage/Dropbox/1_Investigación/10_BBDD/2026_Verdu_Carchano_Platania_EdE')

DATA = pd.read_csv('Verdu_Carchano_Platania_2026_Data.csv', sep = ';')
DATA = DATA.drop('Unnamed: 0', axis=1)
DATAC = DATA.drop_duplicates(subset=['RIC'])

L = len(DATA)
F = len(DATAC)

DATAE = DATA[DATA['ESG'].notna() & (DATA['ESG'] != '')]
DATAEC = DATAE.drop_duplicates(subset=['RIC'])

LE = len(DATAE)
FE = len(DATAEC)

Groups = ['1D', '3D', '5D']
Codes = ['1', '3', '5']

Countries = ['Canada', 'United States', 'Australia', 'China (Mainland)', 'Hong Kong', 'India', 'Indonesia', 'Japan', 'Malaysia'
        , 'New Zealand', 'Philippines', 'Singapore', 'South Korea', 'Taiwan', 'Thailand', 'Turkey', 'France', 'Germany', 'Italy', 'Norway', 'Poland'
        , 'Spain', 'Sweden', 'Switzerland', 'United Kingdom']

Sectors = ['Academic & Educational Services', 'Basic Materials', 'Consumer Cyclicals', 'Consumer Non-Cyclicals', 'Energy', 'Financials', 'Government Activity'
, 'Healthcare', 'Industrials', 'Real Estate', 'Technology', 'Utilities']

ESG = ['A', 'B', 'C', 'D']

Sample = '2'
Models = ['4', '2']

##### STATISTICAL DESCRIPTION OF THE ABNORMAL RETURNS #####

all_results = []

print(f'========================= Sample Size: Equity Offerings: {L} / Firms: {F} =========================')
print()

for g in range(0, len(Groups)):
    for Model in Models:

        code = 'AR' + Codes[g] + '_' + Sample + Model
        label = f'Total_{Groups[g]}_{Sample}{Model}'
        res = run_all_tests(DATA, code, label)
        if res:
            all_results.append(res)

for country in Countries:
    DATAR = DATA[DATA['COU'] == country]
    DATARC = DATAC[DATAC['COU'] == country]

    L1, F1 = sample_des(DATAR, DATARC)
    print(f'Sample Size for {country}: Equity Offerings: {L1} ({(L1/L)*100:.2f}%)/ Firms: {F1} ({(F1/F)*100:.2f}%)')

    for g in range(0, len(Groups)):
        for Model in Models:

            code = 'AR' + Codes[g] + '_' + Sample + Model
            label = f'{country}_{Groups[g]}_{Sample}{Model}'
            res = run_all_tests(DATAR, code, label)
            if res:
                all_results.append(res)

print()
for sector in Sectors:
    DATAR = DATA[DATA['ECO'] == sector]
    DATARC = DATAC[DATAC['ECO'] == sector]
    
    L1, F1 = sample_des(DATAR, DATARC)
    print(f'Sample Size for {sector}: Equity Offerings: {L1} ({(L1/L)*100:.2f}%)/ Firms: {F1} ({(F1/F)*100:.2f}%)')

    for g in range(0, len(Groups)):
        for Model in Models:

            code = 'AR' + Codes[g] + '_' + Sample + Model
            label = f'{sector}_{Groups[g]}_{Sample}{Model}'
            res = run_all_tests(DATAR, code, label)
            if res:
                all_results.append(res)

print()
for esg in ESG:

    DATAR = DATAE[DATAE['GRA'].str[0] == esg]
    DATARC = DATAEC[DATAEC['GRA'].str[0] == esg]
    
    L1, F1 = sample_des(DATAR, DATARC)
    print(f'Sample Size for {esg}: Equity Offerings: {L1} ({(L1/LE)*100:.2f}%)/ Firms: {F1} ({(F1/FE)*100:.2f}%)')

    for g in range(0, len(Groups)):
        for Model in Models:

            code = 'AR' + Codes[g] + '_' + Sample + Model
            label = f'ESG{esg}_{Groups[g]}_{Sample}{Model}'
            res = run_all_tests(DATAR, code, label)
            if res:
                all_results.append(res)

print()
DATAR = DATA[DATA['DIL'] == 1]
DATARC = DATAC[DATAC['DIL'] == 1]
L1, F1 = sample_des(DATAR, DATARC)
print(f'Sample Size for Dilutives: Equity Offerings: {L1} ({(L1/L)*100:.2f}%)/ Firms: {F1} ({(F1/F)*100:.2f}%)')

DATAR = DATA[DATA['DIL'] == 0]
DATARC = DATAC[DATAC['DIL'] == 0]
L1, F1 = sample_des(DATAR, DATARC)
print(f'Sample Size for Non-Dilutives: Equity Offerings: {L1} ({(L1/L)*100:.2f}%)/ Firms: {F1} ({(F1/F)*100:.2f}%)')

print()
DATAR = DATA[DATA['INS'] == 1]
DATARC = DATAC[DATAC['INS'] == 1]
L1, F1 = sample_des(DATAR, DATARC)
print(f'Sample Size for Insured: Equity Offerings: {L1} ({(L1/L)*100:.2f}%)/ Firms: {F1} ({(F1/F)*100:.2f}%)')

DATAR = DATA[DATA['INS'] == 0]
DATARC = DATAC[DATAC['INS'] == 0]
L1, F1 = sample_des(DATAR, DATARC)
print(f'Sample Size for Not Insured: Equity Offerings: {L1} ({(L1/L)*100:.2f}%)/ Firms: {F1} ({(F1/F)*100:.2f}%)')

print()
DATAR = DATA[DATA['IDX'] == 1]
DATARC = DATAC[DATAC['IDX'] == 1]
L1, F1 = sample_des(DATAR, DATARC)
print(f'Sample Size for Indexed: Equity Offerings: {L1} ({(L1/L)*100:.2f}%)/ Firms: {F1} ({(F1/F)*100:.2f}%)')

DATAR = DATA[DATA['IDX'] == 0]
DATARC = DATAC[DATAC['IDX'] == 0]
L1, F1 = sample_des(DATAR, DATARC)
print(f'Sample Size for Not Indexed: Equity Offerings: {L1} ({(L1/L)*100:.2f}%)/ Firms: {F1} ({(F1/F)*100:.2f}%)')

print()
DATAR = DATA[DATA['RGT'] == 1]
DATARC = DATAC[DATAC['RGT'] == 1]
L1, F1 = sample_des(DATAR, DATARC)
print(f'Sample Size for Rights: Equity Offerings: {L1} ({(L1/L)*100:.2f}%)/ Firms: {F1} ({(F1/F)*100:.2f}%)')

DATAR = DATA[DATA['RGT'] == 0]
DATARC = DATAC[DATAC['RGT'] == 0]
L1, F1 = sample_des(DATAR, DATARC)
print(f'Sample Size for Not Rights: Equity Offerings: {L1} ({(L1/L)*100:.2f}%)/ Firms: {F1} ({(F1/F)*100:.2f}%)')

print()
DATAR = DATA[DATA['PER1'] == 1]
DATARC = DATAC[DATAC['PER1'] == 1]
L1, F1 = sample_des(DATAR, DATARC)
print(f'Sample Size for Period 1: Equity Offerings: {L1} ({(L1/L)*100:.2f}%)/ Firms: {F1} ({(F1/F)*100:.2f}%)')

DATAR = DATA[DATA['PER2'] == 1]
DATARC = DATAC[DATAC['PER2'] == 1]

L1, F1 = sample_des(DATAR, DATARC)
print(f'Sample Size for Period 2: Equity Offerings: {L1} ({(L1/L)*100:.2f}%)/ Firms: {F1} ({(F1/F)*100:.2f}%)')

DATAR = DATA[DATA['PER3'] == 1]
DATARC = DATAC[DATAC['PER3'] == 1]
L1, F1 = sample_des(DATAR, DATARC)
print(f'Sample Size for Period 3: Equity Offerings: {L1} ({(L1/L)*100:.2f}%)/ Firms: {F1} ({(F1/F)*100:.2f}%)')

# Convert to DataFrame
df_results = pd.DataFrame(all_results)

# Apply corrections for each test type
for test in ['t_pval', 'patell_pval', 'bmp_pval', 'gsign_pval']:
    
    # Benjamini-Hochberg correction (controls FDR)
    _, pvals_bh, _, _ = multipletests(df_results[test], method='fdr_bh')
    df_results[test.replace('pval', 'bh')] = pvals_bh

RES = pd.DataFrame(df_results)

print()
print('========================= Statistical Results =========================')
print()

for d in range(0, len(RES)):     
    print(f"{RES['label'][d]}: {RES['AVE'][d]:.4f}% (t: {RES['t_bh'][d]:.4f} / Patell: {RES['patell_bh'][d]:.4f} / GSign: {RES['gsign_bh'][d]:.4f})")

##### MODELIZATION OF THE ABNORMAL RETURNS #####

Model = '4'

for g in range(0,len(Groups)):

    #print(f"Processing {Groups[g]} {Sample}{Model}", file=sys.stderr)

    print()
    print(f'{"#" * 60}')
    print(f'                         {Groups[g]} {Sample}{Model}')
    print(f'{"#" * 60}')

    code = 'AR' + Codes[g] + '_' + Sample + Model

    model(DATA, code)

#################### END OF THE CODE ####################