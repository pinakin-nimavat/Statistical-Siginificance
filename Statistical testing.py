import pandas as pd
import numpy as np
from scipy import stats
import pprint

def chi2_score(df,col1,col2):

    df_cont = pd.crosstab(index=df[col1], columns=df[col2])     #contigency table
    degree_f = (df_cont.shape[0] - 1) * (df_cont.shape[1] - 1)  #calculating degree of freedom


    df_cont.loc[:, 'Total'] = df_cont.sum(axis=1)       #observed values
    df_cont.loc['Total'] = df_cont.sum()
    # print("O------------------",df_cont)

    df_exp = df_cont.copy()         #calculating expected values
    df_exp.iloc[:, :] = np.multiply.outer(
    df_cont.sum(1).values, df_cont.sum().values) / df_cont.sum().sum()
    # print("E------------------",df_exp)

    df_chi2 = ((df_cont - df_exp) ** 2) / df_exp        #calculating chi square
    df_chi2.loc[:, 'Total'] = df_chi2.sum(axis=1)
    df_chi2.loc['Total'] = df_chi2.sum()

    chi_square_score = df_chi2.iloc[:-1, :-1].sum().sum()

    p = stats.distributions.chi2.sf(chi_square_score, degree_f)

    return chi_square_score, degree_f, p


def welch_ttest(df1, df2, col):
    x, y = df1[col], df2[col]

    dof = (x.var() / x.size + y.var() / y.size) ** 2 / (
                (x.var() / x.size) ** 2 / (x.size - 1) + (y.var() / y.size) ** 2 / (y.size - 1))

    t, p = stats.ttest_ind(x, y, equal_var=False)

    return t,p,dof

def perform_tests(df):
    chi_li = []
    shapiro_li = []
    ttest_li = []
    mann_li = []
    k_out = ['mann_whitney','ttest','chi_square','shapiro_wilk']
    v_out = [mann_li,ttest_li,shapiro_li,chi_li]
    output = {}

    categorical_feature = df.select_dtypes('int64').columns.to_list()
    numeric_feature = df.select_dtypes('float64').columns.to_list()
    death = categorical_feature.pop(0)

    for c in range(len(categorical_feature)):
        chi_score, degree_f, p = chi2_score(df, 'death', categorical_feature[c])
        chi_li.append((categorical_feature[c],float(f'{p:.4f}')))

    # preparing data for shapiro wilk test
    death_0 = df[df["death"] == 0]
    death_0 = death_0[numeric_feature]

    death_1 = df[df["death"] == 1]
    death_1 = death_1[numeric_feature]

    ttest_feature = []
    mann_feature = []

    for val in range(len(numeric_feature)):
        t_0,p_0 = stats.shapiro(death_0[numeric_feature[val]])
        t_1,p_1 = stats.shapiro(death_1[numeric_feature[val]])
        if p_0 and p_1 >= 0.05:
            ttest_feature.append(numeric_feature[val])
        else:
            mann_feature.append(numeric_feature[val])
        shapiro_li.append((numeric_feature[val],(float(f'{p_0:.4f}'),float(f'{p_1:.4f}'))))

    for val in ttest_feature:
        t,p,dof = welch_ttest(death_0, death_1, val)
        ttest_li.append((val,float(f'{p:.4f}')))

    for val in mann_feature:
        t,p = stats.mannwhitneyu(death_0[val], death_1[val])
        mann_li.append((val,float(f'{p:.4f}')))

    for key,value in zip(k_out,v_out):
        output[key] = value

    return(output)


df = pd.read_csv("medical_data.csv")
pprint.pprint(perform_tests(df))



