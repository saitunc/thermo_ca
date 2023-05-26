import pandas as pd
import numpy as np 



def write_to_class(B,D,cls):
    contemp = pd.DataFrame({'rule_number':[f"B : {B} , D : {D}"],
                            'Class':[f'Class {cls}']})
    contemp.to_csv('classes.csv',mode='a',index=False,header=False)


def CVA(fun1,fun2):
    return np.std(fun1[0:40]/fun2[0:40])/np.mean(fun1[0:40]/fun2[0:40])

def CVT(fun1,fun2):
    return np.std(fun1[0:25]/fun2[0:25])/np.mean(fun1[0:25]/fun2[0:25])
