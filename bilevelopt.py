import random
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn import svm
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt


arr_random = np.random.randint(low=-10, high=10, size=(1000000,21))
for i in range(0,1000000):
  if arr_random[i][0] + arr_random[i][1] + arr_random[i][2]+ arr_random[i][3]+ arr_random[i][4]+ arr_random[i][5] + arr_random[i][6]+ arr_random[i][7]+ arr_random[i][8]+ arr_random[i][9] + arr_random[i][10] + arr_random[i][11] + arr_random[i][12]+ arr_random[i][13]+ arr_random[i][14]+ arr_random[i][15] + arr_random[i][16]+ arr_random[i][17]+ arr_random[i][18]+ arr_random[i][19]<= 0:
    arr_random[i][20] = -1
  else:
    arr_random[i][20] = 1
  #print(arr_random[i],i)


#df_g15 = pd.DataFrame(arr_random, columns=["A","B"])
#df_g15
#print(type(arr_random))
listrandom = arr_random.tolist()
print(listrandom)
column_names=["A","B", "C","D","E","F","G","H","I","J","A2","B2", "C2","D2","E2","F2","G2","H2","I2","J2","Class"]
df = pd.DataFrame(listrandom,index=None, columns=column_names)
print(df)


train, test = train_test_split(df, test_size=0.2)

w = [0.1, 0.2, 0.3, -0.1, -0.2, -0.3, 0.1, 0.2, 0.3, -0.1]
Cmin = 0
Cmax = 1000
alphatc = 0.1
alphatw = 0.1
n=10
C_bilevel = 0.0001
t = 0
L = len(train)
N = len(test)

Te = []
Ve = []
C_bilevel_list = []

for i in range(0, L-1):
  wxy = np.dot(w,train[["A","B", "C","D","E","F","G","H","I","J"]].iloc[i]) * train['Class'].iloc[i]
  if i%10000 == 0:
    print(" i ",i, wxy)
  if wxy < 1:
    Te.append(i)

for j in range(0, N-1):
    wxy = np.dot(w,test[["A","B", "C","D","E","F","G","H","I","J"]].iloc[j]) * test['Class'].iloc[j]
    if wxy < 1:
      Ve.append(j)

Ne = len(Ve)
Le = len(Te)
print(Le, Ne)

while t <= 10000:
  l = random.randint(0, Le-1)
  p = random.randint(0, Ne-1)
  l = Te[l]
  p = Ve[p]
  #alphatc = 1/(t*math.sqrt(n))
  #alphatw = 1/()
  Grad_w = w - C_bilevel*train['Class'].iloc[l]*train[["A","B", "C","D","E","F","G","H","I","J"]].iloc[l]

  Grad_c = np.dot( -test['Class'].iloc[p] * test[["A","B", "C","D","E","F","G","H","I","J"]].iloc[p],
                  train['Class'].iloc[l]*train[["A","B", "C","D","E","F","G","H","I","J"]].iloc[l] )
  if Grad_c == 0:
    Grad_c = 0.1

  alphatc = 1/(t*math.sqrt(n)*abs(Grad_c))
  #alphatw = 1/()
  w = w - alphatc * Grad_w
  C_bilevel = C_bilevel - (alphatc * Grad_c)
  if C_bilevel < Cmin:
    C_bilevel = Cmin
  if C_bilevel > Cmax:
    C_bilevel = Cmax
  C_bilevel_list.append(C_bilevel)
  print(C_bilevel)
  t = t +1

print(w)
print(C_bilevel)
print(C_bilevel_list)
plt.title('Valor del hyperpametro C para el dataset Synthetic gR2016')
plt.ylabel('C =  '+str(C_bilevel))
plt.plot(C_bilevel_list)
