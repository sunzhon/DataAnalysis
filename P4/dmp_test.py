import numpy as np  
from matplotlib import pyplot as plt  
import pdb

file_path="/home/suntao/workspace/gorobots/examples/dmp/dmp_standard/dmp.dat"
data=np.loadtxt(file_path)  
#pdb.set_trace()

plt.plot(data[:,0],data[:,1],'bo')  

X=data[:,0]  
Y=data[:,1]  

#plt.plot(X,Y,':ro')  
plt.show()  
