# In[Preamble]:
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import ESN
from matplotlib import cm
import matplotlib.ticker as ticker
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.regularizers import l1,l2
from tensorflow.compat.v1.keras.models import Sequential,Model
from tensorflow.compat.v1.keras.layers import Reshape,InputLayer,Dense,Flatten, Conv2D,Conv1D, Dropout, Input,ZeroPadding2D,ZeroPadding1D,MaxPooling2D
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
K.set_floatx('float64')  

def lorenz(X, t, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

def sig_scaled(a,b,c,d):
    def sig_tmp(x):
        return a / (1 + K.exp(-b*(x-c)))+d
    return sig_tmp

# In[56]:


# Lorenz paramters and initial conditions
sigma, beta, rho = 10, 2.667, 28
u0, v0, w0 = 0, 1, 1.05

# Maximum time point and total number of time points
tmax, n = 11000, 1100001

plotlength = 8000
burnin = 100001
t = np.linspace(0, tmax, n)
ls = np.linspace(0,1,plotlength)
cmap = cm.viridis(ls)

# Integrate the Lorenz System on the time grid t
f = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))
LS_x, LS_y, LS_z = f.T

# Add noise
#noise_scale = 0
#LS_x = LS_x + np.random.normal(0,noise_scale,size = n)
#LS_z = LS_z + np.random.normal(0,noise_scale,size = n)
"""
# Plot the Lorenz attractor using a Matplotlib 3D projection
fig = plt.figure(figsize = (6,4))
fig.suptitle("3D-scatter of a Lorenzcurve")
ax = fig.gca(projection='3d')
ax.scatter(LS_x[burnin:plotlength+burnin], LS_y[burnin:plotlength+burnin], LS_z[burnin:plotlength+burnin], color = cmap, s = 5)
plt.show()

fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Lorenzcurve per Axis")
ax1.scatter(t[burnin:plotlength+burnin],LS_x[burnin:plotlength+burnin], color = cmap, s = 5)
ax2.scatter(t[burnin:plotlength+burnin],LS_y[burnin:plotlength+burnin], color = cmap, s = 5)
ax3.scatter(t[burnin:plotlength+burnin],LS_z[burnin:plotlength+burnin], color = cmap, s = 5)
plt.show()

"""
# In[57]:


d = 40
#Create Takens A
A= np.eye(d,k=-1)

#Create Takens C
C = np.zeros(d)
C[0] = 1

#create trajectory
x = np.zeros((d,n))  
error = np.zeros((d,n))
for k in range(1,n):
    x[:,k] = A@x[:,k-1] + C*LS_x[k]
    error[:,k] = A@error[:,k-1] + C*np.random.normal(0,1)
    
"""
#Project x onto the first 3 princripal components
U, S, V = np.linalg.svd(x[:,burnin:plotlength+burnin])
projx = U[:,0:1].T@x[:,burnin:plotlength+burnin]
projy = U[:,1:2].T@x[:,burnin:plotlength+burnin]
projz = U[:,2:3].T@x[:,burnin:plotlength+burnin]

# Plot the Takens embedded attractor attractor using a Matplotlib 3D projection
fig = plt.figure(figsize = (6,4))
ax = fig.gca(projection='3d')
ax.scatter(projx, projy, projz, color = cmap, s = 5)
plt.show()
"""
Takens_x = x
Takens_error = error


data_output = [LS_x[burnin:],LS_y[burnin:],LS_z[burnin:]]
data_input  = x[:,burnin-1:-1].T


def buildNN(min_,max_):
    NN = Sequential()
    NN.add(InputLayer(input_shape=data_input.shape[1]))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    #NN.add(Dense(1,activation = "linear",use_bias=True))
    NN.add(Dense(1,activation = sig_scaled(max_-min_,1,0,min_),use_bias=True))
    #initial_learning_rate = 1e-3
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #initial_learning_rate,decay_steps=32000, decay_rate=0.5,staircase=True)
    #NN.compile(loss = "MSE", optimizer =Adam(learning_rate = lr_schedule,clipvalue=10),metrics=["MAPE"])
    return NN

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience = 500 ,restore_best_weights=True)
NN1tak = buildNN(-30,30)
NN1tak.compile(loss = "MSE", optimizer =Adam(clipvalue=10),metrics=["MAPE"])
history1tak = NN1tak.fit(data_input,data_output[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1tak.compile(loss = "MSE", optimizer =Adam(learning_rate =1e-4,clipvalue=10),metrics=["MAPE"])
history2tak = NN1tak.fit(data_input,data_output[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1tak.save_weights("tak1_udnerfitting.h5")


NN1tak.compile(loss = "MSE", optimizer =Adam(learning_rate =1e-5,clipvalue=10),metrics=["MAPE"])
history3tak = NN1tak.fit(data_input,data_output[0], batch_size=1000, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1tak.save_weights("tak1_udnerfittinge5.h5")


NN1tak.compile(loss = "MSE", optimizer =Adam(learning_rate =5e-6,clipvalue=10),metrics=["MAPE"])
history3tak = NN1tak.fit(data_input,data_output[0], batch_size=1000, epochs =10000, verbose = True, shuffle=1,callbacks=[es])

NN1tak.load_weights("tak1long.h5")

NN2tak = buildNN(-30,30)
#history2tak = NN2tak.fit(data_input,data_output[1], batch_size=500, epochs =20000, verbose = True, shuffle=1,callbacks=[es])    
#NN2tak.save_weights("tak2.h5")
NN2tak.load_weights("tak2.h5")

NN3tak = buildNN(0,55)
#history3tak = NN3tak.fit(data_input,data_output[2], batch_size=500, epochs =20000, verbose = True, shuffle=1,callbacks=[es])
#NN3tak.save_weights("tak3.h5")
NN3tak.load_weights("tak3.h5")


LS_tak_x = NN1tak.predict(data_input)
LS_tak_y = NN2tak.predict(data_input)
LS_tak_z = NN3tak.predict(data_input)
"""
# Plot the Lorenz attractor using a Matplotlib 3D projection
fig = plt.figure(figsize = (6,4))
fig.suptitle("3D-scatter of a Lorenzcurve NN Takens")
ax = fig.gca(projection='3d')
ax.scatter(LS_tak_x[:plotlength], LS_tak_y[:plotlength],LS_tak_z[:plotlength], color = cmap, s = 5)
plt.show()

fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Lorenzcurve per Axis NN Takens")
ax1.scatter(t[burnin:plotlength+burnin],LS_tak_x[:plotlength], color = cmap, s = 5)
ax2.scatter(t[burnin:plotlength+burnin],LS_tak_y[:plotlength], color = cmap, s = 5)
ax3.scatter(t[burnin:plotlength+burnin],LS_tak_z[:plotlength], color = cmap, s = 5)
plt.show()
"""

mape_x = np.zeros(80000,)
mape_y = np.zeros(80000,)
mape_z = np.zeros(80000,)

for i in range(80000):
    mape_x[i] =100*np.abs((LS_tak_x[i]-LS_x[burnin+i])/(LS_x[burnin+i]))
    mape_y[i] =100*np.abs((LS_tak_y[i]-LS_y[burnin+i])/(LS_y[burnin+i]))
    mape_z[i] =100*np.abs((LS_tak_y[i]-LS_y[burnin+i])/(LS_z[burnin+i]))
    
mse_x = np.zeros(80000,)
mse_y = np.zeros(80000,)
mse_z = np.zeros(80000,)

for i in range(80000):
    mse_x[i] =(LS_tak_x[i]-LS_x[burnin+i])**2
    mse_y[i] =(LS_tak_y[i]-LS_y[burnin+i])**2
    mse_z[i] =(LS_tak_y[i]-LS_y[burnin+i])**2
#fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
#fig.suptitle("Mape per Axis NN Takens")
#mape_x= 100*np.abs((LS_tak_x-LS_x[burnin:])/(LS_x[burnin:]))
#mape_y= 100*np.abs((LS_tak_y-LS_y[burnin:])/(LS_x[burnin:]))
#mape_z= 100*np.abs((LS_tak_z-LS_z[burnin:])/(LS_x[burnin:]))
#ax1.scatter(LS_x[burnin:],LS_tak_x)
#ax2.scatter(LS_y[burnin:],LS_tak_y)
#ax3.scatter(LS_z[burnin:],LS_tak_z)
#plt.show()



fc_length = 8000
fc_tak1_x = np.zeros(fc_length,)
fc_tak1_y = np.zeros(fc_length,)
fc_tak1_z = np.zeros(fc_length,)
for i in range(fc_length):
    if i%1000==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xt = data_input[0,:]
        fc_tak1_x[i] =NN1tak.predict(xt.reshape(1,d)).flatten()
        #fc_tak1_y[i] =NN2tak.predict(xt.reshape(1,d)).flatten()
        #fc_tak1_z[i] =NN3tak.predict(xt.reshape(1,d)).flatten()
        #print(np.linalg.norm(xt-data_input[i]))
    else:
        xt = A@xt + C*fc_tak1_x[i-1]
        #print(np.linalg.norm(xt-data_input[i]))
        fc_tak1_x[i] =NN1tak.predict(xt.reshape(1,d)).flatten()
        #fc_tak1_y[i] =NN2tak.predict(xt.reshape(1,d)).flatten()
        #fc_tak1_z[i] =NN3tak.predict(xt.reshape(1,d)).flatten()


mse_fc =(fc_tak1_x-data_output[0][:8000])**2

fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("only lr=1e-5 training /underfitting")
ax1.plot(fc_tak1_x[:5000])
ax1.set_xlabel("x-forecast")
ax2.plot(data_output[0][:5000])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc[:5000])
ax3.set_xlabel("mse")

plt.show()
       



 # forecasting
fc_length = 8000
fc_tak2_x = np.zeros(fc_length,)
fc_tak2_y = np.zeros(fc_length,)
fc_tak2_z = np.zeros(fc_length,)
for i in range(fc_length):
    if i%500==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xt = data_input[-1,:]
        fc_tak2_x[i] =NN1tak.predict(xt.reshape(1,d)).flatten()
        #fc_tak2_y[i] =NN2tak.predict(xt.reshape(1,d)).flatten()
        #fc_tak2_z[i] =NN3tak.predict(xt.reshape(1,d)).flatten()
        
    else:
        xt = A@xt + C*fc_tak2_x[i-1]
        fc_tak2_x[i] =NN1tak.predict(xt.reshape(1,d)).flatten()
        #fc_tak2_y[i] =NN2tak.predict(xt.reshape(1,d)).flatten()
        #fc_tak2_z[i] =NN3tak.predict(xt.reshape(1,d)).flatten()

mse_fc2 =(fc_tak2_x-LS_x2[1100000:1100000+fc_length])**2

# Maximum time point and total number of time points
tmax, n = 12000, 1200001
t = np.linspace(0, tmax, n)
ls = np.linspace(0,1,plotlength)
cmap = cm.viridis(ls)

# Integrate the Lorenz System on the time grid t
f2 = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))
LS_x2, LS_y2, LS_z2 = f2.T


fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Out-of-sample network Takens")
ax1.plot(fc_tak2_x[:5000])
ax1.set_xlabel("x-forecast")
ax2.plot(LS_x2[1100000:1100000+5000])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc2[:5000])
ax3.set_xlabel("mse")

plt.show()
       # In[58]:

np.random.seed(40)
#Create reservoir A#
Arand = np.random.uniform(low=-1.0, high=1.0, size =(d,d))
#Arand /=np.linalg.norm(A,2)
param = 0.9
Arand /= np.real(np.max(np.linalg.eigvals(Arand)))/param

#Create Reservoir C
Crand = np.random.uniform(low = -1.0, high = 1.0, size = d)
Crand /= np.linalg.norm(Crand)


#create trajectory
xrand = np.zeros((d,n))  
error_rand = np.zeros((d,n))
for k in range(1,n):
    xrand[:,k] = Arand@xrand[:,k-1] + Crand*LS_x[k]
    error_rand[:,k] = Arand@error_rand[:,k-1] + Crand*np.random.normal(0,1)
"""    
#Project x onto the first 3 princripal components
U, S, V = np.linalg.svd(xrand[:,burnin:plotlength+burnin])
projx = U[:,0:1].T@xrand[:,burnin:plotlength+burnin]
projy = U[:,1:2].T@xrand[:,burnin:plotlength+burnin]
projz = U[:,2:3].T@xrand[:,burnin:plotlength+burnin]
# Plot the Takens embedded attractor attractor using a Matplotlib 3D projection
fig = plt.figure(figsize = (6,4))
ax = fig.gca(projection='3d')
ax.scatter(projx, projy, projz, color = cmap, s = 5)
plt.show()
"""
data_output2 = [LS_x[burnin:],LS_y[burnin:],LS_z[burnin:]]
data_input2  = xrand[:,burnin-1:-1].T
## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience = 500 ,restore_best_weights=True)
NN1res = buildNN(-30,30)
NN1res.compile(loss = "MSE", optimizer =Adam(clipvalue=10),metrics=["MAPE"])
history1res = NN1res.fit(data_input2,data_output2[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])

NN1res.compile(loss = "MSE", optimizer =Adam(learning_rate =1e-4,clipvalue=10),metrics=["MAPE"])
history2res = NN1res.fit(data_input2,data_output2[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1res.save_weights("tes1_udnerfitting.h5")
NN1res.compile(loss = "MSE", optimizer =Adam(learning_rate =1e-5,clipvalue=10),metrics=["MAPE"])
history3res = NN1res.fit(data_input2,data_output2[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1res.save_weights("tes1_udnerfitting5.h5")


NN1res = buildNN(-30,25)
#history1res = NN1res.fit(data_input2,data_output2[0], batch_size=500, epochs =15000, verbose = True, shuffle=1,callbacks=[es])
#NN1res.save_weights("res1.h5")
NN1res.load_weights("res1.h5")

NN2res = buildNN(-30,30)
#history2res = NN2res.fit(data_input2,data_output2[1], batch_size=500, epochs =15000, verbose = True, shuffle=1,callbacks=[es])    
#NN2res.save_weights("res2.h5")
NN2res.load_weights("res2.h5")

NN3res = buildNN(0,55)  
#NN3res.load_weights("res3.h5")
#history3res = NN3res.fit(data_input2,data_output2[2], batch_size=500, epochs =15000, verbose = True, shuffle=1,callbacks=[es])
NN3res.save_weights("res3.h5")



mape_x_res = np.zeros(80000,)
mape_y_res = np.zeros(80000,)
mape_z_res = np.zeros(80000,)
LS_res_x = NN1res.predict(data_input2)
LS_res_y = NN2res.predict(data_input2)
LS_res_z = NN3res.predict(data_input2)
for i in range(80000):
    mape_x_res[i] =100*np.abs((LS_res_x[i]-LS_x[burnin+i])/(LS_x[burnin+i]))
    mape_y_res[i] =100*np.abs((LS_res_y[i]-LS_y[burnin+i])/(LS_y[burnin+i]))
    mape_z_res[i] =100*np.abs((LS_res_y[i]-LS_y[burnin+i])/(LS_z[burnin+i]))
    
mse_x_res = np.zeros(80000,)
mse_y_res = np.zeros(80000,)
mse_z_res = np.zeros(80000,)

for i in range(80000):
    mse_x_res[i] =(LS_res_x[i]-LS_x[burnin+i])**2
    mse_y_res[i] =(LS_res_y[i]-LS_y[burnin+i])**2
    mse_z_res[i] =(LS_res_y[i]-LS_y[burnin+i])**2

# forecasting
fc_length = 8000
fc_res1_x = np.zeros(fc_length,)
fc_res1_y = np.zeros(fc_length,)
fc_res1_z = np.zeros(fc_length,)
for i in range(fc_length):
    if i%500==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xtres = data_input2[0]
        fc_res1_x[i] =NN1res.predict(xtres.reshape(1,d)).flatten()
        #fc_res1_y[i] =NN2res.predict(xtres.reshape(1,d)).flatten()
        #fc_res1_z[i] =NN3res.predict(xtres.reshape(1,d)).flatten()
        #print(np.linalg.norm(xtres-data_input[i]))

    else:     
        xtres = Arand@xtres + Crand*fc_res1_x[i-1]
        #print(np.linalg.norm(xtres-data_input[i]))
        fc_res1_x[i] =NN1res.predict(xtres.reshape(1,d)).flatten()
        #fc_res1_y[i] =NN2res.predict(xtres.reshape(1,d)).flatten()
        #fc_res1_z[i] =NN3res.predict(xtres.reshape(1,d)).flatten()

mse_fc_res =(fc_res1_x-data_output2[0][:8000])**2

fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Resorvoir:")
ax1.plot(fc_res1_x[:5000])
ax1.set_xlabel("x-forecast")
ax2.plot(data_output2[0][:5000])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc_res[:5000])
ax3.set_xlabel("mse")

plt.show()
       
 # forecasting
fc_length = 8000
fc_res2_x = np.zeros(fc_length,)
fc_res2_y = np.zeros(fc_length,)
fc_res2_z = np.zeros(fc_length,)
for i in range(fc_length):
    if i%500==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xtres = data_input2[-1]
        fc_res2_x[i] =NN1res.predict(xtres.reshape(1,d)).flatten()
        #[i] =NN2res.predict(xtres.reshape(1,d)).flatten()
        #fc_res2_z[i] =NN3res.predict(xtres.reshape(1,d)).flatten()
    else:     
        xtres = Arand@xtres + Crand*fc_res2_x[i-1]
        fc_res2_x[i] =NN1res.predict(xtres.reshape(1,d)).flatten()
        #fc_res2_y[i] =NN2res.predict(xtres.reshape(1,d)).flatten()
        #fc_res2_z[i] =NN3res.predict(xtres.reshape(1,d)).flatten()

mse_fc_res2 =(fc_res2_x-LS_x2[1100000:1100000+fc_length])**2


fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Out-of-sample network resorvoir")
ax1.plot(fc_res2_x[:5000])
ax1.set_xlabel("x-forecast")
ax2.plot(LS_x2[1100000:1100000+5000])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc_res2[:5000])
ax3.set_xlabel("mse")
plt.show()


      
fig = plt.figure(figsize = (6,4))
fig.suptitle("3D-scatter of a Lorenzcurve of pseudoforecast NN Takens")
ax = fig.gca(projection='3d')
ax.scatter(fc_tak1_x[:100],fc_tak1_y[:100],fc_tak1_z[:100])
#ax.scatter(fc_tak2_x[:plotlength],fc_tak2_y[:plotlength],fc_tak2_z[:plotlength], color = cmap, s = 5)
plt.show()
# In[Reservoir with nilpotent matrix]:


np.random.seed(40)
#Create reservoir A#
Anil = np.random.uniform(low=-1.0, high=1.0, size =(d,d))
Anil = np.tril(Anil,k=-1)
#Create Reservoir C
Cnil = np.random.uniform(low = -1.0, high = 1.0, size = d)
Cnil /= np.linalg.norm(Crand)


#create trajectory
xnil = np.zeros((d,1100001))  
error_xnil = np.zeros((d,1100001))
for k in range(1,1100001):
    xnil[:,k] = Anil@xnil[:,k-1] + Cnil*LS_x[k]
    error_xnil[:,k] = Anil@error_xnil[:,k-1] + Cnil*np.random.normal(0,1)
"""    
#Project x onto the first 3 princripal components
U, S, V = np.linalg.svd(xrand[:,burnin:plotlength+burnin])
projx = U[:,0:1].T@xrand[:,burnin:plotlength+burnin]
projy = U[:,1:2].T@xrand[:,burnin:plotlength+burnin]
projz = U[:,2:3].T@xrand[:,burnin:plotlength+burnin]
# Plot the Takens embedded attractor attractor using a Matplotlib 3D projection
fig = plt.figure(figsize = (6,4))
ax = fig.gca(projection='3d')
ax.scatter(projx, projy, projz, color = cmap, s = 5)
plt.show()
"""
data_output3 = [LS_x[burnin:],LS_y[burnin:],LS_z[burnin:]]
data_input3  = xnil[:,burnin-1:-1].T
## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience = 500 ,restore_best_weights=True)
NN1nil = buildNN(-30,30)
NN1nil.compile(loss = "MSE", optimizer =Adam(clipvalue=10),metrics=["MAPE"])
history1nil = NN1nil.fit(data_input3,data_output3[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1nil.compile(loss = "MSE", optimizer =Adam(learning_rate =1e-4,clipvalue=10),metrics=["MAPE"])
history2nil = NN1nil.fit(data_input3,data_output3[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1nil.load_weights("nil1_udnerfitting.h5")
NN1nil.compile(loss = "MSE", optimizer =Adam(learning_rate =1e-5,clipvalue=10),metrics=["MAPE"])
history3nil = NN1nil.fit(data_input3,data_output3[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1nil.save_weights("nil1_udnerfitting5.h5")


fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Resorvoir:")
ax1.plot(fc_res1_x[:5000])
ax1.set_xlabel("x-forecast")
ax2.plot(data_output2[0][:5000])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc_res[:5000])
ax3.set_xlabel("mse")

plt.show()
       
 # forecasting
fc_length = 8000
fc_nil2_x = np.zeros(fc_length,)
fc_nil2_y = np.zeros(fc_length,)
fc_nil2_z = np.zeros(fc_length,)
for i in range(fc_length):
    if i%500==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xtnil = data_input3[-1]
        fc_nil2_x[i] =NN1nil.predict(xtnil.reshape(1,d)).flatten()
        #[i] =NN2res.predict(xtres.reshape(1,d)).flatten()
        #fc_res2_z[i] =NN3res.predict(xtres.reshape(1,d)).flatten()
    else:     
        xtnil = Anil@xtnil + Cnil*fc_nil2_x[i-1]
        fc_nil2_x[i] =NN1nil.predict(xtnil.reshape(1,d)).flatten()
        #fc_res2_y[i] =NN2res.predict(xtres.reshape(1,d)).flatten()
        #fc_res2_z[i] =NN3res.predict(xtres.reshape(1,d)).flatten()

mse_fc_nil2 =(fc_nil2_x-LS_x2[1100000:1100000+fc_length])**2


fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Out-of-sample network nil resorvoir")
ax1.plot(fc_nil2_x[:5000])
ax1.set_xlabel("x-forecast")
ax2.plot(LS_x2[1100000:1100000+5000])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc_nil2[:5000])
ax3.set_xlabel("mse")
plt.show()

# forecasting
fc_length = 8000
fc_nil_x = np.zeros(fc_length,)
fc_nil_y = np.zeros(fc_length,)
fc_nil_z = np.zeros(fc_length,)
for i in range(fc_length):
    if i%500==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xtnil = data_input3[0]
        fc_nil_x[i] =NN1nil.predict(xtnil.reshape(1,d)).flatten()
        #[i] =NN2res.predict(xtres.reshape(1,d)).flatten()
        #fc_res2_z[i] =NN3res.predict(xtres.reshape(1,d)).flatten()
    else:     
        xtnil = Anil@xtnil + Cnil*fc_nil_x[i-1]
        fc_nil_x[i] =NN1nil.predict(xtnil.reshape(1,d)).flatten()
        #fc_res2_y[i] =NN2res.predict(xtres.reshape(1,d)).flatten()
        #fc_res2_z[i] =NN3res.predict(xtres.reshape(1,d)).flatten()

mse_fc_nil =(fc_nil_x-data_output3[0][:fc_length])**2


fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("In-sample  nil resorvoir")
ax1.plot(fc_nil_x[:5000])
ax1.set_xlabel("x-forecast")
ax2.plot(data_output3[0][:5000])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc_nil[:5000])
ax3.set_xlabel("mse")
plt.show()




























#add noise
noise_scale = 0

#training data

# features
Takens_features = Takens_x[:,100:] + noise_scale*Takens_error[:,100:]
Reservoir_features = x[:,100:] + noise_scale*error[:,100:]

# targets
targets = LS_z[100:] + noise_scale*np.random.normal(0,1,size = n-100)

# find a map from the Takens_features to the targets using some supervised learning technique (e.g. a feedforward NN)

# find a map from the Reservoir_features to the targets using the same technique

# You may reserve a segment of training data as 'validation data' to validate the model

# by increasing the 'noise scale' you add noise to the training data, which makes things harder.

# We expect that the Reservoir_features are more robust under noise than the Takens features.



# In[ ]:




