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

# In[Generate Lorenzcurve]:
    
# Lorenz paramters and initial conditions
sigma, beta, rho = 10, 2.667, 28
u0, v0, w0 = 0, 1, 1.05

# Maximum time point and total number of time points
tmax, n = 12000, 1200001

plotlength = 2000
burnin = 100001
outofsample=1100001
t = np.linspace(0, tmax, n)
ls = np.linspace(0,1,plotlength)
cmap = cm.viridis(ls)

# Integrate the Lorenz System on the time grid t
f = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))
LS_x, LS_y, LS_z = f.T

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
# In[Takes Network]:

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


data_output = [LS_x[burnin:outofsample],LS_y[burnin:outofsample],LS_z[burnin:outofsample]]
data_outofsample = [LS_x[outofsample:],LS_y[outofsample:],LS_z[outofsample:]]
data_input_tak  = x[:,burnin-1:outofsample-1].T


def buildNN(min_,max_):
    NN = Sequential()
    NN.add(InputLayer(input_shape=d))
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
NN1tak.save_weights("tak_e5_x.h5")
LS_tak_x = NN1tak.predict(data_input_tak)

mape_x = np.zeros(1000000,)
for i in range(1000000):
    mape_x[i] =100*np.abs((LS_tak_x[i]-LS_x[burnin+i])/(LS_x[burnin+i]))
    
mse_x = np.zeros(1000000,)
for i in range(1000000):
    mse_x[i] =(LS_tak_x[i]-LS_x[burnin+i])**2



fc_length = 2000
fc_tak1_x = np.zeros(fc_length,)
fc_tak1_y = np.zeros(fc_length,)
fc_tak1_z = np.zeros(fc_length,)

for i in range(fc_length):
    if i%1000==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xt = data_input_tak[0,:]
        fc_tak1_x[i] =NN1tak.predict(xt.reshape(1,d)).flatten()
        fc_tak1_y[i] =NN1tak_y.predict(xt.reshape(1,d)).flatten()
        fc_tak1_z[i] =NN1tak_z.predict(xt.reshape(1,d)).flatten()
    else:
        xt = A@xt + C*fc_tak1_x[i-1]
        fc_tak1_x[i] =NN1tak.predict(xt.reshape(1,d)).flatten()
        fc_tak1_y[i] =NN1tak_y.predict(xt.reshape(1,d)).flatten()
        fc_tak1_z[i] =NN1tak_z.predict(xt.reshape(1,d)).flatten()
   
mse_fc =(fc_tak1_x-data_output[0][:fc_length])**2
mse_fc_y =(fc_tak1_y-data_output[1][:fc_length])**2
mse_fc_z =(fc_tak1_z-data_output[2][:fc_length])**2


fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Takens NN: Pseudo-Insample Forecast")
ax1.plot(fc_tak1_x)
ax1.set_xlabel("x-forecast")
ax2.plot(data_output[0][:fc_length])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc)
ax3.set_xlabel("mse")

plt.show()
fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Takens NN: Pseudo-Insample Forecast yaxis")
ax1.plot(fc_tak1_x)
ax1.set_xlabel("x-forecast")
ax2.plot(data_output[0][:fc_length])
ax2.set_xlabel("Lorenz y-axis")
ax3.plot(mse_fc_y)
ax3.set_xlabel("mse")

plt.show()


 # forecasting
fc_tak2_x = np.zeros(fc_length,)
fc_tak2_y = np.zeros(fc_length,)
fc_tak2_z = np.zeros(fc_length,)

for i in range(fc_length):
    if i%500==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xt = data_input_tak[-1]
        fc_tak2_x[i] =NN1tak.predict(xt.reshape(1,d)).flatten()
        fc_tak2_y[i] =NN1tak_y.predict(xt.reshape(1,d)).flatten()
        fc_tak2_z[i] =NN1tak_z.predict(xt.reshape(1,d)).flatten()
    else:     
        xt = A@xt +C*fc_tak2_x[i-1]
        fc_tak2_x[i] =NN1tak.predict(xt.reshape(1,d)).flatten()
        fc_tak2_y[i] =NN1tak_y.predict(xt.reshape(1,d)).flatten()
        fc_tak2_z[i] =NN1tak_z.predict(xt.reshape(1,d)).flatten()
    

mse_fc_2 =(fc_tak2_x[1:]-data_outofsample[0][:fc_length-1])**2
mse_fc_2y =(fc_tak2_y[1:]-data_outofsample[1][:fc_length-1])**2
mse_fc_2z =(fc_tak2_z[1:]-data_outofsample[2][:fc_length-1])**2

plt.show()
fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Takens NN:outof sample yxis")
ax1.plot(fc_tak2_y)
ax1.set_xlabel("y-forecast")
ax2.plot(data_output[1][:fc_length])
ax2.set_xlabel("Lorenz y-axis")
ax3.plot(mse_fc_2y)
ax3.set_xlabel("mse")

plt.show()


# In[Reservoir Sparse rho0.9]:


np.random.seed(40)
#Create reservoir A#
Aspat = np.random.uniform(low=-1.0, high=1.0, size =(d,d))
percen =0.5
rand_list = np.random.permutation(d*d)[:int(d*d*percen)]
ind_list = np.zeros((d*d,1),dtype=bool)
ind_list[rand_list] = True
ind_list = ind_list.reshape((d,d))
Aspat[ind_list] =0
param = 0.9
Aspat /= np.real(np.max(np.linalg.eigvals(Aspat)))/param

#Create Reservoir C
Cspat = np.random.uniform(low = -1.0, high = 1.0, size = d)
Cspat /= np.linalg.norm(Cspat)


#create trajectory
xspat = np.zeros((d,n))  
error_spat = np.zeros((d,n))
eps = np.random.normal(0,1,n)
for k in range(1,n):
    xspat[:,k] = Aspat@xspat[:,k-1] + Cspat*LS_x[k]
    error_spat[:,k] = Aspat@error_spat[:,k-1] + Cspat*eps[i]

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
data_input_spat = xspat[:,burnin-1:outofsample-1].T
## Training
NN1spat = buildNN(-30,30)
NN1spat.compile(loss = "MSE", optimizer =Adam(clipvalue=10),metrics=["MAPE"])
history1spat = NN1spat.fit(data_input_spat,data_output[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1spat.save_weights("spat_e3.h5")
NN1spat.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-4),metrics=["MAPE"])
history2spat = NN1spat.fit(data_input_spat,data_output[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1spat.save_weights("spat_e4.h5")
NN1spat.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=["MAPE"])
history3spat = NN1spat.fit(data_input_spat,data_output[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1spat.load_weights("spat_e5.h5")
data_input_spat = xspat[:,burnin-1:outofsample-1].T
## Training
NN1spaty = buildNN(-30,30)
NN1spaty.compile(loss = "MSE", optimizer =Adam(clipvalue=10),metrics=["MAPE"])
history1spaty = NN1spaty.fit(data_input_spat,data_output[1], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1spaty.save_weights("spat_e3_y.h5")
NN1spaty.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-4),metrics=["MAPE"])
history2spaty = NN1spaty.fit(data_input_spat,data_output[1], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1spaty.save_weights("spat_e4_y.h5")
NN1spaty.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=["MAPE"])
history3spaty = NN1spaty.fit(data_input_spat,data_output[1], batch_size=2500, epochs =15000, verbose = True, shuffle=1,callbacks=[es])
NN1spaty.save_weights("spat_e5_y.h5")

es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience = 200 ,restore_best_weights=True)
NN1spatz = buildNN(-5,55)
NN1spatz.load_weights("spat_e3_z.h5")
NN1spatz.compile(loss = "MSE", optimizer =Adam(clipvalue=10),metrics=["MAPE"])
history1spatz = NN1spatz.fit(data_input_spat,data_output[2], batch_size=2500, epochs =2000, verbose = True, shuffle=1,callbacks=[es])
NN1spatz.save_weights("spat_e3_z.h5")
NN1spatz.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-4),metrics=["MAPE"])
history2spatz = NN1spatz.fit(data_input_spat,data_output[2], batch_size=2500, epochs =2000, verbose = True, shuffle=1,callbacks=[es])
NN1spatz.save_weights("spat_e4_z.h5")
NN1spatz.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=["MAPE"])
history3spatz = NN1spatz.fit(data_input_spat,data_output[2], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1spatz.save_weights("spat_e5_z.h5")


NN1tak_y = buildNN(-30,30)
NN1tak_y.compile(loss = "MSE", optimizer =Adam(clipvalue=10),metrics=["MAPE"])
history1taky = NN1tak_y.fit(data_input_tak,data_output[1], batch_size=2500, epochs =2000, verbose = True, shuffle=1,callbacks=[es])
NN1tak_y.save_weights("tak_e3_y.h5")
NN1tak_y.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-4),metrics=["MAPE"])
history2taky = NN1tak_y.fit(data_input_tak,data_output[1], batch_size=2500, epochs =2000, verbose = True, shuffle=1,callbacks=[es])
NN1tak_y.save_weights("tak_e4_y.h5")
NN1tak_y.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=["MAPE"])
history3taky = NN1tak_y.fit(data_input_tak,data_output[1], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1tak_y.save_weights("tak_e5_y.h5")
NN1tak_z = buildNN(-5,55)
NN1tak_z.compile(loss = "MSE", optimizer =Adam(clipvalue=10),metrics=["MAPE"])
history1takz = NN1tak_z.fit(data_input_tak,data_output[2], batch_size=2500, epochs =2000, verbose = True, shuffle=1,callbacks=[es])
NN1tak_z.save_weights("tak_e3_z.h5")
NN1tak_z.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-4),metrics=["MAPE"])
history2takz = NN1tak_z.fit(data_input_tak,data_output[2], batch_size=2500, epochs =2000, verbose = True, shuffle=1,callbacks=[es])
NN1tak_z.load_weights("tak_e4_z.h5")
NN1tak_z.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=["MAPE"])
history3takz = NN1tak_z.fit(data_input_tak,data_output[2], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1tak_z.save_weights("tak_e5_z.h5")




"""mape_x_res = np.zeros(80000,)
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
"""
# forecasting
fc_spat1_x = np.zeros(fc_length,)
fc_spat1_y = np.zeros(fc_length,)
fc_spat1_z = np.zeros(fc_length,)

for i in range(fc_length):
    if i%500==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xtspat = data_input_spat[0]
        fc_spat1_x[i] =NN1spat.predict(xtspat.reshape(1,d)).flatten()
        fc_spat1_y[i] =NN1spaty.predict(xtspat.reshape(1,d)).flatten()
        fc_spat1_z[i] =NN1spatz.predict(xtspat.reshape(1,d)).flatten()
    else:     
        xtspat = Aspat@xtspat + Cspat*fc_spat1_x[i-1]
        fc_spat1_x[i] =NN1spat.predict(xtspat.reshape(1,d)).flatten()
        fc_spat1_y[i] =NN1spaty.predict(xtspat.reshape(1,d)).flatten()
        fc_spat1_z[i] =NN1spatz.predict(xtspat.reshape(1,d)).flatten()
 
mse_fc_spat =(fc_spat1_x-data_output[0][:fc_length])**2
mse_fc_spaty =(fc_spat1_y-data_output[1][:fc_length])**2
mse_fc_spatz =(fc_spat1_z-data_output[2][:fc_length])**2

fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Resorvoir Sparse NN: Pseudo-Insample Forecast")
ax1.plot(fc_spat1_x)
ax1.set_xlabel("x-forecast")
ax2.plot(data_output[0][:fc_length])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc_spat)
ax3.set_xlabel("mse")

plt.show()


fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Resorvoir Sparse NN: Pseudo-Insample Forecast yaxis")
ax1.plot(fc_spat1_y)
ax1.set_xlabel("y-forecast")
ax2.plot(data_output[1][:fc_length])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc_spaty)
ax3.set_xlabel("mse")

plt.show()

fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Resorvoir Sparse NN: Pseudo-Insample Forecast zaxis")
ax1.plot(fc_spat1_z)
ax1.set_xlabel("z-forecast")
ax2.plot(data_output[2][:fc_length])
ax2.set_xlabel("Lorenz z-axis")
ax3.plot(mse_fc_spatz)
ax3.set_xlabel("mse")

plt.show()
  





 # forecasting
fc_spat2_x = np.zeros(fc_length,)
fc_spat2_y = np.zeros(fc_length,)
fc_spat2_z = np.zeros(fc_length,)

for i in range(fc_length):
    if i%500==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xtspat = data_input_spat[-1]
        fc_spat2_x[i] =NN1spat.predict(xtspat.reshape(1,d)).flatten()
        fc_spat2_y[i] =NN1spaty.predict(xtspat.reshape(1,d)).flatten()
        fc_spat2_z[i] =NN1spatz.predict(xtspat.reshape(1,d)).flatten()
    else:     
        xtspat = Aspat@xtspat+ Cspat*fc_spat2_x[i-1]
        fc_spat2_x[i] =NN1spat.predict(xtspat.reshape(1,d)).flatten()
        fc_spat2_y[i] =NN1spaty.predict(xtspat.reshape(1,d)).flatten()
        fc_spat2_z[i] =NN1spatz.predict(xtspat.reshape(1,d)).flatten()
    

mse_fc_spat2 =(fc_spat2_x[1:]-data_outofsample[0][:fc_length-1])**2
mse_fc_spat2y =(fc_spat2_y[1:]-data_outofsample[1][:fc_length-1])**2
mse_fc_spat2z =(fc_spat2_z[1:]-data_outofsample[2][:fc_length-1])**2


fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Out-of-sample network sparse resorvoir")
ax1.plot(fc_spat2_x[1:])
ax1.set_xlabel("x-forecast")
ax2.plot(data_outofsample[0][:fc_length-1])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc_spat2)
ax3.set_xlabel("mse")
plt.show()
fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Out-of-sample network sparse resorvoir yxis")
ax1.plot(fc_spat2_y[1:])
ax1.set_xlabel("y-forecast")
ax2.plot(data_outofsample[1][:fc_length-1])
ax2.set_xlabel("Lorenz y-axis")
ax3.plot(mse_fc_spat2y)
ax3.set_xlabel("mse")
plt.show()
fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Out-of-sample network sparse resorvoir zxis")
ax1.plot(fc_spat2_z[1:])
ax1.set_xlabel("y-forecast")
ax2.plot(data_outofsample[2][:fc_length-1])
ax2.set_xlabel("Lorenz z-axis")
ax3.plot(mse_fc_spat2z)
ax3.set_xlabel("mse")
plt.show()



      
fig = plt.figure(figsize = (6,4))
fig.suptitle("3D-scatter of a Lorenzcurve of pseudoforecast Sparse Reservour")
ax = fig.gca(projection='3d')
ax.scatter(fc_spat1_x[:750],fc_spat1_y[:750],fc_spat1_z[:750])#, color = cmap, s = 5)
ax.scatter(fc_tak1_x[:750],fc_tak1_y[:750],fc_tak1_z[:750])#, color = cmap, s = 5)
ax.scatter(data_output[0][:750],data_output[1][:750],data_output[2][:750])#, color = cmap, s = 5)
fig.legend(["Sparse Reservoir","Takens NN","True Lorenz"])
plt.show()

plt.figure()
plt.suptitle("Out-of-sample Performance")
plt.plot(fc_spat2_x[1:750])
plt.plot(fc_tak2_x[1:750])
plt.plot(data_outofsample[0][:750-1])
plt.legend(["Sparse Reservoir","Takensn NN","True Lorenz"])
plt.show()

plt.figure()
plt.suptitle("Out-of-sample Performance: y-axis")
plt.plot(fc_spat2_y[1:750])
plt.plot(fc_tak2_y[1:750])
plt.plot(data_outofsample[1][:750-1])
plt.legend(["Sparse Reservoir","Takensn NN","True Lorenz"])
plt.show()

plt.figure()
plt.suptitle("Out-of-sample Performance: z-axis")
plt.plot(fc_spat2_z[1:750])
plt.plot(fc_tak2_z[1:750])
plt.plot(data_outofsample[2][:750-1])
plt.legend(["Sparse Reservoir","Takensn NN","True Lorenz"])
plt.show() 

plt.figure()
plt.suptitle("In-sample Performance: x-axis")
plt.plot(fc_spat1_x[:750])
plt.plot(fc_tak1_x[:750])
plt.plot(data_output[0][:750])
plt.legend(["Sparse Reservoir","Takensn NN","True Lorenz"])
plt.show()

plt.figure()
plt.suptitle("In-sample Performance: y-axis")
plt.plot(fc_spat1_y[:750])
plt.plot(fc_tak1_y[:750])
plt.plot(data_output[1][:750])
plt.legend(["Sparse Reservoir","Takensn NN","True Lorenz"])
plt.show()

plt.figure()
plt.suptitle("In-sample Performance: z-axis")
plt.plot(fc_spat1_z[:750])
plt.plot(fc_tak1_z[:750])
plt.plot(data_output[2][:750])
plt.legend(["Sparse Reservoir","Takensn NN","True Lorenz"])
plt.show()



plt.figure()
plt.suptitle("MSE: In-sample Performance: x-axis")
#plt.yscale("log")
plt.plot(mse_fc[:750])
plt.plot(mse_fc_spat[:750])
plt.legend(["Takensn NN: cumMSE="+str(np.sum(mse_fc[:750])),"Sparse Reservoir: cumMSE="+str(np.sum(mse_fc_spat[:750]))])
plt.show()
plt.figure()
plt.suptitle("MSE: In-sample Performance: y-axis")
#plt.yscale("log")
plt.plot(mse_fc_y[:750])
plt.plot(mse_fc_spaty[:750])
plt.legend(["Takensn NN: cumMSE="+str(np.sum(mse_fc_y[:750])),"Sparse Reservoir: cumMSE="+str(np.sum(mse_fc_spaty[:750]))])
plt.show()

plt.figure()
plt.suptitle("MSE: Out-of-sample Performance: x-axis")
#plt.yscale("log")
plt.plot(mse_fc_2[:750])
plt.plot(mse_fc_spat2[:750])
plt.legend(["Takensn NN: cumMSE="+str(np.sum(mse_fc_2[:750])),"Sparse Reservoir: cumMSE="+str(np.sum(mse_fc_spat2[:750]))])
plt.show()
plt.figure()
plt.suptitle("MSE: Out-of-sample Performance: y-axis")
#plt.yscale("log")
plt.plot(mse_fc_2y[:750])
plt.plot(mse_fc_spat2y[:750])
plt.legend(["Takensn NN: cumMSE="+str(np.sum(mse_fc_2y[:750])),"Sparse Reservoir: cumMSE="+str(np.sum(mse_fc_spat2y[:750]))])
plt.show()

plt.figure()
plt.suptitle("MSE: Out-of-sample Performance: z-axis")
#.yscale("log")
plt.plot(mse_fc_2z[:750])
plt.plot(mse_fc_spat2z[:750])
plt.legend(["Takensn NN: cumMSE="+str(np.sum(mse_fc_2z[:750])),"Sparse Reservoir: cumMSE="+str(np.sum(mse_fc_spat2z[:750]))])
plt.show()
plt.figure()
plt.suptitle("MSE: In-sample Performance: z-axis")
#plt.yscale("log")
plt.plot(mse_fc_z[:750])
plt.plot(mse_fc_spatz[:750])
plt.legend(["Takensn NN: cumMSE="+str(np.sum(mse_fc_z[:750])),"Sparse Reservoir: cumMSE="+str(np.sum(mse_fc_spatz[:750]))])
plt.show()

plt.figure()
plt.suptitle("R^3 norm (eukl distance) between True Solution and Sparse Reservoir (insample)")
dist_spat = np.sqrt(mse_fc_spatz[:750]**2+mse_fc_spat[:750]**2+mse_fc_spaty[:750]**2)
dist_tak = np.sqrt(mse_fc_z[:750]**2+mse_fc[:750]**2+mse_fc_y[:750]**2)

plt.plot(dist_spat)
plt.plot(dist_tak)
plt.legend(["Sparse Reservoir","Takensn NN"])

plt.show()

plt.figure()
plt.suptitle("R^3 norm (eukl distance) between True Solution and Sparse Reservoir (out-of-sample)")
dist_spat2 = np.sqrt(mse_fc_spat2z[:750]**2+mse_fc_spat2[:750]**2+mse_fc_spat2y[:750]**2)
dist_tak2 = np.sqrt(mse_fc_2z[:750]**2+mse_fc_2[:750]**2+mse_fc_2y[:750]**2)
plt.plot(dist_spat2)
plt.plot(dist_tak2)
plt.legend(["Sparse Reservoir","Takensn NN"])
plt.show()

plt.figure()
plt.suptitle("Cummulative R^3 norm (eukl distance) between True Solution and Sparse Reservoir (insample)")
plt.plot(np.cumsum(dist_spat))
plt.plot(np.cumsum(dist_tak))
plt.legend(["Sparse Reservoir","Takensn NN"])

plt.show()

plt.figure()
plt.suptitle("Cummulative R^3 norm (eukl distance) between True Solution and Sparse Reservoir (out-of-sample)")
plt.plot(np.cumsum(dist_spat2))
plt.plot(np.cumsum(dist_tak2))
plt.legend(["Sparse Reservoir","Takensn NN"])

plt.show()
plt.figure()
plt.suptitle("Cum. Mean R^3 norm between True Solution and Sparse Reservoir (insample)")
plt.plot(np.cumsum(dist_spat)/np.arange(1,751))
plt.plot(np.cumsum(dist_tak)/np.arange(1,751))
plt.legend(["Sparse Reservoir","Takensn NN"])

plt.show()

plt.figure()
plt.suptitle("Cum. Mean R^3 norm between True Solution and Sparse Reservoir (out-of-sample)")
plt.plot(np.cumsum(dist_spat2)/np.arange(1,751))
plt.plot(np.cumsum(dist_tak2)/np.arange(1,751))
plt.legend(["Sparse Reservoir","Takensn NN"])

plt.show()

# In[Reservoir Sparse rho0.7]:


np.random.seed(40)
#Create reservoir A#
Aspat7 = np.random.uniform(low=-1.0, high=1.0, size =(d,d))
percen =0.5
rand_list = np.random.permutation(d*d)[:int(d*d*percen)]
ind_list = np.zeros((d*d,1),dtype=bool)
ind_list[rand_list] = True
ind_list = ind_list.reshape((d,d))
Aspat7[ind_list] =0
param = 0.7
Aspat7 /= np.real(np.max(np.linalg.eigvals(Aspat7)))/param

#Create Reservoir C
Cspat7 = np.random.uniform(low = -1.0, high = 1.0, size = d)
Cspat7 /= np.linalg.norm(Cspat7)


#create trajectory
xspat7 = np.zeros((d,n))  
error_spat7 = np.zeros((d,n))
eps = np.random.normal(0,1,n)
for k in range(1,n):
    xspat7[:,k] = Aspat7@xspat7[:,k-1] + Cspat7*LS_x[k]
    error_spat7[:,k] = Aspat7@error_spat7[:,k-1] + Cspat7*eps[i]

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
data_input_spat7 = xspat7[:,burnin-1:outofsample-1].T
## Training
NN1spat7 = buildNN(-30,30)
NN1spat7.compile(loss = "MSE", optimizer =Adam(clipvalue=10),metrics=["MAPE"])
history1spat7 = NN1spat7.fit(data_input_spat7,data_output[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1spat7.save_weights("spat7_e3.h5")
NN1spat7.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-4),metrics=["MAPE"])
history2spat7 = NN1spat7.fit(data_input_spat7,data_output[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1spat7.save_weights("spat7_e4.h5")
NN1spat7.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=["MAPE"])
history3spat7 = NN1spat7.fit(data_input_spat7,data_output[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1spat7.save_weights("spat7_e5.h5")

 # forecasting
fc_spat72_x = np.zeros(fc_length,)
for i in range(fc_length):
    if i%500==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xtspat7 = data_input_spat7[-1]
        fc_spat72_x[i] =NN1spat7.predict(xtspat7.reshape(1,d)).flatten()
    else:     
        xtspat7 = Aspat7@xtspat7+ Cspat7*fc_spat72_x[i-1]
        fc_spat72_x[i] =NN1spat7.predict(xtspat7.reshape(1,d)).flatten()
 
mse_fc_spat72 =(fc_spat72_x[1:]-data_outofsample[0][:fc_length-1])**2
fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Out-of-sample network sparse rho=0.7 resorvoir")
ax1.plot(fc_spat72_x[1:])
ax1.set_xlabel("x-forecast")
ax2.plot(data_outofsample[0][:fc_length-1])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc_spat72)
ax3.set_xlabel("mse")
plt.show()

# forecasting
fc_spat71_x = np.zeros(fc_length,)
for i in range(fc_length):
    if i%500==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xtspat7 = data_input_spat7[0]
        fc_spat71_x[i] =NN1spat7.predict(xtspat7.reshape(1,d)).flatten()
    else:     
        xtspat7 = Aspat7@xtspat7 + Cspat7*fc_spat71_x[i-1]
        fc_spat71_x[i] =NN1spat7.predict(xtspat7.reshape(1,d)).flatten()
      
mse_fc_spat7 =(fc_spat71_x-data_output[0][:fc_length])**2

fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Resorvoir Sparse rho=0.7 NN: Pseudo-Insample Forecast")
ax1.plot(fc_spat71_x)
ax1.set_xlabel("x-forecast")
ax2.plot(data_output[0][:fc_length])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc_spat7)
ax3.set_xlabel("mse")

plt.show()
# In[Reservoir Sparse rho0.99]:


np.random.seed(40)
#Create reservoir A#
Aspat11 = np.random.uniform(low=-1.0, high=1.0, size =(d,d))
percen =0.5
rand_list = np.random.permutation(d*d)[:int(d*d*percen)]
ind_list = np.zeros((d*d,1),dtype=bool)
ind_list[rand_list] = True
ind_list = ind_list.reshape((d,d))
Aspat11[ind_list] =0
param = 0.99
print(np.max(np.linalg.eigvals(Aspat11)))
Aspat11 /= np.real(np.max(np.linalg.eigvals(Aspat11)))/param

#Create Reservoir C
Cspat11 = np.random.uniform(low = -1.0, high = 1.0, size = d)
Cspat11 /= np.linalg.norm(Cspat11)


#create trajectory
xspat11 = np.zeros((d,n))  
error_spat11= np.zeros((d,n))
eps = np.random.normal(0,1,n)
for k in range(1,n):
    xspat11[:,k] = Aspat11@xspat11[:,k-1] + Cspat11*LS_x[k]
    error_spat11[:,k] = Aspat11@error_spat11[:,k-1] + Cspat11*eps[i]
    #if k%5000==0:
   #     print(np.min(xspat11))
   #     print(k)

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
data_input_spat11 = xspat11[:,burnin-1:outofsample-1].T
## Training
NN1spat11 = buildNN(-30,30)
NN1spat11.compile(loss = "MSE", optimizer =Adam(clipvalue=10),metrics=["MAPE"])
history1spat11 = NN1spat11.fit(data_input_spat11,data_output[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1spat11.save_weights("spat11_e3.h5")
NN1spat11.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-4),metrics=["MAPE"])
history2spat11 = NN1spat11.fit(data_input_spat11,data_output[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1spat11.save_weights("spat11_e4.h5")
NN1spat11.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=["MAPE"])
history3spat11 = NN1spat11.fit(data_input_spat11,data_output[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1spat11.save_weights("spat11_e5.h5")

 # forecasting
fc_spat11_x = np.zeros(fc_length,)
for i in range(fc_length):
    if i%500==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xtspat11 = data_input_spat11[-1]
        fc_spat11_x[i] =NN1spat11.predict(xtspat11.reshape(1,d)).flatten()
    else:     
        xtspat11 = Aspat11@xtspat11+ Cspat11*fc_spat11_x[i-1]
        fc_spat11_x[i] =NN1spat11.predict(xtspat11.reshape(1,d)).flatten()
 
mse_fc_spat11 =(fc_spat11_x[1:]-data_outofsample[0][:fc_length-1])**2
fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Out-of-sample network sparse rho=0.99 resorvoir")
ax1.plot(fc_spat11_x[1:])
ax1.set_xlabel("x-forecast")
ax2.plot(data_outofsample[0][:fc_length-1])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc_spat11)
ax3.set_xlabel("mse")
plt.show()

# forecasting
fc_spat111_x = np.zeros(fc_length,)
for i in range(fc_length):
    if i%500==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xtspat11 = data_input_spat11[0]
        fc_spat111_x[i] =NN1spat11.predict(xtspat11.reshape(1,d)).flatten()
    else:     
        xtspat11 = Aspat11@xtspat11 + Cspat11*fc_spat111_x[i-1]
        fc_spat111_x[i] =NN1spat11.predict(xtspat11.reshape(1,d)).flatten()
      
mse_fc_spat111 =(fc_spat111_x-data_output[0][:fc_length])**2

fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Resorvoir Sparse rho=0.99 NN: Pseudo-Insample Forecast")
ax1.plot(fc_spat111_x)
ax1.set_xlabel("x-forecast")
ax2.plot(data_output[0][:fc_length])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc_spat111)
ax3.set_xlabel("mse")

plt.show()
# In[Reservoir Sparse rho0.9 SPARSITY 0.75]:


np.random.seed(40)
#Create reservoir A#
Aspat11 = np.random.uniform(low=-1.0, high=1.0, size =(d,d))
percen =0.75
rand_list = np.random.permutation(d*d)[:int(d*d*percen)]
ind_list = np.zeros((d*d,1),dtype=bool)
ind_list[rand_list] = True
ind_list = ind_list.reshape((d,d))
Aspat11[ind_list] =0
param = 0.9
eig_vals = np.linalg.eigvals(Aspat11).reshape(1,-1).T
norm_eig_vals = np.linalg.norm(np.linalg.eigvals(Aspat11).reshape(1,-1),axis=0)
Aspat11 /= np.max(norm_eig_vals)/param

#Create Reservoir C
Cspat11 = np.random.uniform(low = -1.0, high = 1.0, size = d)
Cspat11 /= np.linalg.norm(Cspat11)

U,S,V =np.linalg.svd(Aspat11)
print(S[0])
#create trajectory
xspat11 = np.zeros((d,n))  
error_spat11= np.zeros((d,n))
eps = np.random.normal(0,1,n)
for k in range(1,n):
    xspat11[:,k] = Aspat11@xspat11[:,k-1] + Cspat11*LS_x[k]
    error_spat11[:,k] = Aspat11@error_spat11[:,k-1] + Cspat11*eps[i]
    if k%5000==0:
       print(np.min(xspat11))
       print(k)


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
data_input_spat11 = xspat11[:,burnin-1:outofsample-1].T
## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience = 200 ,restore_best_weights=True)

NN1spat11 = buildNN(-30,30)
NN1spat11.compile(loss = "MSE", optimizer =Adam(clipvalue=10),metrics=["MAPE"])
history1spat11 = NN1spat11.fit(data_input_spat11,data_output[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1spat11.save_weights("spat975_e3.h5")
NN1spat11.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-4),metrics=["MAPE"])
history2spat11 = NN1spat11.fit(data_input_spat11,data_output[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1spat11.save_weights("spat975_e4.h5")
NN1spat11.compile(loss = "MSE", optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=["MAPE"])
history3spat11 = NN1spat11.fit(data_input_spat11,data_output[0], batch_size=2500, epochs =10000, verbose = True, shuffle=1,callbacks=[es])
NN1spat11.save_weights("spat975_e5.h5")

 # forecasting
fc_spat11_x = np.zeros(fc_length,)
for i in range(fc_length):
    if i%500==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xtspat11 = data_input_spat11[-1]
        fc_spat11_x[i] =NN1spat11.predict(xtspat11.reshape(1,d)).flatten()
    else:     
        xtspat11 = Aspat11@xtspat11+ Cspat11*fc_spat11_x[i-1]
        fc_spat11_x[i] =NN1spat11.predict(xtspat11.reshape(1,d)).flatten()
 
mse_fc_spat11 =(fc_spat11_x[1:]-data_outofsample[0][:fc_length-1])**2
fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Out-of-sample network sparse rho=0.9 spar=0.75 resorvoir")
ax1.plot(fc_spat11_x[1:])
ax1.set_xlabel("x-forecast")
ax2.plot(data_outofsample[0][:fc_length-1])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc_spat11)
ax3.set_xlabel("mse")
plt.show()

# forecasting
fc_spat111_x = np.zeros(fc_length,)
for i in range(fc_length):
    if i%500==0:
        print(str(i/fc_length*100)+"%")
    if i==0:
        xtspat11 = data_input_spat11[0]
        fc_spat111_x[i] =NN1spat11.predict(xtspat11.reshape(1,d)).flatten()
    else:     
        xtspat11 = Aspat11@xtspat11 + Cspat11*fc_spat111_x[i-1]
        fc_spat111_x[i] =NN1spat11.predict(xtspat11.reshape(1,d)).flatten()
      
mse_fc_spat111 =(fc_spat111_x-data_output[0][:fc_length])**2

fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Resorvoir Sparse rho=0.9 spar=0.75 NN: Pseudo-Insample Forecast")
ax1.plot(fc_spat111_x)
ax1.set_xlabel("x-forecast")
ax2.plot(data_output[0][:fc_length])
ax2.set_xlabel("Lorenz x-axis")
ax3.plot(mse_fc_spat111)
ax3.set_xlabel("mse")

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




