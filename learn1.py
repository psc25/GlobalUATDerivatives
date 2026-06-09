import numpy as np
import tensorflow as tf
import esig
import matplotlib.pyplot as plt
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(3)

pltname = "learn1"

M = 50000
K = 101
d = 1
T = 1.0
dt = T/K

alpha = 0.4
beta = 0.01
gamma = 1.0

dX = np.random.normal(size = [M, K-1, 1]).astype(np.float32)*np.sqrt(dt)
X = np.zeros([M, K, 1], dtype = np.float32)
X[:, 1:, :] = np.cumsum(dX, axis = 1)
tt = np.linspace(0, T, K, dtype = np.float32)
time_var = np.tile(np.reshape(tt, (1, -1, 1)), [M, 1, 1])

Y = X*np.cumsum(X*dt, axis = 1)
DhY = np.square(X)
DvY = np.cumsum(X*dt, axis = 1)

# Compute Weight
X1 = np.transpose(X, [0, 2, 1])
time_var1 = np.transpose(time_var, [0, 2, 1])
hoelder = np.nanmax(np.abs(X-X1)/np.power(np.abs(time_var-time_var1), alpha), axis = (1, 2))
wght_PNN = np.reshape(np.exp(beta*np.power(X[:, 0, 0] + hoelder, gamma)), [-1, 1, 1])

ep = 5000
eval_every = 200
val_split = 0.2
Mtrain = int((1-val_split)*M)
lr = 5e-5
batch_size = 1000

ind_train = np.arange(Mtrain)
ind_test = np.arange(Mtrain, M)
ind_plot = np.random.choice(ind_test, 3, replace = False)
col = ['b', 'g', 'r']

fig = plt.figure()
for i in range(len(ind_plot)):
    plt.plot(tt, X[ind_plot[i]], c = col[i], ls = ":", alpha = 0.6)
    plt.plot(tt, Y[ind_plot[i]], c = col[i], ls = "-", alpha = 0.6)
    
plt.show()

fig = plt.figure()
for i in range(len(ind_plot)):
    plt.plot(tt, X[ind_plot[i]], c = col[i], ls = ":", alpha = 0.6)
    plt.plot(tt, DhY[ind_plot[i]], c = col[i], ls = "-", alpha = 0.6)
    
plt.show()

fig = plt.figure()
for i in range(len(ind_plot)):
    plt.plot(tt, X[ind_plot[i]], c = col[i], ls = ":", alpha = 0.6)
    plt.plot(tt, DvY[ind_plot[i]], c = col[i], ls = "-", alpha = 0.6)
    
plt.show()

# Learn PNN
N = 30
N1 = 20
init = tf.random_normal_initializer(mean = 0.0, stddev = 0.001)
inp_time = tf.placeholder(shape = (None, K, 1), dtype = tf.float32)
inp_path = tf.placeholder(shape = (None, K, 1), dtype = tf.float32)
y_true_tf = tf.placeholder(shape = (None, K, 1), dtype = tf.float32)
Dh_true_tf = tf.placeholder(shape = (None, K, 1), dtype = tf.float32)
Dv_true_tf = tf.placeholder(shape = (None, K, d), dtype = tf.float32)
wght_b_tf = tf.placeholder(shape = (None, 1, 1), dtype = tf.float32)

W = tf.Variable(initial_value = init(shape = [1, 1, N]), dtype = tf.float32)
A = tf.Variable(initial_value = init(shape = [1, 1, N]), dtype = tf.float32)
B = tf.Variable(initial_value = init(shape = [1, 1, N, d]), dtype = tf.float32)
b = tf.Variable(initial_value = init(shape = [1, 1, N]), dtype = tf.float32)
V = tf.Variable(initial_value = init([1, 1, N, d, N1]), dtype = tf.float32)
U = tf.Variable(initial_value = init([1, 1, N, d, N1]), dtype = tf.float32)
c = tf.Variable(initial_value = init([1, 1, N, d, N1]), dtype = tf.float32)

act_der = lambda x: tf.nn.sigmoid(x)*(1.0-tf.nn.sigmoid(x))

W1_hidden = V*tf.expand_dims(tf.expand_dims(inp_time, axis = -1), axis = -1) + c
W1_output = tf.reduce_sum(U*tf.nn.relu(W1_hidden), axis = -1)
W1_intgnd = W1_output*tf.expand_dims(inp_path, axis = 2)
inp_int = tf.cumsum(W1_intgnd*dt, axis = 1)
hidden1 = A*inp_time + tf.reduce_sum(B*tf.expand_dims(inp_path, axis = 2), axis = -1) + tf.reduce_sum(inp_int, -1) + b
y_pred = tf.reduce_sum(W*tf.nn.sigmoid(hidden1), -1, keepdims = True)
y_pred = y_pred - y_pred[:, 0:1]
Dh_pred = tf.reduce_sum(W*act_der(hidden1)*(W1_intgnd[:, :, :, 0] + A), -1, keepdims = True)
Dv_pred = tf.reduce_sum(tf.expand_dims(W*act_der(hidden1), axis = -1)*B, axis = 2)

loss1 = tf.reduce_mean(tf.square((y_pred - y_true_tf)/wght_b_tf))
loss2 = tf.reduce_mean(tf.square((Dh_pred - Dh_true_tf)/wght_b_tf))
loss3 = tf.reduce_mean(tf.square((Dv_pred - Dv_true_tf)/wght_b_tf))
loss = loss1 + 0.5*loss2 + 0.5*loss3/d

global_step = tf.Variable(0, trainable = False)
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

b_PNN = time.time()
loss_PNN = np.nan*np.ones([ep, 2])
print("\nTraining steps:")
for i in range(ep):
    begin = time.time()
    nr_batch = int(Mtrain/batch_size)
    ind_rand = np.random.permutation(Mtrain)
    loss_batch = np.zeros([nr_batch, 1])
    for l in range(nr_batch):
        feed_dict = {inp_time: time_var[ind_rand[l:l+batch_size]], inp_path: X[ind_rand[l:l+batch_size]],
                     y_true_tf: Y[ind_rand[l:l+batch_size]], Dh_true_tf: DhY[ind_rand[l:l+batch_size]],
                     Dv_true_tf: DvY[ind_rand[l:l+batch_size]], wght_b_tf: wght_PNN[ind_rand[l:l+batch_size]]}
        _, loss_batch[l] = sess.run([train_op, loss], feed_dict)
        
    loss_PNN[i, 0] = np.mean(loss_batch)
    end = time.time()
    print("Step {}, Time {}s, Loss {:g}".format(i+1, round(end-begin, 1), loss_PNN[i, 0]))
    
    if val_split > 0 and (i+1) % eval_every == 0:
        begin = time.time()
        print("\nEvaluation on test data:")
        nr_batch = int((M-Mtrain)/batch_size)
        ind_rand = np.random.permutation(np.arange(Mtrain, M))
        loss_batch = np.zeros([nr_batch, 1])
        for l in range(nr_batch):
            feed_dict = {inp_time: time_var[ind_rand[l:l+batch_size]], inp_path: X[ind_rand[l:l+batch_size]],
                         y_true_tf: Y[ind_rand[l:l+batch_size]], Dh_true_tf: DhY[ind_rand[l:l+batch_size]],
                         Dv_true_tf: DvY[ind_rand[l:l+batch_size]], wght_b_tf: wght_PNN[ind_rand[l:l+batch_size]]}
            loss_batch[l] = sess.run(loss, feed_dict)
            
        loss_PNN[i, 1] = np.mean(loss_batch)
        end = time.time()
        print("Step {}, Time {}s, Loss {:g}".format(i+1, round(end-begin, 1), loss_PNN[i, 1]))
        print("")
        
feed_dict = {inp_time: time_var[ind_plot], inp_path: X[ind_plot]}
Y_PNN, DhY_PNN, DvY_PNN = sess.run([y_pred, Dh_pred, Dv_pred], feed_dict)
e_PNN = time.time()

# Learn Signature
b_Sig = time.time()
N = 6
N1 = np.power(2, N+1) - 1
tt1 = np.reshape(tt, (-1, 1))
Xhat = np.zeros([M, K, 2], dtype = np.float32)
Sig0 = np.zeros([M, K, 3], dtype = np.float32)
Sig = np.zeros([M, K, N1], dtype = np.float32)
h = 1e-5
lr = 5e-5
for m in range(M):
    print("Compute signature m = " + str(m+1) + "/" + str(M))
    for k in range(K):
        if k != 49 and k != 98: # there is an issue with esig
            Sig0[m, k] = esig.stream2sig(X[m, :(k+1)], 2)

Sig0[:, 49] = 0.5*(Sig0[:, 48]+Sig0[:, 50])
Sig0[:, 98] = 0.5*(Sig0[:, 97]+Sig0[:, 99])

for m in range(M):
    print("Compute time-extended signature m = " + str(m+1) + "/" + str(M))
    Xhat[m] = np.concatenate((tt1, X[m]), axis = -1)
    for k in range(K):
        if k != 49 and k != 98: # there is an issue with esig
            Sig[m, k] = esig.stream2sig(Xhat[m, :(k+1)], N)
        
Sig[:, 49] = 0.5*(Sig[:, 48]+Sig[:, 50])
Sig[:, 98] = 0.5*(Sig[:, 97]+Sig[:, 99])
    
Dh_Sig = np.zeros([M, K, N1], dtype = np.float32)
Dv_Sig = np.zeros([M, K, N1], dtype = np.float32)
for n in range(1, N+1):
    print("Compute derivatives n = " + str(n) + "/" + str(N))
    ind1 = np.arange(np.power(2, n-1)-1, np.power(2, n)-1)
    ind2 = np.arange(np.power(2, n)-1, np.power(2, n+1)-1, 2)
    Dh_Sig[:, :, ind2] = Sig[:, :, ind1]
    Dv_Sig[:, :, ind2+1] = Sig[:, :, ind1]
    
# Compute Weight
time_var1 = np.transpose(time_var, [0, 2, 1])
log_Sig = Sig0
log_Sig[:, :, 2] = log_Sig[:, :, 2] - 0.5*np.square(log_Sig[:, :, 1])
log_Sig1 = np.expand_dims(log_Sig, axis = 2)
log_Sig2 = np.transpose(log_Sig1, [0, 2, 1, 3])
hoelder = np.nanmax(np.linalg.norm(log_Sig1-log_Sig2, axis = -1)/np.power(np.abs(time_var-time_var1), alpha), axis = (1, 2))
wght_Sig = np.reshape(np.exp(beta*np.power(hoelder, gamma)), [-1, 1, 1])

init = tf.random_normal_initializer(mean = 0.0, stddev = 0.001)
inp_Sig = tf.placeholder(shape = (None, K, N1), dtype = tf.float32)
inp_Dh_Sig = tf.placeholder(shape = (None, K, N1), dtype = tf.float32)
inp_Dv_Sig = tf.placeholder(shape = (None, K, N1), dtype = tf.float32)
y_true_tf = tf.placeholder(shape = (None, K, 1), dtype = tf.float32)
wght_b_tf = tf.placeholder(shape = (None, 1, 1), dtype = tf.float32)

L = tf.Variable(initial_value = init(shape = [1, 1, N1]), dtype = tf.float32)
y_pred = tf.reduce_sum(L*inp_Sig, axis = -1, keepdims = True)
Dh_pred = tf.reduce_sum(L*inp_Dh_Sig, axis = -1, keepdims = True)
Dv_pred = tf.reduce_sum(L*inp_Dv_Sig, axis = -1, keepdims = True)

loss1 = tf.reduce_mean(tf.square((y_pred - y_true_tf)/wght_b_tf))
loss2 = tf.reduce_mean(tf.square((Dh_pred - Dh_true_tf)/wght_b_tf))
loss3 = tf.reduce_mean(tf.square((Dv_pred - Dv_true_tf)/wght_b_tf))
loss = loss1 + 0.5*loss2 + 0.5*loss3/d
loss = tf.reduce_mean(tf.square((y_pred - y_true_tf)/wght_b_tf))

global_step = tf.Variable(0, trainable = False)
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

b_Sig = time.time()
loss_Sig = np.nan*np.ones([ep, 2])
print("\nTraining steps:")
for i in range(ep):
    begin = time.time()
    nr_batch = int(Mtrain/batch_size)
    ind_rand = np.random.permutation(Mtrain)
    loss_batch = np.zeros([nr_batch, 1])
    for l in range(nr_batch):
        feed_dict = {inp_Sig: Sig[ind_rand[l:l+batch_size]], inp_Dh_Sig: Dh_Sig[ind_rand[l:l+batch_size]], inp_Dv_Sig: Dv_Sig[ind_rand[l:l+batch_size]], 
                     y_true_tf: Y[ind_rand[l:l+batch_size]], wght_b_tf: wght_Sig[ind_rand[l:l+batch_size]]}
        _, loss_batch[l] = sess.run([train_op, loss], feed_dict)
        
    loss_Sig[i, 0] = np.mean(loss_batch)
    end = time.time()
    print("Step {}, Time {}s, Loss {:g}".format(i+1, round(end-begin, 1), loss_Sig[i, 0]))
    
    if val_split > 0 and (i+1) % eval_every == 0:
        begin = time.time()
        print("\nEvaluation on test data:")
        nr_batch = int((M-Mtrain)/batch_size)
        ind_rand = np.random.permutation(np.arange(Mtrain, M))
        loss_batch = np.zeros([nr_batch, 1])
        for l in range(nr_batch):
            feed_dict = {inp_Sig: Sig[ind_rand[l:l+batch_size]], inp_Dh_Sig: Dh_Sig[ind_rand[l:l+batch_size]], inp_Dv_Sig: Dv_Sig[ind_rand[l:l+batch_size]], 
                         y_true_tf: Y[ind_rand[l:l+batch_size]], wght_b_tf: wght_Sig[ind_rand[l:l+batch_size]]}
            loss_batch[l] = sess.run(loss, feed_dict)
            
        loss_Sig[i, 1] = np.mean(loss_batch)
        end = time.time()
        print("Step {}, Time {}s, Loss {:g}".format(i+1, round(end-begin, 1), loss_Sig[i, 1]))
        print("")
        
feed_dict = {inp_Sig: Sig[ind_plot], inp_Dh_Sig: Dh_Sig[ind_plot], inp_Dv_Sig: Dv_Sig[ind_plot]}
Y_Sig, DhY_Sig, DvY_Sig = sess.run([y_pred, Dh_pred, Dv_pred], feed_dict)
e_Sig = time.time()

# Plot Results
fig = plt.figure()
for i in range(len(ind_plot)):
    plt.plot(tt, X[ind_plot[i]], c = col[i], ls = ":", alpha = 0.6)
    plt.plot(tt, Y[ind_plot[i]], c = col[i], ls = "-", alpha = 0.6)
    plt.plot(tt, Y_PNN[i], c = col[i], ls = "--", alpha = 0.6)
    plt.plot(tt, Y_Sig[i], c = col[i], ls = "-.", alpha = 0.6)
    
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = ":", label = r'$x(t)$')
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = "-", label = r'$f_1(t,x)$')
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = "--", label = 'PNN')
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = "-.", label = 'Sig')
plt.legend(loc = "upper left", ncol = 4)
plt.xlabel("Time")
plt.savefig(pltname + "_result.png", bbox_inches = 'tight', dpi = 400)
plt.close(fig)

fig = plt.figure()
for i in range(len(ind_plot)):
    plt.plot(tt, X[ind_plot[i]], c = col[i], ls = ":", alpha = 0.6)
    plt.plot(tt, DhY[ind_plot[i]], c = col[i], ls = "-", alpha = 0.6)
    plt.plot(tt, DhY_PNN[i], c = col[i], ls = "--", alpha = 0.6)
    plt.plot(tt, DhY_Sig[i], c = col[i], ls = "-.", alpha = 0.6)
    
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = ":", label = r'$x(t)$')
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = "-", label = r'$\mathcal{D} f_1(t,x)$')
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = "--", label = 'PNN')
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = "-.", label = 'Sig')
plt.legend(loc = "upper left", ncol = 4)
plt.xlabel("Time")
plt.savefig(pltname + "_result_Dh.png", bbox_inches = 'tight', dpi = 400)
plt.close(fig)

fig = plt.figure()
for i in range(len(ind_plot)):
    plt.plot(tt, X[ind_plot[i]], c = col[i], ls = ":", alpha = 0.6)
    plt.plot(tt, DvY[ind_plot[i]], c = col[i], ls = "-", alpha = 0.6)
    plt.plot(tt, DvY_PNN[i], c = col[i], ls = "--", alpha = 0.6)
    plt.plot(tt, DvY_Sig[i], c = col[i], ls = "-.", alpha = 0.6)
    
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = ":", label = r'$x(t)$')
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = "-", label = r'$\mathscr{D} f_1(t,x)$')
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = "--", label = 'PNN')
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = "-.", label = 'Sig')
plt.legend(loc = "upper left", ncol = 4)
plt.xlabel("Time")
plt.savefig(pltname + "_result_Dv.png", bbox_inches = 'tight', dpi = 400)
plt.close(fig)

# Plot Loss
fig = plt.figure()
res2 = np.nan*np.ones(ep)
res2[np.arange(eval_every-1, ep, eval_every)] = 1.0
plt.plot(range(ep), loss_PNN[:, 0], c = "silver", ls = "-", label = "PNN Train")
plt.plot(range(ep), loss_PNN[:, 1], ls = "None", marker = "o", markerfacecolor = "None", markeredgecolor = "k", label = "PNN Test")
plt.plot(range(ep), loss_Sig[:, 0], c = "dimgray", ls = "-", label = "Sig Train")
plt.plot(range(ep), loss_Sig[:, 1], ls = "None", marker = "x", markerfacecolor = "None", markeredgecolor = "k", label = "Sig Test")
plt.legend(loc = "upper right", ncol = 2)
plt.xlabel("Epochs")
plt.ylabel("Weighted MSE")
plt.savefig(pltname + "_training.png", bbox_inches = 'tight', dpi = 400) 
plt.close(fig)

# Print Times
print("Time for PNN: " + str(np.round(e_PNN-b_PNN, 2)))
print("Time for Sig: " + str(np.round(e_Sig-b_Sig, 2)))