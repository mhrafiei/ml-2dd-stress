import random
import sklearn.neural_network
import numpy
import matplotlib.pyplot
import matplotlib
import math
import h5py
import random
import pickle
import glob
import ntpath 
import joblib
import os
import scipy.io
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, Dense, Dropout, LSTM, PReLU
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from keras import regularizers

sc = StandardScaler()

#$ INPUTS
epochs           = 50
batch_size       = 128
verbose          = 1
shuffle          = True
validation_split = 0.01
dropout          = 0.0
reg_rate         = 0.00 # regularization rate 
optimizer        = 'adam'
activation_1st   = 'relu'
activation_hid   = 'relu'
activation_lst   = 'linear'
loss             = 'mean_squared_error'
plot_train_cond  = True
plot_test_cond   = True
standard_val     = False
lb               = 0
ub               = +1

#directories and files
dir_current      = os. getcwd()
dir_parent       = os.path.split(dir_current)[0]
dir_code_keras   = os.path.join(dir_parent,'code-keras')
dir_code_scikit  = os.path.join(dir_parent,'code-scikit')
dir_data_matlab  = os.path.join(dir_parent,'data-matlab')
dir_data_python  = os.path.join(dir_parent,'data-python')

file_data_python = os.path.join(dir_data_python,'data.npy')
file_ind_python  = os.path.join(dir_data_python,'ind.npy')
file_data_matlab = os.path.join(dir_data_matlab,'data_inou.mat')

############################

def fun_scale(x,lb,ub):
	return ((x-numpy.min(x,axis=0))/(numpy.max(x,axis=0)-numpy.min(x,axis=0)))*(ub-lb) + lb

class ClsFuns():
    """docstring for Cls_Funs"""

    statement = "thi is in cls"

    def __init__(self,i0,i1,lay_neuron):
        self.i0 = i0
        self.i1 = i1
        self.lay_neuron = lay_neuron

    #i0 = rtt
    #i1 = rrs


    def fun_run(self,datain,dataou,ind_train,ind_test):

        # requires: 
        i0 = self.i0
        i1 = self.i1
        lay_neuron = self.lay_neuron

        #normalization
        if standard_val:
            datain = sc.fit_transform(datain)
            dataou = sc.fit_transform(dataou)
            
        else:
            datain = fun_scale(datain,lb,ub)
            dataou = fun_scale(dataou,lb,ub)
        
        # get the indices 
        ind_train = ind_train[i0-1][i1-1]
        ind_test  = ind_test[i0-1][i1-1]
        
        # get the training data
        datain_train = datain[ind_train]
        dataou_train = dataou[ind_train]
        
        # get the testing data
        datain_test  = datain[ind_test]
        dataou_test  = dataou[ind_test]

        tail_val = ''
        for jj in range(len(lay_neuron)):
            tail_val = tail_val + str(lay_neuron[jj])+ '_'

        tail_val = tail_val[:-1]

        filename = "model_keras_{}_{}_{}.h5".format(i0,i1,tail_val)
        filename = os.path.join(dir_data_python,filename)

        filename_dir = glob.glob(filename)
        if len(filename_dir) == 0:
            print("Initiate new Sequential network from scratch")

            model = Sequential()
            model.add(Dense(lay_neuron[0], input_dim=datain_train.shape[1], activation=activation_1st))
            #model.add(Dense(lay_neuron[0], input_dim=datain_train.shape[1]))
            #model.add(PReLU(alpha_initializer="zeros",alpha_regularizer=None,alpha_constraint=None,shared_axes=None,))
            
            #model.add(Dropout(dropout))
            for l0 in lay_neuron[1:]:
            	print(l0)
            	model.add(Dense(l0, activation=activation_hid, kernel_regularizer=regularizers.l2(reg_rate)))
            	#model.add(Dense(l0, kernel_regularizer=regularizers.l2(reg_rate)))
            	#model.add(PReLU(alpha_initializer="zeros",alpha_regularizer=None,alpha_constraint=None,shared_axes=None,))
            	model.add(Dropout(dropout))

            model.add(Dense(dataou_train.shape[1], activation=activation_lst))
            #sgd = optimizers.SGD(lr=0.5, decay=1e-9, momentum=0.9, nesterov=True)
            model.compile(loss=loss, optimizer=optimizer)
        else:
        	print("Initiate Sequential transfer learning")
        	model = load_model(filename)

        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.002,
        #                      patience=5, min_lr=0.000000000000000001)
        
        # train

        print("-------------------------------")
        print("TR-LEARN-KERAS | rrt {} | rrs {} | Net {}".format(i0,i1,tail_val))
        print("-------------------------------")

        history       = model.fit(datain_train, dataou_train, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=shuffle, validation_split=validation_split) #callbacks=[reduce_lr]
        # save the model to disk
        #pickle.dump(my_neural_network, open(filename, 'wb'))
        model.save(filename)

        loss_vals     = history.history['loss'] 
        val_loss_vals = history.history['val_loss'] 
        
        
        filename = "model_keras_loss_{}_{}_{}.txt".format(i0,i1,tail_val)
        filename = os.path.join(dir_data_python,filename)
        
        try:
        	f = open(filename,'x')
        except:
        	f = open(filename,'w')

        f.write("{}".format(loss_vals))
        f.close()
        
        filename = "model_keras_val_loss_{}_{}_{}.txt".format(i0,i1,tail_val)
        filename = os.path.join(dir_data_python,filename)
        try:
            f = open(filename,'x')
        except:
            f = open(filename,'w')

        f.write("{}".format(val_loss_vals))
        f.close()

        print("-------------------------------")
        print("TE-LEARN-KERAS | rrt {} | rrs {} | Net {}".format(i0,i1,tail_val))
        print("-------------------------------")

        # estimate train and test data
        dataes_train = model.predict(datain_train)
        dataes_test  = model.predict(datain_test)


        trou = numpy.reshape(dataou_train,(dataou_train.shape[0]*dataou_train.shape[1],1))
        tres = numpy.reshape(dataes_train,(dataes_train.shape[0]*dataes_train.shape[1],1))
        teou = numpy.reshape(dataou_test ,(dataou_test.shape[0] *dataou_test.shape[1],1)) 
        tees = numpy.reshape(dataes_test ,(dataes_test.shape[0] *dataes_test.shape[1],1))

        filename = 'trou_{}_{}_{}.csv'.format(i0,i1,tail_val)
        filename = os.path.join(dir_data_python,filename)
        numpy.savetxt(filename, trou, delimiter=",")

        filename = 'tres_{}_{}_{}.csv'.format(i0,i1,tail_val)
        filename = os.path.join(dir_data_python,filename)
        numpy.savetxt(filename, tres, delimiter=",")

        filename = 'teou_{}_{}_{}.csv'.format(i0,i1,tail_val)
        filename = os.path.join(dir_data_python,filename)
        numpy.savetxt(filename, teou, delimiter=",")       

        filename = 'tees_{}_{}_{}.csv'.format(i0,i1,tail_val)
        filename = os.path.join(dir_data_python,filename)
        numpy.savetxt(filename, tees, delimiter=",")
 
        if plot_train_cond:
            print("TR-PLOT-KERAS  | rrt {} | rrs {} | Net {}".format(i0,i1,tail_val))
    
            matplotlib.pyplot.figure(figsize=[10,10])
            matplotlib.rc('xtick', labelsize=20)
            matplotlib.rc('ytick', labelsize=20) 
            matplotlib.rc('font',family='Times New Roman')
            matplotlib.pyplot.plot(trou, tres,'.',markersize=1)
            if standard_val:
                lb_val = -2.5
                ub_val = +2.5
            else:
            	lb_val = lb
            	ub_val = ub

            matplotlib.pyplot.plot([lb_val,ub_val],[lb_val,ub_val],'-g')
            matplotlib.pyplot.xlabel('Real', fontsize=20, fontname='Times New Roman')
            matplotlib.pyplot.ylabel('Estimated', fontsize=20, fontname='Times New Roman')
            matplotlib.pyplot.title('Train | RTT = {} | RRS = {} | Net_{}'.format(i0,i1,tail_val), fontsize=20, fontname='Times New Roman')
            
            filename = 'tr_keras_{}_{}_{}.png'.format(i0,i1,tail_val)
            filename = os.path.join(dir_data_python,filename)
            matplotlib.pyplot.savefig(filename, dpi=300)  
        
        if plot_test_cond:
            print("TE-PLOT-KERAS  | rrt {} | rrs {} | Net {}".format(i0,i1,tail_val))
            matplotlib.pyplot.figure(figsize=[10,10])
            matplotlib.rc('xtick', labelsize=20)
            matplotlib.rc('ytick', labelsize=20) 
            matplotlib.rc('font',family='Times New Roman')
            matplotlib.pyplot.plot(teou, tees,'.',markersize=1)
            if standard_val:
                lb_val = -2.5
                ub_val = +2.5
            else:
            	lb_val = lb
            	ub_val = ub

            matplotlib.pyplot.plot([lb_val,ub_val],[lb_val,ub_val],'-g')
            matplotlib.pyplot.xlabel('Real', fontsize=20, fontname='Times New Roman')
            matplotlib.pyplot.ylabel('Estimated', fontsize=20, fontname='Times New Roman')
            matplotlib.pyplot.title('Test | RTT = {} | RRS = {} | Net_{}'.format(i0,i1,tail_val), fontsize=20, fontname='Times New Roman')
            
            filename = 'te_keras_{}_{}_{}.png'.format(i0,i1,tail_val)
            filename = os.path.join(dir_data_python,filename)
            matplotlib.pyplot.savefig(filename, dpi=300)  
        
    def fun_losscurve(self):
        i0 = self.i0
        i1 = self.i1
        lay_neuron = self.lay_neuron

        tail_val = ''
        for jj in range(len(lay_neuron)):
            tail_val = tail_val + str(lay_neuron[jj])+ '_'

        tail_val = tail_val[:-1]


        filename1     = "model_keras_loss_{}_{}_{}.txt".format(i0,i1,tail_val)
        filename1     = os.path.join(dir_data_python,filename1)
        
        filename2     = "model_keras_val_loss_{}_{}_{}.txt".format(i0,i1,tail_val)
        filename2     = os.path.join(dir_data_python,filename2)
        
        f1            = open(filename1,'r')
        f2            = open(filename2,'r')
        
        f1_val = eval(f1.read())
        f2_val = eval(f2.read())

        a1 = range(1,len(f1_val)+1)
        a1 = numpy.array(a1)
        b1 = numpy.log10(numpy.array(f1_val))

        a2 = range(1,len(f2_val)+1)
        a2 = numpy.array(a2)
        b2 = numpy.log10(numpy.array(f2_val))

        f1.close()
        f2.close()

        print("LS-PLOT-KERAS  | rrt {} | rrs {} | Net {}".format(i0,i1,tail_val))
        
        matplotlib.pyplot.figure()
        matplotlib.pyplot.figure(figsize=[10,10])
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20) 
        matplotlib.rc('font',family='Times New Roman')
        matplotlib.pyplot.plot(a1, b1,'-b',markersize=1)
        matplotlib.pyplot.plot(a2, b2,'-r',markersize=1)
        matplotlib.pyplot.legend(['loss', 'val_loss'])
        matplotlib.pyplot.xlabel('Iterations', fontsize=20, fontname='Times New Roman')
        matplotlib.pyplot.ylabel('Natural Logarithm of Mean Squared Error (MSE)', fontsize=20, fontname='Times New Roman')
        matplotlib.pyplot.title('Train_Progress | RTT = {} | RRS = {} | Net_{}'.format(i0,i1,tail_val), fontsize=20, fontname='Times New Roman')
        
        filename = 'ls_keras_{}_{}_{}.png'.format(i0,i1,tail_val)
        filename = os.path.join(dir_data_python,filename)
        matplotlib.pyplot.savefig(filename, dpi=300) 

        filename     = "vall_{}_{}_{}.csv".format(i0,i1,tail_val)
        filename     = os.path.join(dir_data_python,filename)
        numpy.savetxt(filename,f1_val, delimiter=",")
