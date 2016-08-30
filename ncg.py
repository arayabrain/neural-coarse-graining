import numpy as np
from math import *

import pickle
import theano
import theano.tensor as T

from lasagne.utils import floatX

import lasagne
from lasagne.layers import batch_norm
import sys
import os

import argparse

# Stuff to load/save models
def read_model(lastlayers, filename):
	lasagne.layers.set_all_param_values(lastlayers, pickle.load(open(filename,"rb")))

def write_model(lastlayers, filename):
	pickle.dump( lasagne.layers.get_all_param_values(lastlayers), open(filename,"wb"))

# We'll need this distributed softmax so that each point in the convolution output gets its own softmax over channels
# The exponential is capped above and below to avoid NaNs when the network is very sure of a particular class
def convSoftmax(x):
	emx = T.exp(T.maximum(T.minimum(x,20),-20))
	return emx/(1e-8 + T.sum(emx,axis=1,keepdims=True))

def leakyReLU(x):
	return T.maximum(x,0.05*x)

parser = argparse.ArgumentParser(description='Commandline arguments for Neural Coarse-Graining reference code')
parser.add_argument('--input', required=True, type=str,
                   help='Space-separated text file containing input signal')
parser.add_argument('--output', required=False, type=str, 
                   help='Output file containing the coarse-grained categorical values')
parser.add_argument('--preds', required=False, type=str, 
                   help='Output file containing the predictions of the coarse-grained categorical values')
parser.add_argument('--valid', required=False, type=str,
                   help='Space-separated text file containing a validation signal')
parser.add_argument('--error', required=False, type=str, help='File to output the train/validation error curves during training')
parser.add_argument('--process', action='store_true', help='Run a trained model on the input, but don\'t update the weights')
parser.add_argument('--classes', required=False, type=int, default=3,
                   help='Number of coarse-grained categorical variables')
parser.add_argument('--dt', required=False,  type=int, default=10, help='Time delta over which to predict')
parser.add_argument('--save_model', required=False, type=str, help='File to save the trained model in')
parser.add_argument('--load_model', required=False, type=str, help='File to load a trained model from')
parser.add_argument('--epochs', required=False, default=100, type=int, help='Number of training epochs')
parser.add_argument('--lr',required=False, default=1e-3, type=float, help='Learning rate')
parser.add_argument('--tr_filt1',required=False,default=20,type=int, help='Number of filters in the first transformer layer')
parser.add_argument('--tr_filt2',required=False,default=20,type=int, help='Number of filters in the second transformer layer')
parser.add_argument('--pr_filt1',required=False,default=20,type=int, help='Number of filters in the first predictor layer')
parser.add_argument('--pr_filt2',required=False,default=20,type=int, help='Number of filters in the second predictor layer')
parser.add_argument('--tr_fs1',required=False,default=3,type=int, help='Filter size of the first transformer layer')
parser.add_argument('--tr_fs2',required=False,default=3,type=int, help='Filter size of the second transformer layer')
parser.add_argument('--pr_fs1',required=False,default=3,type=int, help='Filter size of the first predictor layer')
parser.add_argument('--pr_fs2',required=False,default=3,type=int, help='Filter size of the second predictor layer')

args = parser.parse_args()

DISTANCE = args.dt
CLASSES = args.classes

# Load in the data, make sure it has the shape (signal length, channels)
data = np.loadtxt(args.input)
if data.ndim<2:
	data = data.reshape((data.shape[0],1))
	
if args.valid:
	test = np.loadtxt(args.valid)
	if test.ndim<2:
		test = test.reshape((test.shape[0],1))

invar1 = T.tensor3('presignal')

def build_model():
	net = {}
	
	net["input1"] = lasagne.layers.InputLayer(shape=(None,data.shape[1],None), input_var = invar1 )
	
	# Transformer part of the network - in-place convolution to transform to the new coarse-grained classes
	net["transform1"] = batch_norm(lasagne.layers.Conv1DLayer(incoming=net["input1"], num_filters=args.tr_filt1, filter_size=args.tr_fs1, pad="same", nonlinearity=leakyReLU, W = lasagne.init.GlorotUniform(gain='relu')))
	net["transform2"] = batch_norm(lasagne.layers.Conv1DLayer(incoming=net["transform1"], num_filters=args.tr_filt2, filter_size=args.tr_fs2, pad="same", nonlinearity=leakyReLU, W = lasagne.init.GlorotUniform(gain='relu')))
	net["transform3"] = (lasagne.layers.Conv1DLayer(incoming=net["transform2"], num_filters=CLASSES, filter_size=1, pad="same", nonlinearity=convSoftmax, W = lasagne.init.GlorotUniform(gain='relu')))
	
	# Predictor part. Take the coarse-grained classes and predict them at an offset of DISTANCE
	net["predictor1"] = batch_norm(lasagne.layers.Conv1DLayer(incoming = net["transform3"], num_filters = args.pr_filt1, filter_size=args.pr_fs1, pad="same", nonlinearity = leakyReLU, W = lasagne.init.GlorotUniform(gain='relu')))
	net["predictor2"] = batch_norm(lasagne.layers.Conv1DLayer(incoming = net["predictor1"], num_filters = args.pr_filt2, filter_size=args.pr_fs2, pad="same", nonlinearity = leakyReLU, W = lasagne.init.GlorotUniform(gain='relu')))
	net["predictor3"] = (lasagne.layers.Conv1DLayer(incoming = net["predictor2"], num_filters = CLASSES, filter_size=1, pad="same", nonlinearity = convSoftmax, W = lasagne.init.GlorotUniform(gain='relu')))

	return net

net = build_model()

if args.load_model:
	read_model(net["predictor3"], args.load_model)

# We want both the transformed data and the predictions
output,transform = lasagne.layers.get_output( (net["predictor3"], net["transform3"]) )
params = lasagne.layers.get_all_params( (net["predictor3"]), trainable = True)

# This gives us the offset transformed signal, that we want the predictor to output. Clip the ends by DISTANCE to avoid the overflow
rtransform = T.roll(transform,-DISTANCE,axis=2)[:,:,DISTANCE:-DISTANCE]
routput = output[:,:,DISTANCE:-DISTANCE]

# Entropy term measures the entropy of the average transformed signal. We want to make this large
entropy = -1 * (rtransform.mean(axis = (0,2)) * T.log(rtransform.mean(axis=(0,2))+1e-6)).sum()

# Info term measures the error of the predictor (standard cross-entropy error between the prediction and true distribution). We want to make this small
info = -1 * ((rtransform * T.log( routput + 1e-6 )).mean(axis=(0,2))).sum()

# Combined loss function
loss = info - entropy

lr = args.lr

updates = lasagne.updates.adam(loss, params, learning_rate = lr)
train_fn = theano.function([invar1], [info,entropy], updates=updates, allow_input_downcast = True)
test_fn = theano.function([invar1], [info,entropy], allow_input_downcast = True)

process = theano.function([invar1], (output, transform), allow_input_downcast = True)

if (not args.process):
	for epoch in range(args.epochs):		
		tr_i, tr_e = train_fn(data.transpose((1,0)).reshape((1,data.shape[1],data.shape[0])))
		train_error = tr_i - tr_e
		
		if (args.valid):
			ts_i, ts_e = test_fn(test.transpose((1,0)).reshape((1,data.shape[1],test.shape[0])))
			test_error = ts_i - ts_e
			
		if (args.valid):
			print "Epoch %d, Train %.6g, Valid %.6f" % (epoch, train_error, test_error)
		else:
			print "Epoch %d, Train %.6g" % (epoch, train_error)
	
		if (args.error):
			f = open(args.error,"a")
			if args.valid:
				f.write("{} {} {} {} {} {} {}\n".format(epoch, train_error, tr_i, -tr_e, test_error, ts_i, -ts_e))
			else:
				f.write("{} {} {} {}\n".format(epoch, train_error, tr_i, -tr_e))
			f.close()
	
prediction, result = process( data.transpose((1,0)).reshape((1,data.shape[1],data.shape[0]) ))

if (args.output):
	np.savetxt(args.output, result[0].T)
	
if (args.preds):
	np.savetxt(args.preds, prediction[0].T)
			
if args.save_model:
	write_model(net["predictor3"], args.save_model)
