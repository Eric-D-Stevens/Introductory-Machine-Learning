import numpy as np
import random as rand
import matplotlib.pyplot as plt


def load_matrix(filename):

	# load data into X matrix form file
	data = np.genfromtxt(filename)
	X = np.matrix(data)

	# create Y matrix form X
	Y = X.copy()
	Y = Y[:,13]

	# place one
	X[:,13] = 1

	# return the matricies
	return [X,Y]


def weight_vector(X,Y):
	
	# W =(X.T*X)^-1 * (X.T*Y)
	A = (X.T*X).I
	B = X.T*Y
	W = A*B
	
	return W

def no_dummy(X):
	#Xc = X.copy()
	X=X[:,:13]
	return X

def sum_sq_err(W,X,Y):
	A = (Y-X*W).T
	B = Y-(X*W)
	E = A*B
	return E[0,0]

def add_feature(X,a):
	rows = len(X)
	feat = np.matrix(np.random.rand(rows,1))
	feat = feat*a
	X = np.append(X,feat,axis=1 )
	return X

def quick_sse(X,Y):
	W = weight_vector(X,Y)
	E = sum_sq_err(W,X,Y)
	return [W,E]

def feature_relationships(numOfFeats):

	[X,Y] = load_matrix("housing_train.txt")
	[Xt,Yt] = load_matrix("housing_test.txt")
	W = weight_vector(X,Y)
	SSE = sum_sq_err(W,X,Y)
	SSEt = sum_sq_err(W,Xt,Yt)
	
	num_feat = []
	List_SSE = []
	List_SSEt = []
	
	num_feat.append(0)
	List_SSE.append(SSE)
	List_SSEt.append(SSEt)

	a = 1

	for feats in range(1,numOfFeats+1):
		
		a=a*(rand.random()*10)
		X=add_feature(X,a)
		Xt=add_feature(Xt,a)

		W = weight_vector(X,Y)
		SSE = sum_sq_err(W,X,Y)
		SSEt = sum_sq_err(W,Xt,Yt)
		
		num_feat.append(feats)
		List_SSE.append(SSE)
		List_SSEt.append(SSEt)
		print X.shape
	return [num_feat, List_SSE, List_SSEt]

def plot_relationship(num_feat, List_SSE, List_SSEt, xaxis):

	plt.figure(1)
	plt.subplot(211)
	plt.ylabel("SSE")
	plt.plot(num_feat, List_SSE, 'ro')
	plt.title("Training Data")
	plt.subplot(212)
	plt.xlabel(xaxis)
	plt.ylabel("SSE")
	plt.plot(num_feat, List_SSEt, 'bo')
	plt.title("Testing Data")
	plt.show()
	return 0				

def plot_lambda_relationship(lambda_list, List_SSE, List_SSEt):

	plt.figure(1)
	plt.subplot(211)
	plt.ylabel("SSE")
	plt.plot(num_feat, List_SSE, 'ro')
	plt.title("Training Data")
	plt.subplot(212)
	plt.xlabel("Value of lambda")
	plt.ylabel("SSE")
	plt.plot(num_feat, List_SSEt, 'bo')
	plt.title("Testing Data")
	plt.show()
	return 0				




def weight_vector_variant(X,Y,lamb):
	
	# W = (Xt*X + lamb*I)^-1 * XtY
	XtX = X.T*X
	sz = len(XtX)
	I=np.matrix(np.identity(sz))
	A = (XtX + lamb*I).I
	B =  X.T*Y
	W = A*B
	
	return W

def lambda_relationships(lambda_list):

	[X,Y] = load_matrix("housing_train.txt")
	[Xt,Yt] = load_matrix("housing_test.txt")
	
	List_SSE = []
	List_SSEt = []

	for lam in lambda_list:
		
		W = weight_vector_variant(X,Y,lam)
		
		SSE = sum_sq_err(W,X,Y)
		SSEt = sum_sq_err(W,Xt,Yt)
	
		List_SSE.append(SSE)
		List_SSEt.append(SSEt)
	return [lambda_list, List_SSE, List_SSEt]



def printMatrixE(a):
   print "Matrix["+("%d" %a.shape[0])+"]["+("%d" %a.shape[1])+"]"
   rows = a.shape[0]
   cols = a.shape[1]
   for i in range(0,rows):
      for j in range(0,cols):
         print("%6.3f\t" %a[i,j]),
      print
   print 

#####################################################################
################################# MAIN ##############################

### SSE with dummy

#load matrix with data and dummy
[X,Y] = load_matrix("housing_train.txt")
[Xt,Yt] = load_matrix("housing_test.txt")
#get weight vector
W = weight_vector(X,Y)
Wt = weight_vector(Xt,Yt)
#calculate sum square error
E = sum_sq_err(W,X,Y)
Et = sum_sq_err(W,Xt,Yt)
#print sum square error
print("Training Data With Dummy SSE: ", E)
print("Test Data With Dummy SSE: ", Et)



### SSE witout dummy

#remove dummy
X_nd = no_dummy(X)
Xt_nd = no_dummy(Xt)
#get weight vector
W_nd = weight_vector(X_nd,Y)
#calculate sum square error
E_nd = sum_sq_err(W_nd,X_nd,Y)
Et_nd = sum_sq_err(W_nd,Xt_nd,Yt)
#print sum square error
print("Training Data Without Dummy SSE: ", E_nd)
print("Test Data Without Dummy SSE: ", Et_nd)



### adding features
[num_feat, List_SSE, List_SSEt] = feature_relationships(10)
#plot_relationship(num_feat, List_SSE, List_SSEt,"Number of Additonal Features")
print num_feat
print List_SSE
print List_SSEt

#### regression with lambda
lambda_list = [0.01, 0.05, 0.1, 0.5, 1, 5]
[lambda_lsit, List_SSE, List_SSEt] = lambda_relationships(lambda_list)
#plot_relationship(lambda_list, List_SSE, List_SSEt,"Value of Lambda")
print lambda_list
print List_SSE
print List_SSEt

printMatrixE(X_nd)
