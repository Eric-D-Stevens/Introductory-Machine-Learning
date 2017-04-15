
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
	X=X[:,:X.shape[1]-1]
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
	return [num_feat, List_SSE, List_SSEt]

def plot_relationship(num_feat, List_SSE, List_SSEt, xaxis):

	plt.figure(1)
	plt.subplot(211)
	plt.ylabel("SSE")
	plt.plot(num_feat, List_SSE, 'r')
	plt.title("Training Data")
	plt.subplot(212)
	plt.xlabel(xaxis)
	plt.ylabel("SSE")
	plt.plot(num_feat, List_SSEt, 'b')
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

#######################################################
print "\n\n######   Parts 1 & 2   ######"

# load matrix with training data and dummy variable
[X,Y] = load_matrix("housing_train.txt")

# get weight vector
W = weight_vector(X,Y)

# calculate sum square error using traning data
E = sum_sq_err(W,X,Y)

# print the calculated weight vector
print "\nW = ", W



#######################################################
print "\n\n######     Parts 3     ######"

# load matrix with test data and dummy variable
[Xt,Yt] = load_matrix("housing_test.txt")

# calculate SSE for testing data
Et = sum_sq_err(W,Xt,Yt)

# print the SSE with dummy variable
print "Traning Data With Dummy SSE: ", E
print "Test Data With Dummy SSE: ", Et



########################################################
print "\n\n######     Parts 4     ######"

# remove dummy
X_nd = no_dummy(X)
Xt_nd = no_dummy(Xt)

# get weight vector with no dummy
W_nd = weight_vector(X_nd,Y)

# calculate sum square error for no dummy
E_nd = sum_sq_err(W_nd,X_nd,Y)
Et_nd = sum_sq_err(W_nd,Xt_nd,Yt)

print "\nWeight Vector with no dummy"
print "W = ", W

# print SSEs for no dummy
print "\nTraining Data Without Dummy SSE: ", E_nd
print "Test Data Without Dummy SSE: ", Et_nd



########################################################
print "\n\n######     Parts 5     ######"

added = 20

### adding features
[num_feat, List_SSE, List_SSEt] = feature_relationships(added)

#plot
#plot_relationship(num_feat, List_SSE, List_SSEt,"Number of Additonal Features")
print "\nPlot output supressed, uncomment to plot"



########################################################
print "\n\n######     Parts 6     ######"

# generate a list of lambdas to test
lambda_list = np.arange(0,5,.01)

# calculate SSE for every lambda in lambda_list
[lambda_lsit, List_SSE, List_SSEt] = lambda_relationships(lambda_list)

# plot 
#plot_relationship(lambda_list, List_SSE, List_SSEt,"Value of Lambda")
print "\nPlot output supressed, uncomment to plot"



########################################################
print "\n\n######     Parts 7     ######"

print "\nSee report"



########################################################
print "\n\n######     Parts 8     ######"

print "\nSee report"


print "\n\n\n\n"

