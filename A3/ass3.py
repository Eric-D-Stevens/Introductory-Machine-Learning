import numpy as np
import math
from matplotlib import pyplot as plt


def plot(x, y):
   plt.figure(1)
   plt.plot(x, y, 'r')
   plt.show()


def printMatrixE(a):
   print "Matrix["+("%d" %a.shape[0])+"]["+("%d" %a.shape[1])+"]"
   rows = a.shape[0]
   cols = a.shape[1]
   for i in range(0,rows):
      for j in range(0,cols):
            print("%6f  " %a[i,j]),
      print
   print 


def load_matrix(filename):
   # load data into matrix X from file
   data = np.genfromtxt(filename,delimiter=',')
   X = np.matrix(data)
   
   # create Y from X
   Y = X.copy()
   Y = Y[:,0]
    
   # remove last column form X
   #X = X[:,1:X.shape[1]]
   # return the matricies
   return [X,Y]


# Gets the max and min values for each column of training data
def get_normalizers(X):
   
   max_vals = []
   min_vals = []

   for i in range(1, X.shape[1]):

      min_vals.append(X[:,i].min())
      max_vals.append(X[:,i].max()-min_vals[i-1])

   return [min_vals, max_vals]


# Creat a normalized version of the training data
def normalize_matrix(X, min_vals, max_vals):
   X_normed = X.copy()
   for i in range(1,X_normed.shape[1]):
      X_normed[:,i] = X_normed[:,i]-min_vals[i-1]
      X_normed[:,i] = X_normed[:,i]/max_vals[i-1]
   return X_normed

# calculate distance
def get_distance(x1,x2):
   xd = x1[0,1:] - x2[0,1:]
   xd = xd*xd.T
   return math.sqrt(float(xd[0,0]))

# Uses the normalized training data guess the class of a single input
def knn_test(X_normed, X_test, k):

   distances = []
   for i in range(0,X_normed.shape[0]):
      distances.append(get_distance(X_normed[i,:],X_test))
   distances = np.matrix(distances)
   distances = distances.T
   
   inde = np.argsort(distances, axis=0)
   index = []
   for i in range(0,inde.shape[0]): 
      index.append(int(inde[i,0]))
      #print distances[index[i]] 
   results = []
  
   # print distances[index[0]] 

   for j in range(0, k):
      results.append(float(X_normed[index[j],0])) 
   final = sum(results)
   if final <0: return -1.0
   else: return 1.0
   
    
def full_file_knn(X_normed,Xt_normed,k):
   passes = 0
   for i in range(0, Xt_normed.shape[0]):
      guess = knn_test(X_normed,Xt_normed[i,:],k)
      real = Xt_normed[i,0]
      #print(real,  "   ", guess)  
      if int(guess) == int(real): passes += 1
      #else: print "fail at ", i
   #print float(passes), float(Xt_normed.shape[0])
   return float(passes)/float(Xt_normed.shape[0])

    
def full_file_knn_L1O(X_normed,Xt_normed,k):
   passes = 0
   for i in range(0, Xt_normed.shape[0]):
      X_L1O = X_normed.copy()
      X_L1O = np.delete(X_L1O, (i), axis = 0)
      guess = knn_test(X_L1O,Xt_normed[i,:],k)
      real = Xt_normed[i,0]
      #print(real,  "   ", guess)  
      if int(guess) == int(real): passes += 1
      #else: print "fail at ", i
   #print float(passes), float(Xt_normed.shape[0])
   return float(passes)/float(Xt_normed.shape[0])





###########################################################################

train_filename = "knn_train.csv"

[X,Y] = load_matrix(train_filename)



###############   NORMALIZE   ###################

[min_vals, max_vals] = get_normalizers(X)
X_normed = normalize_matrix(X, min_vals, max_vals)


##########    KNN    ################


test_filename = "knn_test.csv"
[Xt,Yt] = load_matrix(test_filename)
Xt_normed = normalize_matrix(Xt, min_vals, max_vals)


"""
training_results = []
print "Training Data:"
print "k\tAccuracy"
for i in range(0, 1):
   percent_correct =  full_file_knn(X_normed,X_normed,(i*2+1))
   print str(i*2+1) + '\t' + str(percent_correct)
   training_results.append(percent_correct)



training_L1O_results = []
print "Training Data (Leave One Out):"
print "k\tAccuracy"
for i in range(0, 26):
   percent_correct =  full_file_knn_L1O(X_normed,X_normed,(i*2+1))
   print str(i*2+1) + '\t' + str(percent_correct)
   training_L1O_results.append(percent_correct)



testing_results = []
print "Testing Data:"
print "k\tAccuracy"
for i in range(0, 26):
   percent_correct = full_file_knn(X_normed,Xt_normed,(i*2+1))
   print str(i*2+1) + '\t' +  str(percent_correct)
   testing_results.append(percent_correct)
   
"""
###############################################################################
###############################################################################

def gain(b1,b2):

   #count total pos and negs
   pt = 0.00001
   nt = 0.00001
   p1 = 0.00001
   p2 = 0.00001
   n1 = 0.00001
   n2 = 0.00001
   for i in range(0,len(b1)): 
      if b1[i] < 0: 
         nt += 1.0
         n1 += 1.0
      else: 
         pt += 1.0
         p1 += 1.0
   for i in range(0,len(b2)): 
      if b2[i] < 0: 
         nt += 1.0
         n2 += 1.0
      else: 
         pt += 1.0
         p2 += 1.0

   total = nt+pt

   Hs = (-1)*(pt/total)*math.log(pt/total,2) - (nt/total)*math.log(nt/total,2)
   Hs1 = -(p1/float(len(b1)))*math.log(p1/float(len(b1)),2) - (n1/float(len(b1)))*math.log(n1/float(len(b1)),2)
   Hs2 = -(p2/float(len(b2)))*math.log(p2/float(len(b2)),2) - (n2/float(len(b2)))*math.log(n2/float(len(b2)),2)
   prob1 = float(len(b1))/total
   prob2 = float(len(b2))/total
   InfoGain = Hs - (prob1*Hs1 + prob2*Hs2)

   return InfoGain

def get_gain(Xin,feature):

   index = np.argsort(Xin[:,feature], axis=0)
   hold_index = []
   for i in range(0,index.shape[0]): hold_index.append(int(index[i,0]))
   hold_class = []
   for i in range(0,index.shape[0]): hold_class.append(float(Xin[hold_index[i],0]))
  

   xax = []
   yax = []
   
   max_gain = 0.0
   max_gain_index = 0
   best_threshold = 0.0
   for i in range(1, len(hold_index)): 
      the_gain  = gain(hold_class[:i],hold_class[i:]) 
      
      if the_gain > max_gain: 
         max_gain = the_gain
         max_gain_index = i
         best_threshold = (float(Xin[hold_index[i-1],feature])+float(Xin[hold_index[i],feature]))/2
      xax.append(i)
      yax.append(the_gain)

   return [max_gain, max_gain_index, best_threshold] 




def find_best_test(Xin):

   best_feature = 1
   best_gain = 0
   best_threshold = 0
   best_index = 0
   for i in range(1,X.shape[1]):
      [current_gain, curind, current_threshold] = get_gain(Xin,i)
      if current_gain > best_gain:
         best_gain = current_gain
         best_feature = i
         best_threshold = current_threshold
         best_index = curind
      #print i, current_gain
   #print
   #print best_feature, best_gain
   return [best_feature, best_gain, best_threshold, best_index]





def learned_stump(Xin, best_feat, threshold):

   actual = []
   stump_answer = []   
   for i in range (0,Xin.shape[0]):
      actual.append(float(Xin[i,0]))
       
      if float(Xin[i,best_feat]) > threshold:
         stump_answer.append(1.0)
      else:
         stump_answer.append(-1.0)

   correct = 0
   for x in range(0, len(actual)):
      if stump_answer[x] == actual[x]: correct+=1
      #print actual[x], stump_answer[x]

   # returns accuracy
   return  float(correct)/float(len(actual))
   





# Implements the stump  
"""
[best_feature, best_gain, best_threshold] = find_best_test(X_normed)
learned_stump(Xt_normed, best_feature, best_threshold)
"""



###############################################################################



class Node:
   def __init__(self,parent,feature,threshold, datalength):
      
      self.threshold = threshold
      self.parent = parent
      self.feature = feature
      self.right = None
      self.left = None
      self.datalength = datalength

def okkk(Xin, parent, depth,  node_array):
   

   [best_feature, best_gain, best_threshold, best_index] = find_best_test(Xin)

   index = np.argsort(Xin[:,best_feature], axis=0)
   hold_index = []
   for i in range(0,index.shape[0]): hold_index.append(int(index[i,0]))
   


   Current = Node(parent, best_feature, best_threshold, len(hold_index))


   if depth > 1:
      Current.left = okkk(Xin[hold_index[:best_index],:], Current, depth-1, node_array)
      Current.right = okkk(Xin[hold_index[best_index:],:], Current, depth-1, node_array)
      # print len(hold_index[:best_index]), len(hold_index[best_index:])
   node_array.append(Current)
   return Current


 




thenodearray = []


okkk(X_normed, None, 6, thenodearray)


print thenodearray[len(thenodearray)-1].parent  


def single_test(Xi, node_array):

   Current = node_array[len(node_array)-1]

   while 1:

      if Xi[0,Current.feature] < Current.threshold:

         if Current.left == None: return -1
         else: Current = Current.left

      else:

         if Current.right == None: return 1
         else: Current = Current.right



def performance(Xin, node_array):
   actual = []
   the_result = []

   for i in range(0,Xin.shape[0]):
      the_result.append(single_test(Xin[i,:], node_array))
      actual.append(int(Xin[i,0]))

   accuracy = 0
   for i in range(0, len(actual)):
      if actual[i] == the_result[i]: accuracy += 1

   return  float(accuracy)/len(actual)



def printTree(node_array):

   print "entered print function\n\n"

   Root = node_array[len(node_array)-1]
   twod = [[Root]]
   for i in range(0,6):
      oned = []
      for j in twod[i]:
         oned.append(j.left)
         oned.append(j.right)
      twod.append(oned)

   for i in range(0, len(twod)):
      string = ""
      for j in range(0,len(twod[i])):
         string = string + " - " + str(twod[i][j].threshold) + " + "
         #string = string + str(j.threshold)
      print string





printTree(thenodearray)


print performance(X_normed, thenodearray)
print performance(Xt_normed, thenodearray)

print X_normed.shape[0]
print Xt_normed.shape[0]









