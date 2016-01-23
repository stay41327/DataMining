import random

# item numbers for stopping classification
MIN_TH_CLASS = 50
# Pre-Pruning Threashold
TH_CRATE = 0.90
# Holdout sampling trainning number
HOLDOUT = 10000*6/10
# Generalized Error Param
GERR = 0.5

# Data Pool
# Pre-process
def randPick( Sampling_num=HOLDOUT, dest='training.data' ):
    source = open('source.data','r')
    wrt = open(dest,'w')
    wrt.truncate()
    wrt.close()
    wrt = open(dest,'a')
    buf = []
    for l in source:
        buf.append(l)
    for c in range(Sampling_num):
        ind = random.randint(0,len(buf)-1)
        wrt.write(buf[ind])
    wrt.close
    
def initialize():
    '''return value : initialized data pool'''
    dat = []
    source = open('training.data','r')
    for line in source:
        buf = []
        for d in line.split(','):
            buf.append(float(d))
        dat.append(buf)
    source.close()
    return dat

# Discretize attribute
def discret( subDat, index):
    ''' subDat-> subset of Data pool; index-> specify attribute;
        return value: (float) split point '''
    # sort by attribute[index]
    subDat.sort(key = lambda d: d[index])

    # if attribute all the same, return directly
    if subDat[0][index] == subDat[-1][index]:
        return 0

    # calculate Gini
    gini = []
    for i in range(len(subDat)-1):
        psplit = (subDat[i][index] + subDat[i+1][index])/2
        cntMtx = [[0.0,0.0],    # 0 <= >
                  [0.0,0.0]]    # 1 <= >
        for tup in subDat:
            if tup[index] <= psplit and tup[-1] == 0:
                cntMtx[0][0] += 1
            elif tup[index] > psplit and tup[-1] == 0:
                cntMtx[0][1] += 1
            elif tup[index] <= psplit and tup[-1] == 1:
                cntMtx[1][0] += 1
            elif tup[index] > psplit and tup[-1] == 1:
                cntMtx[1][1] += 1
        
        try:
            g = (cntMtx[0][0]+cntMtx[1][0])/(cntMtx[0][0]+cntMtx[0][1]+cntMtx[1][0]+cntMtx[1][1]) * \
                (1-((cntMtx[0][0]/(cntMtx[0][0]+cntMtx[1][0]))**2)-((cntMtx[1][0]/(cntMtx[0][0]+cntMtx[1][0]))**2))\
                + \
                (cntMtx[0][1]+cntMtx[1][1])/(cntMtx[0][0]+cntMtx[0][1]+cntMtx[1][0]+cntMtx[1][1]) * \
                (1-((cntMtx[0][1]/(cntMtx[0][1]+cntMtx[1][1]))**2)-((cntMtx[1][1]/(cntMtx[0][1]+cntMtx[1][1]))**2))

            gini.append( [psplit, g] )

        except ZeroDivisionError:
            pass
            
    # pick the one with least gini
    gini.sort(key = lambda d: d[1])
    return gini[0][0]

# Tree Generation
#   Constructing trees with Greedy Algorithm
def nodePick( subDat ):
    buf = [[i,discret(subDat,i)] for i in range(14)]
    buf.sort( key = lambda d: d[1] )
    return buf[0] # return [index, splitPoint]

# The Stopping-Condition
def stopping_cond( subData , crate ):
    # NO.1 Same Class - Successfully Classified
    if crate == 1:
        return True

    # NO.2 Class Number Under Threshold
    if len(subData) <= MIN_TH_CLASS :
        return True

    # NO.??? Nothing matched
    return False

def treeGen( DataSet ):
    # Calculate Corrate Rate & Statistical Guess
    # Specifically Designed for Pre-Prunning
    crate = 0.0
    rpguess = -1
    cnt = [0.0,0.0] # [ 1: num , 0: num ]
    for l in DataSet:
        if l[-1] == 1:
            cnt[0] += 1
        else:
           cnt[1] += 1
    try:
        if cnt[0] > cnt[1]:
            crate = cnt[0]/(cnt[0]+cnt[1])
            rpguess = 1
        else:
            crate = cnt[1]/(cnt[0]+cnt[1])
            rpguess = 0
    except:
        pass
    
    # Stopping Condition
    if stopping_cond( DataSet , crate ) == True :
        # Leaf node, id = -1, th = 0|1, pointer = NIL
        return [ -1, rpguess, crate, [], []]

    # Pre-Prunning
    if crate >= TH_CRATE:
        return [ -1, rpguess, crate, [], []]
        
    Psplit = nodePick( DataSet )
    # node Structure
    # [ index , split_point , correct rate , child_node_<= , child_node_> ]
    node = [Psplit[0],Psplit[1],crate]
    lchild = [] # child_node_<= dataset
    rchild = [] # child_node_> dataset
    for l in DataSet:
        if l[Psplit[0]] <= Psplit[1]:
            lchild.append(l)
        else:
            rchild.append(l)
    # recurse to generate sub_tree
    node.append(treeGen(lchild))
    node.append(treeGen(rchild))
    return node

# Post-Pruning
# REP Pruning
def pruning(node,testSet):
    # Stop Condition
    if node[0] == -1:
        return 0

    # Recurisive Pruning
    rChild = []
    lChild = []
    for item in testSet:
        if item[node[0]] <= node[1]:
            lChild.append(item)
        else:
            rChild.append(item)
    pruning(node[3],lChild)
    pruning(node[4],rChild)

    # Major Function
    # Introduce Generallized Error
    try:
        cnt = 0.0
        for item in testSet:
            if classify( item, node) == True:
                cnt += 1
        gErr = GERR*len(str(node).split('['))/2
        cnt = len(testSet) - cnt
        tree_error = (cnt + gErr)/len(testSet)
        cnt = 0.0
        for item in testSet:
            if item[-1] == 0:
                cnt += 1
        value = -1
        ratioE = 0.0
        if cnt/len(testSet) > 0.5:
            value = 0
            ratioE = 1-cnt/len(testSet)
        else:
            value = 1
            ratioE = cnt/len(testSet)
        if ratioE <= tree_error:
            # Prune with leaf node
            node = [-1,value,cnt/len(testSet),[],[]]
    except ZeroDivisionError:
        pass

# Classify data
# Input: 1 line of data with 15 attributes
# Return: True, if classify correct; else, return False
def classify( item , node ):
    # Stop Condition
    if node[0] == -1:
        return item[-1] == node[1]
    # Recurise
    if item[node[0]] <= node[1]:
        return classify(item,node[3])
    else:
        return classify(item,node[4])

# Print Tree
def Tprint( node , level=0):
    # Stop Condition
    if node[0] == -1:
        return 0
    for i in range(level):
        print "--->",
    level += 1
    print "index:%d; <= %f\n" %(node[0]+1,node[1])
    Tprint( node[3], level )
    for i in range(level - 1):
        print "--->",
    print "index:%d;  > %f\n" %(node[0]+1,node[1])
    Tprint( node[4], level )

# Evaluation
def confuseMtx(root, testData):
    # Confusion Metrix
    mtx = [[0.0,0.0],      # Predict: 0 , 1  Real: 0
           [0.0,0.0]]      #                       1
                     # [TP,TN,FP,FN]
    data_0 = []
    data_1 = []
    for i in testData:
        if i[-1] == 0:
            data_0.append(i)
        else:
            data_1.append(i)
    for i in data_0:
        if classify( i, root) == True:
            mtx[0][0] += 1
        else:
            mtx[1][0] += 1
    for i in data_1:
        if classify( i, root) == True:
            mtx[1][1] += 1
        else:
            mtx[0][1] += 1
    return mtx

def accurency(mtx):
    return (mtx[0][0]+mtx[1][1])/(mtx[0][0]+mtx[0][1]+mtx[1][0]+mtx[1][1])

def complexity(root):
    return len(str(root).split('['))/2




# Main Function
# Set Maxmium Recursion Depiton to 1000000000
import sys
sys.setrecursionlimit(1000000000)
# import training data
randPick()
dat = initialize()
# Generate Decision Tree
root = treeGen(dat)
# import test data
randPick(HOLDOUT/2)
testdat = initialize()
# post-pruning
pruning(root,testdat)
# print Decision Tree
Tprint(root)
print "++++++++++++++++++++++++++++++++++Evaluation+++++++++++++++++++++++++++++++"
# confuse matrix
mtx = confuseMtx(root,testdat)
print "Confuse Matrix:"
print str(mtx)
# accureny
print "Accurency: "+str(accurency(mtx))
# Tree complexity
print "Tree Complexity: "+str(complexity(root))

