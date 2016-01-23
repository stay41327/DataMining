KNN_Param = 200
SNN_TH = 0.5

# Pre-process
fd = open('plants.data','r')
raw_dat = []
for line in fd:
    raw_dat.append( line.strip().split(',') )
fd.close()
print 'Load Data Done.'

# Construct Proximixity Matrix

def commItem( l1, l2 ):
    cnt = 0
    for r1 in l1:
        for r2 in l2:
            if r1 == r2:
                cnt += 1
    return cnt

# Origin: directly use the common item number as 'Proximity'
# Improve: the the common item number in percentage of min (state number)
prx_mtx =  {}
for i in range( len(raw_dat) ):
    item = raw_dat[i]
    tempDic = {}
    for cmpItem in raw_dat[i+1:]:
 #       tempDic[ cmpItem[0] ] = commItem( item[1:] , cmpItem[1:] )
        tempDic[ cmpItem[0] ] = float( commItem( item[1:] , cmpItem[1:] ) ) / min( len(item[1:]), len(cmpItem[1:]) )
    prx_mtx[ item[0] ] = tempDic

# Clustering
def getEntry( name1, name2, ref ):
    if name1 == name2:
        return 0
    try:
        return ref[name1][name2]
    except:
        return ref[name2][name1]
#Generate K-NN Dictionary
nameList = [ i for i in prx_mtx]

knn_dict = {}

for name1 in nameList:
    knn_buf = [ ['',-1] for i in range(KNN_Param) ]
    for name2 in nameList:
        knn_buf.sort( key=lambda x:x[1] )
        proximity = getEntry( name1,name2,prx_mtx )
        if proximity >= knn_buf[0][1]:
            knn_buf[0][1] = proximity
            knn_buf[0][0] = name2
    knn_dict[name1] = dict(knn_buf)
#Generate SNN Similarity Matrix
#   Trick: reuse the prx_mtx
snn_mtx = prx_mtx.copy()
for node in prx_mtx:
    knn_node = knn_dict[node]
    for testnode in prx_mtx[node]:
        knn_test = knn_dict[testnode]
        if commItem( knn_node, [testnode] ) == 0 \
        or commItem( knn_test, [node] ) == 0:
            snn_mtx[node][testnode] = 0
        else:
            snn_mtx[node][testnode] = commItem( knn_node, knn_test )
print 'Graph Generated!'

# Clustering Result
cluster = []
outlier = []
nameList = [ i for i in snn_mtx ]
threshold = KNN_Param * SNN_TH

def iterClst (clstList, namesList, threshold):
    for node in clstList:
        j = 0
        while j < len(nameList):
            if getEntry(node, nameList[j], snn_mtx) >= threshold:
                clstList.append( nameList[j] )
                del nameList[j]
            j += 1
    return clstList

while nameList:
    j = 0
    subclst_buf = [ nameList[0] ]
    del nameList[0]
    while j < len(nameList):
        if getEntry(subclst_buf[0], nameList[j], snn_mtx) >= threshold:
            subclst_buf.append( nameList[j] )
            del nameList[j]
        else:
            j += 1
    if len(subclst_buf) == 1:
        outlier.append( subclst_buf[0] )
    else:
        cluster.append( iterClst(subclst_buf, nameList, threshold) )


print 'Cluster Number: ', len(cluster)
print 'Outlier Number: ', len(outlier)

# Evaluation
# 'state' set for each cluster
stateList = []
for c in cluster:
    state = []
    for indx in c:
        for record in raw_dat:
            if record[0] == indx:
                state.append( record[1:] )
                break
    stateList.append( reduce( lambda x,y: list(set(x)|set(y)), state ) )
for i in range(len(cluster)):
    print 'Cluster ', i, ': '
    print '---> Number of Nodes: ', len(cluster[i])
    print '---> States included: ', str(stateList[i])

# matplotlib visualized similarity matrix
# NOTE!: Having Problem in GUI when on Servers
# Solutioin: dump prm_mtx and snn_mtx into files & depic it locally

def ServerSide (prx_mtx,snn_mtx):
    import pickle
    fd = open('prx_mtx','w')
    pickle.dump( prx_mtx, fd )
    fd.close()
    fd = open('snn_mtx','w')
    pickle.dump( snn_mtx,fd)
    fd.close()

def ClientSide():
    # For Local
    fd = open('prx_mtx','r')
    prx_mtx = pickle.load(fd)
    fd.close()
    fd = open('snn_mtx','r')
    snn_mtx = pickle.load(fd)
    fd.close()
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    #   Origin ( before clustering )
    data = []
    for x in prx_mtx:
        data.append( [getEntry(x,y,prx_mtx) for y in prx_mtx] )
    np.array( data )
    cmap = matplotlib.cm.jet
    norm = matplotlib.colors.Normalize( vmin=min([min(i) for i in data]),
                                        vmax=max([max(i) for i in data]) )
    im = plt.imshow(data,cmap=cmap,norm=norm)
    plt.colorbar(im)
    plt.savefig('origin.png')
    # Clear the plot ( plt.show() can also be an option)
    plt.clf()
    
    #   SNN Similarity ( in clustering )
    data = []
    for x in snn_mtx:
        data.append( [getEntry(x,y,snn_mtx) for y in snn_mtx] )
    np.array( data )
    cmap = matplotlib.cm.jet
    norm = matplotlib.colors.Normalize( vmin=min([min(i) for i in data]),
                                        vmax=max([max(i) for i in data]) )
    im = plt.imshow(data,cmap=cmap,norm=norm)
    plt.colorbar(im)
    plt.savefig('origin_snn.png')
    # Clear the plot ( plt.show() can also be an option)
    plt.clf()
