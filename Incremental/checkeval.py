import os

def evaluateCheckPoints(prefix):
    files=[]
    for fle in os.listdir():
        if fle.startswith(prefix):
            files.append(fle)
    maxvacc=0
    maxdx=0
    print(files)
    for dx in range(len(files)):
        arr = files[dx].split("-")
        print(arr[0])
        if  float(arr[3])>maxvacc:
            maxvacc = float(arr[3])
            maxdx = dx
    arr = files[maxdx].split("-")
    retloss = float(arr[2])
    retf1 = float(arr[4])
    retprecision = float(arr[5])
    retrecall = float(arr[6])
    acc = float(arr[7])
    #for fle in files:
    #    os.remove(fle)
    return retloss,acc, maxvacc, retf1, retprecision, retrecall
    
    
loss, valacc,f1,precision, recall, acc  = evaluateCheckPoints("dense2lstm")
    
print('Loss for best accuracy:', loss)
print('Best validation accuracy:', valacc)
print('Best training accuracy:', acc)

