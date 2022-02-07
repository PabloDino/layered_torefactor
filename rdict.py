f=open('dict.txt','r')
fout = open('dict50.csv','w')
data = f.read()
lmap=eval(data)
fout.write('classID,class\n')
for l in lmap:
  strval =str(lmap[l])+','+l +'\n'
  fout.write(strval)
  
  print(strval)
fout.close()
