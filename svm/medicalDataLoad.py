
#-*- coding:utf-8 -*-

# In[1]:

import csv
import os
import pickle
import numpy as np
import requests
import json


#存储list或者dict
def saveLists(filePath, fileName,saveList):
    f1 = open(os.path.join(filePath,fileName ),"wb")
    pickle.dump(saveList, f1)
    f1.close()


# In[3]:

def loadLists(filePath, fileName):
    saveList = list()
    f1 = open(os.path.join(filePath,fileName ),"rb")
    saveList = pickle.load(f1)
    f1.close()
    return saveList


# In[32]:

import re
#数据预处理，去除一些字符，过滤掉字母和数字
def doWordFormat(word):
    #word=u"128我(我是我）"
    word = word.replace("（","(")
    word = word.replace("）",")")
    word = word.replace(" ",",")
    fil1 = re.compile(r'\((.*?)\)' )
    wordFilter = fil1.sub('',word)#.decode("utf8")
    #print("%s"  %(wordFilter))
    filtrate = re.compile(u'[A-Za-z0-9_]')#非中文
    filtered_str = filtrate.sub(r'', wordFilter)#.decode("utf8")#replace
  
    format_str =  ("%s" %(filtered_str))
    #print(" fileter:%s" %(filtered_str))
    #print(format_str)
    return format_str


# In[33]:

#通过restAPI接口，计算关键字
def doKeyWord(text):
    url = 'http://10.110.13.171:9080/nlp-core-service/nlp/annotators/keywords'
    params = {
        "text": text,
        "access_token": "123456",
        "lang": "Chinese",
        "output_format": "json"
            }
    r = requests.get(url, params = params)
    return r.text
#通过restAPI接口，计算关键字
def doKeyWordPost(text):
    url = 'http://10.110.13.172:9700/nlp-core-service/nlp/annotators/keywords'
    params = {
        "text": text,
        "access_token": "123456",
        "lang": "Chinese",
        "output_format": "json",
        "size":"10"
            }
    paramsJson = json.dumps(params)
    #print(params)
    #r = requests.post(url, data = paramsJson)
    r = requests.post(url, json = params)
    #print(r.url)
    #print(r.status_code)
    #print(r.content)
    return r.text

# In[34]:

#通过restAPI接口，计算分词
def doSeg(text):
    url = 'http://10.110.13.171:9080/nlp-core-service/nlp/annotators/segment'
    params = {
        "text": text,
        "access_token": "123456",
        "lang": "Chinese",
        "output_format": "text"
            }
    r = requests.get(url, params = params)
    return r.text
def doSegmentPost(text):
    #url = 'http://10.110.13.172:8316/nlp-segment-service/nlp/annotators/pos'
    url = 'http://10.110.13.172:9600/nlp-cs-segment/nlp/annotators/segment'
    params = {
        "text": text,
        "access_token": "123456",
        "lang": "Chinese",
        "output_format": "text"
            }
    #paramsJson = json.dumps(params)
    #print(params)
    #r = requests.post(url, data = paramsJson)
    r = requests.post(url, json = params)
    #print(r.url)
    #print(r.status_code)
    if(r.status_code!=200):
        return ''
    #print(r.content)
    return r.text

# In[35]:

#从返回的json中，获取关键字text
def getKeyWordFromJson(res):
    num=0
    wordList = list()
    keyWordStr=''
    try:

        dicList=json.loads(res)

        for item in dicList :
            #print item,dicList[item]
            for keys in dicList[item]:
                num += 1
                keyWordStr += keys['keyword']+ " "
                wordList.append(keys['keyword'])
    except:
        print("No Key Words:"+ res)
        
    return keyWordStr,num, wordList

def getKeyWordFromSource(item):
    keyStr = doKeyWord(item)
    resStrR = keyStr.replace("'", "\"")
    remStrR = re.sub(r",\s*?]", "]", resStrR)
    #print(remStrR)
    keyWordStr,num, tmpList = getKeyWordFromJson(remStrR)
    return keyWordStr

def doPrepareSourceDiseaseAndAdmin(disease, admin):
    #print("admin=",admin)
    adminKey = getKeyWordFromSource(admin)
    return disease +" "+ adminKey

# In[36]:

#读取csv文件。并进行预处理
def doLoadDisCSVFile(filePath,csvFileName, midFileName):
    dis = open(os.path.join(filePath, midFileName), 'wt')

    labelList = list()
    #typeDict = dict()
    with open(os.path.join(filePath, csvFileName), 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        #headers = next(spamreader)
        for row in spamreader:
           
            if row is None:
                continue
           # print("%s %s %s " %(row[0], row[1], row[2]))
            gen = row[1]
            des = row[2]
            #对样本中，标注的性别转换，0：女 1：男 2：未知
            gen=(int(gen))
            if gen ==2:
                gen = 0
            else:
                if gen == 3:
                    gen = 2
            #print("gen=",gen)
            labelList.append(gen)
            #print des
           
            #desTrans = doWordFormat(des.decode("utf8"))
            #remove some characters
            desTrans = doWordFormat(des)
           
            dis.write(desTrans+"\n")
    return labelList


# In[37]:

#filePath = "/data/1xiu/project/pythonProject/medicalPOC/hospital/datasets/"
#csvFileName = 'hospitalSel4-3types0725-7110.csv'
#midFileName = "hospitalSel4-3types0725-7110.txt"
#labelList = list()
#labelList = doLoadDisCSVFile(filePath,csvFileName, midFileName)  
#print len(labelList)


# In[38]:

def saveLabels(filePath, labelName, labelList):
    f1 = open(os.path.join(filePath, labelName),"wb")
    pickle.dump(labelList, f1)
    f1.close()


# In[39]:

#labelName='hospitalSel4-3types0725-7110.labels'
#saveLabels(filePath, labelName, labelList)


# In[40]:

def getStopWordDict(filePath, stopFileName):
    #get stop word list
    stopDict = dict()
    dis = open(os.path.join(filePath, stopFileName), 'rb')
    index=0
    for stop in dis:
        stopDict[stop]=index
        index +=1
   
    return stopDict


# In[41]:

#load stop word list
#stopFileName =  "stopword.list"
#stopDict = dict()
#stopDict = getStopWordDict(filePath,stopFileName)


# In[42]:

def getKeyWords(filePath, fileName):
    dis = open(os.path.join(filePath, fileName), 'rb')
    num = 0
    disDict = dict()
    wordList = list()
    tmpList = list()
    keywordRowList = list()
    for item in dis:
    
        if item is None:
            continue
        
        #get key words
        num +=1 
        keyStr = doKeyWord(item)

        resStrR = keyStr.replace("'", "\"")
        remStrR = re.sub(r",\s*?]", "]", resStrR)
            
        keyWordStr,num, tmpList = getKeyWordFromJson(remStrR)
        #print ("str=%s" %(keyWordStr))
        wordList.extend(tmpList)
        keywordRowList.append(keyWordStr)
        
    #print num
    return wordList,keywordRowList


# In[43]:

#get key words
#wordList = list()
#keywordRowList = list()
#wordList,keywordRowList = getKeyWords(filePath, midFileName)
#print("wordList len=%d" %(len(wordList)))


# In[44]:

#读取词列表，去除停用词，构建新的关键词词典

def getKeyWordDict(wordList, stopDict):
    wordDict = dict()
    wordDictNew = dict()
    index = 0
    #list to dict
    for item in wordList:
        if item not in wordDict:
            wordDict[item] = index
            index +=1
   
    #remove stop word in wordDict
    index=0
    for item in wordDict:
        if item not in stopDict:
            wordDictNew[item] = index
            index +=1    
    vecSize =  len(wordDictNew)
   
    return wordDictNew,vecSize


# In[46]:

#get keyword Dict
#print len(wordList)
#wordDictNew = dict()
#wordDictNew, vecSize = getKeyWordDict(wordList,stopDict )
#print ("vecSize=%d" %(vecSize))


# In[47]:

#save key word dict 
#saveLists(filePath, 'hospitalWordDict3types', wordDictNew)


# In[48]:

#获取每行文本的向量
def getRowVector(wordDictNew, row, vecSize):
    tmp = np.zeros(vecSize)
    rowArr = row.split(' ')
    for i in range(len(rowArr)):
        if rowArr[i] in wordDictNew:
            index = wordDictNew[rowArr[i]]
            tmp[index]=1
    return tmp


# In[49]:

#获取所有文本行的向量
def getAllVectors(wordDictNew, keywordRowList,vecSize):
    wordVectorList = list()
    for row in keywordRowList:
        tmp = getRowVector(wordDictNew, row, vecSize)
        wordVectorList.append(tmp)
    return wordVectorList


# In[50]:

#generate all  vectors
#wordVectorList = list()
#wordVectorList = getAllVectors(wordDictNew, keywordRowList, vecSize)


# In[52]:

#sava all vectors
#saveLists(filePath, 'hospitalSelWordVectors3types', wordVectorList)


# In[53]:

def saveKeyWordUTF8(filePath, fileName, wordList):
    dis = open(os.path.join(filePath, fileName), 'wb')
    for item in wordList:
        tmp = ("%s" %(item.encode("utf8")))
        dis.write( tmp+"\n")
    dis.close()


# In[54]:

#filePath = "/data/1xiu/project/pythonProject/medicalPOC/word2vec/datasets/"
#fileName = "hospitalKeyWord.txt"
#saveKeyWordUTF8(filePath , fileName, wordList)


# In[56]:

def preProcessAllDatasets(filePath, fileName, saveKeyWordRowName, saveKeyWordDictName):
    #filePath = "/data/1xiu/project/pythonProject/medicalPOC/hospital/datasets/"
    #csvFileName = 'hospitalSel4-3types0725-7110.csv'
    midFileName = "hospitalMid.txt"
    labelList = list()
    labelList = doLoadDisCSVFile(filePath,fileName, midFileName)  
    print( len(labelList))
    
    #get key words
    wordList = list()
    keywordRowList = list()
    wordList,keywordRowList = getKeyWords(filePath, midFileName)
    print("wordList len=%d" %(len(wordList)))
    
    #get keyword Dict
    print( len(wordList))
    wordDictNew = dict()
    wordDictNew, vecSize = getKeyWordDict(wordList)
    print ("vecSize=%d" %(vecSize))
    
    saveKeyWordUTF8(filePath , saveKeyWordRowName, wordList)
    
    #save key word dict 
    saveLists(filePath, saveKeyWordDictName, wordDictNew)


# In[ ]:

#filePath = "/data/1xiu/project/pythonProject/medicalPOC/word2vec/datasets/"
#fileName = "hospital1-yy.csv"
#saveKeyWordRowName = "hospital1-yy.txt"
#saveKeyWordDictName = "hospital1-yy.dict"
#preProcessAllDatasets(filePath, fileName, saveKeyWordRowName, saveKeyWordDictName)


# In[ ]:



# In[ ]:



