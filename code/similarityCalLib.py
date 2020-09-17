from dateutil.parser import parse
import numpy as np
from math import sqrt 

class Scores:
    commitType = None
    author = None
    file = None
    # codeDiff = {}
    commitDate = None
    branch = None
    tag = None
    cloc = None
    message = None
    topoDist = None
    scoreSum = None

    def calSimilarityScore(self, a, b, totalCommitDays, longestLength, pathLengthDic, isTime=False):
        sourceId = a["id"]
        targetId = b["id"]

        self.author = getSameScore(a["author"], b["author"])
        self.commitType = getCommitTypeDiffScore(a["commitType"], b["commitType"])
        self.file = getFileNamesDiffScore(a, b)
        self.commitDate = getDateDiffScore(a, b, totalCommitDays)
        self.branch = getBranchDiff(a, b)
        # scDic.tag[targetId] = getTagDiff(a, b)
        self.cloc = getClocDiff(a, b)
        #self.message = getWordsDiffScore(a["corpus"], b["corpus"])
        self.message = getTfidfCosineSimilarity(a["tfidf"],b["tfidf"])
        # scDic.message[targetId] = getMessageDiff(a, b)

        self.topoDist = getTopologyDistanceDiffScore(targetId, longestLength, pathLengthDic)
        # scDic.topoDist[targetId] = getTopoDist(sourceId, targetId, allPairsDic)
        
        self.scoreSum = 0
        if not isTime: 
            self.calSum()
        else:
            self.calSimWOTemporal()

    def calSimWOTemporal(self):
        excludes = ["scoreSum", "topoDist", "commitDate"]
        self.scoreSum = 0
        for instanceName in self.__dict__.keys():
            if not instanceName in excludes:
                self.scoreSum = self.scoreSum + self.__dict__[instanceName]


    def calSum(self):
        self.scoreSum = 0
        for instanceName in self.__dict__.keys():
            if instanceName != "scoreSum":
                self.scoreSum = self.scoreSum + self.__dict__[instanceName]

    def getDetailDic(self):
        detailDic = {}
        for instanceName in self.__dict__.keys():
            detailDic[instanceName] = round(self.__dict__[instanceName], 2)
        return detailDic


    # def calSum(self, commitId):
    #     for instanceName in self.__class__.__dict__.keys():
    #         dic = self.__class__.__dict__.get(instanceName)
    #         if not (instanceName.startswith("__") or callable(dic) or instanceName == "sumById"):
    #             if commitId in dic.keys():
    #                 self.sumById[commitId] = self.sumById.get(commitId, 0) + dic[commitId]

    # def getSumDetailDicById(self, id):
    #     sumDic = {}
    #     # print(self.__class__.__dict__.keys())
    #     for instanceName in self.__class__.__dict__.keys():
    #         dic = self.__class__.__dict__.get(instanceName)
    #         if not (instanceName.startswith("__") or callable(dic)):
    #             value = dic.get(id, None)
    #             if value != None:
    #                 sumDic[instanceName] = round(value, 2)
    #     return sumDic

    # def calSimilarityScore(a, b, totalCommitDays, longestLength, pathLengthDic):
    #     scDic = Scores()
    #     sourceId = a["id"]
    #     targetId = b["id"]

    #     scDic.author[targetId] = getSameScore(a["author"], b["author"])
    #     # scDic.commitType[targetId] = getSameScore(a["commitType"], b["commitType"])
    #     scDic.files[targetId] = getFileNamesDiffScore(a, b)
    #     scDic.commitDate[targetId] = getDateDiffScore(a, b, totalCommitDays)
    #     scDic.branch[targetId] = getBranchDiff(a, b)
    #     # scDic.tag[targetId] = getTagDiff(a, b)
    #     scDic.cloc[targetId] = getClocDiff(a, b)
    #     # scDic.message[targetId] = getMessageDiff(a, b)
    #     scDic.topoDist[targetId] = getTopologyDistanceDiffScore(targetId, longestLength, pathLengthDic)
    #     # print(sourceId, targetId, pathLengthDic[targetId])
    #     # scDic.topoDist[targetId] = getTopoDist(sourceId, targetId, allPairsDic)
    #     scDic.calSum(targetId)
        
    #     return scDic


def getWordsDiffScore(a, b):
    return getJaccardSimilarity(a, b)

def getJaccardSimilarity(a, b):
    intersectionLen = len(set(a).intersection(b))
    unionLen = len(set().union(a, b))

    return (intersectionLen / unionLen) if unionLen != 0 else 1

def getTfidfCosineSimilarity(a,b):
   
    ## Empty corpus
    if "-1" in a.keys() or "-1" in b.keys():
        return 0
    
    a_indices = sorted(a.keys())
    b_indices = sorted(b.keys())
    total_length = len(a_indices) + len(b_indices)
    a_iter = 0
    b_iter = 0
    
    inner_product = 0
             
    while True:
        a_idx = a_indices[a_iter]
        b_idx = b_indices[b_iter]
        if a_idx == b_idx:
            inner_product += a[a_idx] * b[b_idx]
            if a_iter != len(a_indices) - 1:
                a_iter += 1
            elif b_iter != len(b_indices) - 1:
                b_iter += 1
            else:
                break              
        elif a_idx > b_idx:
            if b_iter == len(b_indices) - 1:
                if a_iter == len(a_indices) -1:
                    break
                else:
                    a_iter += 1
            else:
                b_iter += 1
        else:
            if a_iter == len(a_indices) - 1:
                if b_iter == len(b_indices) -1:
                    break
                else:
                    b_iter += 1
            else:
                a_iter += 1
    a_length_sqr = 0
    b_length_sqr = 0
    
    for value in a.values():
        a_length_sqr += value * value
        
    for value in b.values():
        b_length_sqr += value * value
           
    cosine_sim = inner_product / (sqrt(a_length_sqr) * sqrt(b_length_sqr))

    return cosine_sim
   

def getCommitTypeDiffScore(a, b):
    if a == "not_mapped" or b =="not_mapped":
        return 0
    else:
        return getSameScore(a, b)

def getSameScore(a, b):
    return 1 if a == b else 0

def getFileNamesDiffScore(a, b):
    # 같은게 아니라, 얼마나 포함할 수 있는지를 봐야함.
    # 포함관계가 아니냐?

    # jaccard coefficient
    a = a.get("diffStat", {}).get("files", {}).keys()
    b = b.get("diffStat", {}).get("files", {}).keys()

    return getJaccardSimilarity(a, b)

def getTopologyDistanceDiffScore(targetId, longestLength, pathLengthDic):
    return np.interp(pathLengthDic[targetId], (1, longestLength), (1, 0))

def getDateDiffScore(a, b, totalCommitDays):
    # diff가 0이면 1이고, total에 가까울 수록 0
    # 0          1
    # same       total
    diffDays = abs((parse(b["date"]) - parse(a["date"])).days)
#     return np.interp(diffDays, (0, totalCommitDays), (1, 0))
#     print(diffDays, np.interp(np.log10(diffDays), (np.log10(0), np.log10(totalCommitDays)), (np.log10(1), np.log10(0)))
    return np.interp(np.log10(diffDays), (0, np.log10(totalCommitDays) ), (1, 0))
        
def getBranchDiff(a, b):
    # branch 따는 작업 필요
    # branch가 2개일 수도 있는 상황?
    a = a.get("branches", [])
    b = b.get("branches", [])
    
    headName = "origin/HEAD"
    if headName in a:
        a.remove(headName)
    if headName in b:
        b.remove(headName)
    
    if a == [] or b == []:
        return 0
    elif len(set(a).intersection(b)) == len(a):
        return 1
    else:
        return 0

def getTagDiff(a, b):
    # tag 따는 작업 필요
    return b["tag"] == a["tag"]

def getClocDiff(a, b):
    aCloc = getTotalCloc(a)
    bCloc = getTotalCloc(b)
    return np.interp( abs(aCloc - bCloc), (0, max(aCloc, bCloc)), (1, 0))

def getTotalCloc(commit):
    diffStat = commit["diffStat"]
    insCloc = diffStat["insertion"] if "insertion" in diffStat else 0
    delCloc = diffStat["deletion"] if "insertion" in diffStat else 0
    
    return insCloc + delCloc
