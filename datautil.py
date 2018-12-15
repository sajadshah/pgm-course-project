import csv
import numpy as np


def readCsvData(path):
    data = np.genfromtxt(path, delimiter=',', dtype=np.float32)
    # data[:, [0, 1]] = data[:, [0, 1]].astype(int)
    # data[:, [0, 1]] = np.floor(data[:, [0, 1]])
    data[:, [0, 1]] = np.array(data[:, [0, 1]], dtype=int)

    return data

def prepareUserItemDicts(data):
    usersDict = {}
    itemsDict = {}
    usersDict[0] = {'items': [], 'userIndex': 0}
    itemsDict[0] = {'users': [], 'itemIndex': 0}
    maxUserId = data.max(axis=0)[0]
    maxItemId = data.max(axis=0)[1]
    for x in data:
        userId = x[0]
        itemId = x[1]
        score = x[2]
        if userId in usersDict:
            usersDict[userId]['items'].append((itemId, score))
        else:
            usersDict[userId] = {'items': [(itemId, score)], 'userIndex': len(usersDict)}

        if itemId in itemsDict:
            itemsDict[itemId]['users'].append((userId, score))
        else:
            itemsDict[itemId] = {'users': [(userId, score)], 'itemIndex': len(itemsDict)}

    return usersDict, maxUserId, itemsDict, maxItemId


def prepareValidationUserItemDicts(data, trainUsersDict, trainItemsDict):
    usersDict = {}
    itemsDict = {}

    for x in data:
        userId = x[0]
        itemId = x[1]
        score = x[2]
        if userId not in trainUsersDict and itemId in trainItemsDict:
            userIndex = 0
            itemIndex = trainItemsDict[itemId]['itemIndex']
        elif userId in trainUsersDict and itemId not in trainItemsDict:
            userIndex = trainUsersDict[userId]['userIndex']
            itemIndex = 0
        elif userId in trainUsersDict and itemId in trainItemsDict:
            userIndex = trainUsersDict[userId]['userIndex']
            itemIndex = trainItemsDict[itemId]['itemIndex']
        else :
            userIndex = 0
            itemIndex = 0

        if userId in usersDict:
            usersDict[userId]['items'].append((itemId, score))
        else:
            usersDict[userId] = {'items': [(itemId, score)], 'userIndex': userIndex}

        if itemId in itemsDict:
            itemsDict[itemId]['users'].append((userId, score))
        else:
            itemsDict[itemId] = {'users': [(userId, score)], 'itemIndex': itemIndex}

    return usersDict, itemsDict

def prepareI_R_Data(usersDict, maxUserId, itemsDict, maxItemId) :
    rData = np.zeros([maxUserId, maxItemId], dtype=np.float32)
    iData = np.zeros([maxUserId, maxItemId], dtype=np.float32)
    sumScores = 0
    count = 0
    for uId, uData in usersDict.items():
        for i in uData['items']:
            score = i[1]
            count += 1
            sumScores += score
            rData[uId][i[0]] += score
            iData[uId][i[0]] = 1
    meanScores = sumScores / count
    rData = removeMean(usersDict, itemsDict, rData, meanScores)
    print(np.count_nonzero(iData))
    print(np.count_nonzero(rData))

    return rData, meanScores, iData

def removeMean(usersDict, itemsDict, rData, meanScores):
    for uId, uData in usersDict.items():
        for i in uData['items']:
            rData[uData['userIndex']][itemsDict[i[0]]['itemIndex']] -= meanScores

    return rData


def prepareValI_R_Data(usersDict, itemsDict, trainUserDict, maxUserId, trainItemDict, maxItemId) :
    rData = np.zeros([nTrainUsers, nTrainItems], dtype=np.float32)
    iData = np.zeros([nTrainUsers, nTrainItems], dtype=np.float32)
    for uId, uData in usersDict.items():
        for i in uData['items']:
            rData[uData['userIndex']][itemsDict[i[0]]['itemIndex']] += i[1]
            iData[uData['userIndex']][itemsDict[i[0]]['itemIndex']] = 1

    return rData, iData



