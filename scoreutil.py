import numpy as np


def calcRMSE(trueScores, pScores):
    assert len(trueScores) == len(pScores)

    f1 = np.array(trueScores).astype(np.float)
    f2 = np.array(pScores).astype(np.float)

    return np.sqrt(np.mean((f1 - f2) ** 2))

    # sumDiff = 0
    #
    # for i in range(len(valData)):
    #     score = valData[i][2]
    #     pScore = pScores[i]
    #
    #     diff = (score - pScore) ** 2
    #     sumDiff += diff
    #
    # return np.sqrt(sumDiff / len(valData))


def printTopDiffs(valData, pScores, n = 100):
    assert len(valData) == len(pScores)
    result = []

    for i in range(len(valData)):
        userId = valData[i][0]
        itemId = valData[i][1]
        score = valData[i][2]
        pScore = pScores[i]

        diff = (score - pScore) ** 2
        result.append((diff, pScore, score, userId, itemId))
    result = sorted(result, key=lambda x: x[0])
    for i in range(n):
        print(result[i])

    result = sorted(result, key=lambda x: -x[0])
    for i in range(n):
        print(result[i])
