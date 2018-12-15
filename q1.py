import tensorflow as tf
import numpy as np
import datautil
import scoreutil
import os

trainDataFile = './dataset/train_user_item_score.txt'
validationDataFile = './dataset/validation_user_item_score.txt'
testDataFile = './96301773_in.txt'
# testDataFile = './dataset/validation_user_item_score.txt'

# hyper parameters
# sigma = 100.0
# sigma_u = 100.0
# sigma_v = 0.001
sigma = 1.0
sigma_u = 1e-1
sigma_v = 1.0
D = 60

trainData = datautil.readCsvData(trainDataFile)
valData = datautil.readCsvData(validationDataFile)
testData = datautil.readCsvData(testDataFile)

meanScores = np.mean(trainData, axis=0)[2]
trainData[:, 2] = trainData[:, 2] - meanScores
valData[:, 2] = valData[:, 2] - meanScores

usersDict, maxUserId, itemsDict, maxItemId = datautil.prepareUserItemDicts(trainData)
valUserDict, valItamDict = datautil.prepareValidationUserItemDicts(valData, usersDict, itemsDict)

userIds = tf.placeholder(tf.int32, shape=None)
itemIds = tf.placeholder(tf.int32, shape=None)
scores = tf.placeholder(tf.float32, shape=None)

m=0.001
U = tf.Variable(tf.random_uniform([int(maxUserId), D], minval=-m, maxval=m), dtype=tf.float32, name="U")
V = tf.Variable(tf.random_uniform([int(maxItemId), D], minval=-m, maxval=m), dtype=tf.float32, name="V")

U_embed = tf.gather(U, userIds)
V_embed = tf.gather(V, itemIds)

pScores = tf.reduce_sum(tf.multiply(U_embed, V_embed), axis=1)

l1 = (1 / (2 * sigma)) * tf.reduce_sum(tf.square(tf.subtract(scores, pScores)))
l2 = (1 / (2 * sigma_u)) * tf.reduce_sum(tf.multiply(U_embed, U_embed))
l3 = (1 / (2 * sigma_v)) * tf.reduce_mean(tf.multiply(V_embed, V_embed))

loss = l1 + l2 + l3

tf.summary.scalar("valid loss", loss)
merge = tf.summary.merge_all()
filewriter1 = tf.summary.FileWriter("log/plot1")
filewriter2 = tf.summary.FileWriter("log/plot2")

lr = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdamOptimizer(lr)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
numSteps = 5000
adaptiveLR = 0.01
patients = 1000
lrDecayStep = 10
with tf.Session() as session:
    saver = tf.train.Saver()
    modelPath = "./models/m_{}.ckpt".format(numSteps)
    if False and os.path.exists(modelPath + '.meta'):
        saver.restore(session, modelPath)
    else:
        bestValidLoss = np.inf
        bestValidRMSE = np.inf
        yc = 0
        session.run(init)
        for step in range(numSteps):
            _, lossTrain, l1val, l2val, l3val, b = session.run([train, loss, l1, l2, l3, merge],
                                          feed_dict={userIds: trainData[:, 0].astype(np.int), itemIds: trainData[:, 1].astype(np.int),
                                                     scores: trainData[:, 2],
                                                     lr: adaptiveLR})

            filewriter1.add_summary(b, step)
            filewriter1.flush()

            lossValid, pScoresVal, b = session.run([loss, pScores, merge],
                                                   feed_dict={userIds: valData[:, 0].astype(np.int), itemIds: valData[:, 1].astype(np.int),
                                                              scores: valData[:, 2]})

            filewriter2.add_summary(b, step)
            filewriter2.flush()
            rmseValid = scoreutil.calcRMSE(valData[:, 2], pScoresVal)
            print('step {}/{}: '.format(step, numSteps))
            print('   Train loss : {}'.format(lossTrain))
            print('   Valid loss : {}'.format(lossValid))
            print('   Adaptiv LR : {}'.format(adaptiveLR))
            print('   Valid RMSE : {}'.format(rmseValid))
            print(l1val, l2val, l3val);

            if rmseValid < bestValidRMSE:
                bestValidRMSE = rmseValid
                modelPath = saver.save(session, modelPath)
                adaptiveLR *= 100/95
                print('model saved in {}'.format(modelPath))
            else:
                yc += 1
                if yc % lrDecayStep == 0:
                    adaptiveLR *= 0.95
            if yc == patients:
                break

    saver.restore(session, modelPath)
    lossValid, pScoresVal = session.run([loss, pScores],
                                        feed_dict={userIds: valData[:, 0].astype(np.int), itemIds: valData[:, 1].astype(np.int),
                                                   scores: valData[:, 2]})

    lossTrain, Uval, Vval = session.run([loss, U, V], feed_dict={userIds: trainData[:, 0].astype(np.int), itemIds: trainData[:, 1].astype(np.int),
                                             scores: trainData[:, 2]})

    print('Best Train loss : {}'.format(lossTrain))
    print('Best Valid loss : {}'.format(lossValid))
    print('Best Valid RMSE : {}'.format(scoreutil.calcRMSE(valData[:, 2], pScoresVal)))

    np.save('bestU.npy', Uval)
    np.save('bestV.npy', Vval)

    scoreutil.printTopDiffs(valData, pScoresVal, 10)

    pScoresTrainVal = session.run(pScores, feed_dict={userIds: trainData[:, 0].astype(np.int), itemIds: trainData[:, 1].astype(np.int)})
    print('train rmse: {}'.format(scoreutil.calcRMSE(trainData[:, 2], pScoresTrainVal)))

    pScoresVal = session.run(pScores, feed_dict={userIds: valData[:, 0].astype(np.int), itemIds: valData[:, 1].astype(np.int)})
    print('valid rmse: {}'.format(scoreutil.calcRMSE(valData[:, 2] + meanScores, pScoresVal + meanScores)))

    pScoresVal = session.run(pScores, feed_dict={userIds: testData[:, 0].astype(np.int), itemIds: testData[:, 1].astype(np.int)})

    with open('test-scores.txt', 'w') as f:
        print(meanScores)
        for i in range(10):
            print(pScoresVal[i])
        for s in pScoresVal:
            f.write(str(s + meanScores) + '\n')
        f.close()
