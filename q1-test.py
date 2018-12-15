import tensorflow as tf
import numpy as np
import datautil
import scoreutil
import os

trainDataFile = './dataset/train_user_item_score.txt'
# testDataFile = './dataset/validation_user_item_score.txt'
testDataFile = './96301773_in.txt'

# hyper parameters
sigma = 1.0
sigma_u = 1e-1
sigma_v = 1.0
D = 60
trainData = datautil.readCsvData(trainDataFile)
testData = datautil.readCsvData(testDataFile)

meanScores = np.mean(trainData, axis=0)[2]

usersDict, maxUserId, itemsDict, maxItemId = datautil.prepareUserItemDicts(trainData)

userIds = tf.placeholder(tf.int32, shape=None)
itemIds = tf.placeholder(tf.int32, shape=None)

m = 0.001
U = tf.Variable(tf.random_uniform([int(maxUserId), D], minval=-m, maxval=m), dtype=tf.float32, name="U")
V = tf.Variable(tf.random_uniform([int(maxItemId), D], minval=-m, maxval=m), dtype=tf.float32, name="V")

U_embed = tf.gather(U, userIds)
V_embed = tf.gather(V, itemIds)

pScores = tf.reduce_sum(tf.multiply(U_embed, V_embed), axis=1)

init = tf.initialize_all_variables()
numSteps = 5000

with tf.Session() as session:
    saver = tf.train.Saver()
    modelPath = "./models/m_{}.ckpt".format(numSteps)

    saver.restore(session, modelPath)
    pScoresVal = session.run(pScores, feed_dict={userIds: testData[:, 0], itemIds: testData[:, 1]})

    with open('test-scores.txt', 'w') as f:
        for s in pScoresVal:
            f.write('{}\n'.format(s + meanScores))
        f.close()
    # scoreutil.printTopDiffs(testData, pScoresVal, 10)
