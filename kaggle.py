import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from PIL import Image , ImageDraw
from sklearn.preprocessing import *

import time
import ast
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        fpath = os.path.join(dirname, filename)
        
print(dirname)
print(filenames[0])

bar = '□□□□□□□□□□'
sw = 1
def percent_bar(array,count,st_time):   #퍼센트를 표시해주는 함수
    global bar
    global sw
    length = 340
    percent = (count/length)*100
    spend_time = time.time()-st_time
    if count == 1 :
        print('preprocessing...')
    print('\r'+bar+'%3s'%str(int(percent))+'% '+str(count)+'/340','%.2f'%(spend_time)+'sec',end='')
    if sw == 1 :
        if int(percent) % 10 == 0 :
            bar = bar.replace('□','■',1)
            sw = 0
    elif sw == 0 :
        if int(percent) % 10 != 0 :
            sw = 1
            
def check_draw(img_arr) :
    for i in range(len(img_arr[k])):
        img = plt.plot(img_arr[k][i][0],img_arr[k][i][1])
        plt.scatter(img_arr[k][i][0],img_arr[k][i][1])
    plt.xlim(0,256)
    plt.ylim(0,256)
    plt.gca().invert_yaxis()
    
def make_img(img_arr) :
    image = Image.new("P", (256,256), color=255)
    image_draw = ImageDraw.Draw(image)
    for stroke in img_arr:
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i], 
                             stroke[1][i],
                             stroke[0][i+1], 
                             stroke[1][i+1]],
                            fill=0, width=5)
    return image

def preprocessing(filenames, test= False) :
    X= []
    Y= []
    class_label = []
    st_time = time.time()
    class_num = 340
    Y_num = 0
    for fname in filenames :
        percent_bar(filenames,Y_num+1,st_time)
        df = pd.read_csv(os.path.join(dirname,fname))
        df['word'] = df['word'].replace(' ','_',regex = True)
        class_label.append(df['word'][0])
        keys = df.iloc[:1000].index
        #print(len(keys))
        drawing = [ast.literal_eval(i) for i in df.loc[keys,'drawing'].values]
        for draw_num in range(len(keys)) :
            img = make_img(drawing[draw_num])
            img = np.array(img.resize((32,32))).reshape(32,32,1)
            X.append(img)
            Y.append(Y_num)
        Y_num += 1
    tmpx = np.array(X)

    Y = np.array([[i] for i in Y])
    enc = OneHotEncoder(categories='auto')
    enc.fit(Y)
    tmpy = enc.transform(Y).toarray()
    
    return tmpx , tmpy , class_label

X_train , Y_train , class_label = preprocessing(filenames)
print('\n',X_train.shape, Y_train.shape, '\n',class_label)
#df.head()
#print(drawing[0])
#img = make_img(drawing[1])
#plt.imshow(img)

import tensorflow.compat.v1 as tf
tf.disable_eager_execution() #like tensorflow v1 activate

learning_rate = 0.001
training_epochs = 30
batch_size = 100

X = tf.placeholder(tf.float32, [None, 32, 32 , 1], name='input')
Y = tf.placeholder(tf.float32, [None, 340], name='output')
keep_prob = tf.placeholder(tf.float32, name = 'dropout')

W0 = tf.Variable(tf.random_normal([3, 3, 1, 16], stddev=0.01),name='w0')
L0 = tf.nn.conv2d(X, W0, strides=[1, 1, 1, 1], padding='SAME')
#32x32x16
L0 = tf.nn.relu(L0)
L0 = tf.nn.max_pool(L0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#16x16x16


W1 = tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.01),name='w1')
L1 = tf.nn.conv2d(L0, W1, strides=[1, 1, 1, 1], padding='SAME')
# 16x16x32
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 8x8x32

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01),name='w2')
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
#8x8x64
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#4x4x64

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev =0.01),name='w3')
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
#2x2x128
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#2x2x128
L3 = tf.nn.dropout(L3, keep_prob)

L3_flat = tf.reshape(L3,[-1, 2 * 2 * 128])
# 최종 출력값 L3 에서의 출력 256개를 입력값으로 받아서 0~339 레이블인 340개의 출력값을 만듭니다.
W4 = tf.Variable(tf.random_normal([2 * 2 * 128, 340], stddev=0.01),name='w4')
hypothesis = tf.matmul(L3_flat, W4)
hypothesis = tf.identity(hypothesis,'hypothesis')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, 
                                                              labels = Y))    #costfunction

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

target1 = tf.nn.softmax(hypothesis)

#saver = tf.train.Saver()
st_time = time.time()
ing_time = st_time
chk_time = 0
print('Learning Started!!')

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs) :
        avg_cost = 0
        total_batch = int(len(X_train)/ batch_size)
        
        for  i in range(total_batch) :
            #batch_xs, batch_ys = 
            #batch_xs = batch_xs.reshape(-1,28,28,1)
            feed_dict = {X:X_train, Y:Y_train , keep_prob: 1}
            cost_val, _ = sess.run([cost,optimizer],feed_dict=feed_dict)
            avg_cost += cost_val / total_batch
            
        correct_prediction2 = tf.equal(tf.argmax(hypothesis),tf.argmax(Y))
        accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2,tf.float32))
        ex_time2 = time.time() - st_time - chk_time
        print('Epoch :', '%04d'%(epoch+1),'cost = ','{:9f}'.format(avg_cost),
              'Accuracy = ',sess.run(accuracy2,feed_dict={X:X_test,Y:Y_test, keep_prob:1}),
              'Time = ',ex_time2,'sec')
        
        #chk_time = ex_time2
    print('Learning Finished!!')
    
    #saver.save(sess,'savedmodel/class30_test_cnn_model.ckpt', global_step = 1000)
    #print('model saved!!')
    correct_prediction = tf.equal(tf.argmax(hypothesis),tf.argmax(Y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    accuracy2 = tf.cast(correct_prediction,tf.float32)
    
    ex_time = time.time() - st_time
    
    print('Excute Time : ', int(ex_time) ,'sec')
    print('Total Accuracy : ', sess.run(accuracy, feed_dict={X:X_test,
                                                       Y:Y_test, keep_prob:1}))
    
    r = random.randint(0,10)
    parts_idx = sess.run(tf.argmax(Y_test[r:r+1]))[0]
    parts_pre_idx = sess.run(tf.argmax(hypothesis),
                             feed_dict={X:X_test[r:r+1],
                                        keep_prob: 1 })[0]
    print('Label : ',sess.run(tf.argmax(Y_test[r:r+1])))
    print('Parts_Label :',parts_class[parts_idx])
    #print('Parts_class : ', parts_class)
    print('Prediction : ',sess.run(tf.argmax(hypothesis),
                                   feed_dict={X:X_test[r:r+1],
                                              keep_prob: 1}))
    print('Parts_prediction : ',parts_class[parts_pre_idx])
    
    #print('Prediction : ',sess.run(target1,
    #                               feed_dict={X:X_test[r:r+1],
    #                                          keep_prob: 1}))

plt.imshow(X_test[r:r+1].reshape(32,32,1),cmap = 'Greys',
           interpolation='nearest')
plt.show()