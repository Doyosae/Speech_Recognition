# (87%) 학습 360개, 검사 15개, 컨볼루션 + 다층 신경망, MFCC 40개.ipynb
# 음소 하나 당 MFCC는 40개의 벡터로, 총 18개 프레임
# 음소를 아에이오우 순서로 섞어서 375세트의 한 집합으로 만듬
# 이 중 앞의 360개는 훈련 데이터, 나머지 15개는 검사 데이터로 활용
# 4개의 컨볼루션 신경망과 2개의 멀티 레이어로 구성

import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

Temp_Array = [] # 리스트에서 리스트를 전달할 인자
INPUT_Data = [] # X 플레이스 홀더에 들어갈 리스트
VALID_Data = [] # Y 플레이스 홀더에 들어갈 리스트

Alpha = [1, 0, 0, 0, 0]
Epsilon = [0, 1, 0, 0, 0]
Iota = [0, 0, 1, 0, 0]
Omicron = [0, 0, 0, 1, 0]
Upilon = [0, 0, 0, 0, 1]
# 각각의 발음 기호에 One - Hot - Encoding으로 라벨링을 한다.

Wave_1 = np.loadtxt ('75Raw_a.txt')
Wave_2 = np.loadtxt ('75Raw_e.txt')
Wave_3 = np.loadtxt ('75Raw_i.txt')
Wave_4 = np.loadtxt ('75Raw_o.txt')
Wave_5 = np.loadtxt ('75Raw_u.txt')
# 텍스트로 저장한 사운드 파일을 모두 읽어들인다

DATA_1 = np.reshape(Wave_1, (-1, 20000))  ## 20000개의 75명 데이터로 나누기
DATA_2 = np.reshape(Wave_2, (-1, 20000))  ## 20000개의 75명 데이터로 나누기
DATA_3 = np.reshape(Wave_3, (-1, 20000))  ## 20000개의 75명 데이터로 나누기
DATA_4 = np.reshape(Wave_4, (-1, 20000))  ## 20000개의 75명 데이터로 나누기
DATA_5 = np.reshape(Wave_5, (-1, 20000))  ## 20000개의 75명 데이터로 나누기
# 읽어들인 텍스트 파일의 정보를 20,000개 단위로 끊는다. 한 파일당 150만개가 있고 20000만개로 나누면 75줄이 된다.
# 실제로 한 발성의 텍스트 파일에 저장된 사람은 75명이다. 1명 = 20,000개

for k in range (75):
    MFCC_Data_1 = librosa.feature.mfcc(DATA_1[k], sr = 44100, n_mfcc = 40, n_fft = 2048, hop_length = 1024)[:, :-2].T
    MFCC_Data_2 = librosa.feature.mfcc(DATA_2[k], sr = 44100, n_mfcc = 40, n_fft = 2048, hop_length = 1024)[:, :-2].T
    MFCC_Data_3 = librosa.feature.mfcc(DATA_3[k], sr = 44100, n_mfcc = 40, n_fft = 2048, hop_length = 1024)[:, :-2].T
    MFCC_Data_4 = librosa.feature.mfcc(DATA_4[k], sr = 44100, n_mfcc = 40, n_fft = 2048, hop_length = 1024)[:, :-2].T
    MFCC_Data_5 = librosa.feature.mfcc(DATA_5[k], sr = 44100, n_mfcc = 40, n_fft = 2048, hop_length = 1024)[:, :-2].T
    print ('MFCC 변환할 때마다의 크기', np.shape(MFCC_Data_1))
    # 2만개를 20개 벡터 18개 성분으로 mfcc 할 때마다
    
    MFCC_Data_1 = np.reshape(MFCC_Data_1, (720))
    MFCC_Data_2 = np.reshape(MFCC_Data_2, (720))
    MFCC_Data_3 = np.reshape(MFCC_Data_3, (720))
    MFCC_Data_4 = np.reshape(MFCC_Data_4, (720))
    MFCC_Data_5 = np.reshape(MFCC_Data_5, (720))
    print ('모든 성분을 하나로 쭉 나열한 다음에', np.shape(MFCC_Data_1))
    # 총 360개 성분이므로 1열로 나열한 후에
    
    Temp_Array.extend(MFCC_Data_1)
    Temp_Array.extend(MFCC_Data_2)
    Temp_Array.extend(MFCC_Data_3)
    Temp_Array.extend(MFCC_Data_4)
    Temp_Array.extend(MFCC_Data_5)
    print ('All SIze after reshape', np.shape(Temp_Array))
    # Temp_Array 리스트에 360개씩 추가한다.


    

for i in range (75) : # 학습 순서에 맞게 정답 레이블들을 작성하는 구문
    VALID_Data.extend(Alpha)
    VALID_Data.extend(Epsilon)
    VALID_Data.extend(Iota)
    VALID_Data.extend(Omicron)
    VALID_Data.extend(Upilon)
    # 이 구문의 결과는 aeiou 라벨1, aeiou 라벨2, aeiou 라벨3 라벨4 라벨5 ... 이렇게 간다.
    
INPUT_Data = np.reshape(Temp_Array, (375, 720))
VALID_Data = np.reshape(VALID_Data, (-1, 5))

print ('INPUT_Data의 크기', np.shape (INPUT_Data))
print ('VALID_Data의 크기 ', np.shape (VALID_Data))




INPUT_Data = np.reshape(INPUT_Data, (-1, 720))
VALID_Data = np.reshape(VALID_Data, (-1, 5))

print ('INPUT_Data의 크기', np.shape (INPUT_Data))
print ('VALID_Data의 크기 ', np.shape (VALID_Data))
print ('INPUT_Data의 크기', np.shape (INPUT_Data[0]))
print ('VALID_Data의 크기 ', np.shape (VALID_Data[0]))

TrainingInputData = INPUT_Data [0:360]
TrainingLabelData = VALID_Data [0:360]
ValidInputData = INPUT_Data [360 : 375]
ValidLabelData = VALID_Data [360 : 375]
# 준비된 데이터 세트 375개 중 360개는 학습 데이터로, 15개는 검사 데이터로 활용하기 위해서 리스트를 자름

print ('INPUT_Data의 크기', np.shape (TrainingInputData))
print ('VALID_Data의 크기 ', np.shape (TrainingLabelData))
print ('INPUT_Data의 크기', np.shape (ValidInputData))
print ('VALID_Data의 크기 ', np.shape (ValidLabelData))





import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

X = tf.placeholder("float", shape = [None, 720])
Y = tf.placeholder("float", shape = [None, 5])
ReshapingData = tf.reshape (X, [-1, 18, 40, 1])
print(ReshapingData)

Weight1= tf.Variable (tf.truncated_normal (shape = [2, 2, 1, 20], stddev = 0.1)) # 2 by 2 필터의 1채널 20개
Bias1 = tf.Variable (tf.truncated_normal (shape = [20], stddev = 0.1)) # 20개의 편향도
Window1 = tf.nn.conv2d (ReshapingData, Weight1, strides = [1, 1, 1, 1], padding = 'SAME') + Bias1
Activation1 = tf.nn.relu (Window1) # 활성화 함수는 Relu
Pooling1 = tf.nn.max_pool (Activation1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
print ("컨볼루션 1층에서 풀링 사이즈", Pooling1)
# 여기까지 컨볼루션 1단계에서 시행하는 연산

Weight2= tf.Variable (tf.truncated_normal (shape = [2, 2, 20, 40], stddev = 0.1)) # 2 by 2 필터의 1채널 40개
Bias2 = tf.Variable (tf.truncated_normal (shape = [40], stddev = 0.1)) # 필터에 대응하는 편향도
Window2 = tf.nn.conv2d (Pooling1, Weight2, strides = [1, 1, 1, 1], padding = 'SAME') + Bias2
Activation2 = tf.nn.relu (Window2)
Pooling2 = tf.nn.max_pool (Activation2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
print ("컨볼루션 2층에서 풀링 사이즈", Pooling2)
# 여기까지 컨볼루션 2단계에서 시행하는 연산

Weight3= tf.Variable (tf.truncated_normal (shape = [2, 2, 40, 80], stddev = 0.1)) # 2 by 2 필터의 80개
Bias3 = tf.Variable (tf.truncated_normal (shape = [80], stddev = 0.1)) # 80개의 필터에 대응하는 편향도
Window3 = tf.nn.conv2d (Pooling2, Weight3, strides = [1, 1, 1, 1], padding = 'SAME') + Bias3
Activation3 = tf.nn.relu (Window3)
Pooling3 = tf.nn.max_pool (Activation3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
print ("컨볼루션 3층에서 풀링 사이즈", Pooling3)

Weight4 = tf.Variable (tf.truncated_normal (shape = [2, 2, 80, 160], stddev = 0.1)) # 2 by 2 필터의 1채널 160개
Bias4 = tf.Variable (tf.truncated_normal (shape = [160], stddev = 0.1)) # 160개의 필터에 대응하는 편향도
Window4 = tf.nn.conv2d (Pooling3, Weight4, strides = [1, 1, 1, 1], padding = 'SAME') + Bias4
Activation4 = tf.nn.relu (Window3)
Pooling4 = tf.nn.max_pool (Activation4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
print ("컨볼루션 4층에서 풀링 사이즈", Pooling4)



# All Layer, three convolution acculate
# Convolution Exit, Next to Fully Connected Layer

POOLING3_With_Flat = tf.reshape(Pooling3, [-1, 3*5*80]) # 컨볼루션 망을 지나온 Pooling을 쫙 펼친다.
FullyConnectedWeight1  = tf.Variable (tf.truncated_normal (shape = [3*5*80, 1200], stddev = 0.1))
FullyConnectedBias1 = tf.Variable (tf.truncated_normal (shape = [1200], stddev = 0.1))
FullyConnectedActivation1 = tf.nn.relu(tf.matmul(POOLING3_With_Flat, FullyConnectedWeight1) + FullyConnectedBias1)
# 쭈우우욱 퍌친 텐서를 쭈욱 펼치고 Fully Connected 1층 레이어에서 연산

FullyConnectedWeight2 = tf.Variable (tf.truncated_normal (shape = [1200, 5], stddev = 0.1))
FullyConnectedBias2 = tf.Variable (tf.truncated_normal (shape = [5], stddev = 0.1))
Hypothesis = tf.matmul(FullyConnectedActivation1, FullyConnectedWeight2) + FullyConnectedBias2
# Fully Connected 2층 레이어에서 연산

Loss = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits (labels = Y, logits = Hypothesis))
Train = tf.train.AdamOptimizer(0.003).minimize(Loss)

CorrectPrediction = tf.equal (tf.argmax (Hypothesis, 1), tf.argmax(Y, 1))
Accuracy = tf.reduce_mean (tf.cast (CorrectPrediction, tf.float32))


with tf.Session() as sess :
    print ("............ Go!")
    sess.run (tf.global_variables_initializer())
    
    for k in range (200) :
        _, LossPrint = sess.run ([Train, Loss], feed_dict = {X : TrainingInputData, Y : TrainingLabelData})
        
        if k % 10 == 0 :
            print ('Epoch', '%d' %(k + 10))
            print ('학습 데이터로 계산한 손실도', '%.10f' %LossPrint)
            Predict = sess.run (Accuracy, feed_dict = {X : ValidInputData, Y : ValidLabelData})
            print ('검사 데이터로 예상한 정확도', '%.3f' %(100*Predict), '%')
