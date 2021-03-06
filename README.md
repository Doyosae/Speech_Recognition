# 문제 정의  
알파벳 모음 a, e, i, o, u의 classification 구현, 그리고 구분하기 힘든 모음을 무엇인지 판별  
- 2020, 02, 22 개선사항  
  Tensorflow 1.14 코드를 Tensorflow 2.1.0 코드로 업데이트  
  의미가 모호할 수 있는 변수명 개선  
  데이터셋과 모델의 시각화  
  
  
# 데이터의 구성과 전처리  
1명의 사람이 a, e, i, o, u 알파벳 모음을 발음. 총 75명의 사람들이 녹음에 참여하여 발음 데이터 375개의 데이터를 구성  
음성 데이터를 mel frequency cepstral coefficient 방법을 적용, .feature.mfcc 메서드를 사용  
또 다른 전처리는 .feature.melspectorgram 메서드를 사용  
- [Reference 1](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)  
- [Reference 2](https://medium.com/@jonathan_hui/speech-recognition-feature-extraction-mfcc-plp-5455f5a69dd9)  
- [Reference 3](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0)  
  
  
# 전처리된 데이터의 Plot 비교 (mfcc and melspectogram)
![Mel Frequency](https://github.com/Doyosae/Speech_Recognition/blob/master/image/MFCC.png)  
![Mel Spectrogram](https://github.com/Doyosae/Speech_Recognition/blob/master/image/melspec.png)  
- .feature.mfcc에서 n_mfcc는 mfcc 계수의 갯수, n_mfcc = 20으로 설정. 가장 많이 사용한느 옵션 범위 중 최솟값  
- .feature.melspectogram에서 n_mels는 9로 설정. mfcc 결과값과의 배열 사이즈를 동일하게 하기 위한 값  
  
  
# 모델의 사양과 학습 조건 그리고 구조  
- Tensorflow 2.1.0  
- tf.keras.optimizers.Adam(0.0025)  
- Epoch : 30  
- Batch size : 8  
- Validation : 0.2  
- 동일한 형태의 Convolution Network 사용  
- 훈련 데이터는 360개, 검증 데이터는 15개로 Split  
  
![Model](https://github.com/Doyosae/Speech_Recognition/blob/master/image/Model.PNG)  
  
  
# 모델의 학습 곡선 및 학습 결과  
위의 사진은 MFCC learning curve, 아래 사진은 Mel Spectrogram learning curve.  
![Mel Freqeuncy](https://github.com/Doyosae/Speech_Recognition/blob/master/image/mfcc_learning.png)  
![Mel Spectrogram](https://github.com/Doyosae/Speech_Recognition/blob/master/image/melspec_Learning.png)  
- Mel Frequency   : 80%  
- Mel Spectrogram : 86%  
  
  
# 모델의 한계점 및 개선점  
- 360개의 데이터는 모델의 성능을 높이기에 매우 부족한 데이터 수  
- 두 가지 방법을 생각해볼 수 있다. GAN으로 ata Augmentation 시도 또는 Few Shot Learning을 시도  
- Test 데이터셋에 대해 가장 잘 일반화할 수 있는 적당한 Split은 몇인지 알아볼 필요성  
