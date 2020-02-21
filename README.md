# 문제 정의  
알파벳 모음 a, e, i, o, u의 classification 구현, 그리고 구분하기 힘든 모음을 찾기  
- 2020, 02, 22 개선사항  
  Tensorflow 1.14 코드를 Tensorflow 2.1.0 코드로 업데이트  
  의미가 모호할 수 있는 변수명 개선  
  데이터셋과 모델의 시각화  
# 데이터셋의 구성과 전처리  
1명의 사람들이 a, e, i, o, u 알파벳 모음을 발음한다.  
총 75명의 사람들이 녹음에 참여하여 발음 데이터 375개의 데이터를 구성  
각 음성 데이터를 librosa 라이브러리에서 Mel Frequency Cepstral Coefficient 방법을 적용  
Reference  
- http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/  
- https://medium.com/@jonathan_hui/speech-recognition-feature-extraction-mfcc-plp-5455f5a69dd9  
- https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0  
  
# 모델 구성과 학습 조건  
- tf.keras.optimizers.Adam(0.0025)  
- Epoch : 30  
- Batch size : 8  
- Validation : 0.2  
- 동일한 형태의 Convolution Network 사용  
  
#  학습 결과
Mel Frequency   : 80%  
Mel Spectrogram : 86%  
