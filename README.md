# 1. 프로젝트 소개

## 프로젝트 명
   **뇌 영상 분석 및 뇌종양 진단 학습 모델 개발**
<br/>
## 개요
 뇌 조직에 풍부하게 존재하는 신경교세포에서 악성 종양이 발생할 경우, 환자의 생명을 크게 위협할 수 있다. 교모세포종 (Glioblastoma, GMB) 으로 알려진 이 질환은 성인에게서 발생하는 뇌종양 중 가장 흔한 유형이며 치명률이 가장 높다.
 지금까지 임상 종양 진단은 순수 형태학적-조직병리학적 분류를 많이 따랐지만 현재 개정된 세계보건기구(WHO)는 이를 통합 분자-세포유전학적 특성으로 전환했다. 이에 따라 손상된 DNA를 수정하는 효소인 O[6]-methylguanine-DNA methyltransferase(MGMT) 프로모터에 메틸화가 되어 있을 경우 환자들의 예후가 좋아진다는 결과가 보고되었으며, 이는 화학 요법에 대한 반응성의 강력한 예측 변수인 것으로 나타났다.
 이번 과제에서 영상화를 통해 MGMT 프로모터 메틸화 상태를 분류하여 교모세포종 환자들의 치료 결정 및 생명 연장에 이바지할 수 있을 것으로 기대된다.
 <br/>
 ## 목적
 본 졸업 과제는 뇌 MRI 영상 데이터를 사용해 MGMT 프로모터 메틸화 상태를 분류하는 딥러닝 모델을 개발하는 것을 목표로 한다.
 <br/>
# 2. 팀 소개
|이름|이메일|:역할:|
|:----:|:-------:|:-------|
|이섬재|jmlee2523@pusan.ac.kr|- 학습 모델 개발 및 최적화 <br/>- 데이터 전처리 모듈 수정 <br/>- 예측 결과 도출 및 시각화|
|최연희|kan1428@pusan.ac.kr|- 학습 모델 테스트 <br/>- 학습 모델 최적화 및 오류 수정 <br/>- 데이터 전처리 및 시각화|
|한혜린|applecheeze808@gmail.com|- 학습 모델 테스트 <br/>- 학습 모델 최적화 및 오류 수정 <br/>- 데이터 전처리|
<br/>

# 3. 구성도
<br/>

![딥러닝 모델 개발 구성도](https://user-images.githubusercontent.com/105485617/195505793-3668c641-d472-4404-9d9c-9ad7f1eb2156.png)

 뇌 MRI 영상 데이터 셋을 통해 모델을 새로 학습시킨다. Radiological Society of North America 에서 제공하는 585명의 mpMRI 환자 데이터를 사용하여 데이터 셋을 구축한다.
 <br/>

 2D 모델과 3D 모델에 데이터를 입력하기 위해 원본 데이터를 전처리한다. 원본 데이터는 환자 한 명 당 224 * 224 크기의 MRI 사진 수십 장으로 이루어져 있다. 2D 모델에 사용할 데이터의 경우 원본 MRI 데이터 사진을 64 * 64 크기로 압축하여 사용하고, 3D 모델에 사용할 데이터의 경우 원본 MRI 데이터를 한 개의 64 * 64 * 64 크기 파일로 압축하여 사용한다.
 <br/>

 전처리된 2D 데이터와 3D 데이터를 각각 2D 모델과 3D 모델에 학습시킨다. 이후 데이터의 편중을 막고 모델의 올바른 일반화 성능을 위해 Stratified 5 Fold Cross Valication 을 적용한다. 교차 검증을 진행함과 동시에 각각의 Fold 가 학습될 때마다 Fold 별 Confusion Matrix 를 출력한다. 5개의 Confusion Matrix 로부터 얻어진 성능 평가 지표 (Accuracy, Precision, Recall, F1 Score) 를 사용해 Average, Std 값을 구한다.
 <br/>

 이 과정을 통해 결과적으로 뇌 MRI 영상에 대한 MGMT 프로모터 메틸화 상태를 분류해보고자 한다.

<br/>

# 4. 소개 및 시연 영상
- 프로젝트 소개 유투브 링크 달기
<br/>

# 5. 사용법

### 1. 파일 다운로드
* 먼저 git clone을 통하여 repository의 파일을 다운받습니다. 2D 모델 학습을 위해서는 train_2D_model.py를, 3D 모델 학습을 위해서는 train_3D_model.py를, 3D 모델 학습에 필요한 전처리 데이터를 직접 준비하고 싶을 경우 resizing_to_3D.py를 준비합니다. 미리 준비된 3D 전처리 데이터를 사용하고 싶을 경우 3D_resized_image directory의 파일을 다운받습니다.
* 2D 모델 학습과 3D 전처리 데이터 코드의 실행을 위해 kaggle(https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/data)에서 제공하는 원본 데이터를 다운받습니다.

### 2. 환경 설정
* 개발 환경은 python 3.6.9입니다.
* 코드 실행에 필요한 python library를 다운받습니다.
```
pip install numpy
pip install pandas
pip install pydicom
pip install matplotlib
pip install seaborn
pip install tensorflow_hub
pip install tensorflow
pip install scikit-learn
pip install tqdm
```
