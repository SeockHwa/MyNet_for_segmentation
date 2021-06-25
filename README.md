# ERFNet in Colab



## 참고
PyTorch를 이용한 ERFNet에 대한 내용은 https://github.com/Eromera/erfnet_pytorch 을 이용하였다. git에 있는 파일들을 이용해서 colab에서 돌릴수 있도록 변경하고 학습을 진행하여보았다. 

ERFNet in PyTorch : https://github.com/Eromera/erfnet_pytorch 

참고 논문

http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf

http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf



## ERFNet

![image](/uploads/fd2d6377f2deb692632c10b9260a9917/image.png)

자율주행은 도로, 교통표지, 조명, 차량, 보행자 인식 등 인식 문제에서 복잡한 해법이 필요한 난제이다. 일반적인 접근 방식은 객체를 독립적으로 찾는 것을 목표로 하는 패치 기반 인식에 초점을 맞추었지만 각각의 객체들은 독립적인 것이 아니기에 개별적으로 다뤄서는 안되고 상호 관계와 맥락을 고려햐여 작업을 한번에 진행해야 한다. 이미지 분할에 있어서 픽셀 수준에서 직접 다양한 객체 클래스를 분류하는 것을 목적으로 한다.


처음에 이미지 분류 작업을 위해 설계된 ConvNet은 오류율이 매우 낮은 이미지에서 픽셀 단위와 end-to-end로 여러 객체 범주를 분류함으로써 segmentation에서 인상적인 가능성을 보여주었다. 최근 연구에는 이런 구조에 의해 얻어진 정확성들에 있어서 많은 발전을 이루었다. 그러나 네트워크의 복잡성을 희생하여 가능하지 않았던 부분들에 대한 정확도를 높였다. 한편으로는 일부 연구는 실시간 세분화에 도달할 수 있는 모델을 제시하여 효율성에 중점을 두었지만 정확성이 떨어진다.


여기서는 이러한 측면 중 하나에만 의존하지 않고 최고의 효율성과 정확성을 달성하는 모델을 제시한다. 새로운 residual block을 기반하여 factorized convolution을 사용한 네트워크는 현대 GPU에서 실시간 작동에 적합한 효율성을 유지하고 성능은 최대화하도록 설계하였다. 복잡한 도시 장면에서 매우 다양한 객체 클래스를 분할 하는 방법을 배울 수 있다는 것을 보여주며, 최첨단 상황에서 가장 복잡한 다른 접근법만큼 높은 정확도로 학습할 수 있고, 계산 비용은 훨씬 낮다. 차량에 탑재할 수 있는 임베디드 장치에서도 최신 GPU에서 여러개의 FPS로 실행할 수 있다. 이를 통해 실시간 운행이 가능하면서도 주행장면을 최대한 이해하려는 자율주행차의 인식 문제에 이상적인 솔루션이 될 것이다.

## Model 


과거에 제안되었던 Residual Layer는 ConvNet 디자인의 새로운 트렌드를 열었다. 심층적인 architecture의 저하(degradation) 문제를 피하기 위해 Convolution layer를 재구성함으로써, 많은 양의 layer를 쌓는 네트워크에서 높은 정확도를 달성할 수 있었다. 이 방법은 이미지 분류 문제 및 semantic segmentation 분야에서 최고의 정확도를 얻는 새로운 architecture로 채택하였지만 합리적인 양의 layer를 고려해보면, 더 많은 convolution으로 깊이를 확대한다면 필요한 리소스가 크게 증가하지만 그에 비해 정확도는 약간만 향상되기 때문에 최적은 아니라고 생각된다. 차량에 적용하기 위해서는 계산 리소스는 주요 제한사항이다. 알고리즘은 안정적으로 작동할 뿐만 아니라 빠르게 작동해야하며, 공간제약으로 인해 임베디드 장치에 적합하고, 차량 자율성에 영향을 최소한으로 하기 위해 전력을 낮게 소비해야한다. 

효율적인 Semantic Segmentation을 위한 ConvNet인 ERFNet이 만들어졌다. 두가지 중요한 포인트는 ERFNet은 Skip connection과 Residual 개념을 적용한 1D factorized convolution(기존 2D 방법을 변형)이고 제안된 블록은 인코더와 디코더에 순차적으로 적용한다는 것이다.

* Factorized Residual Layers

![image](/uploads/6d8c78fa0d0edbb2caa610938eb815e1/image.png)

Residual 방식은 많은 convolution layer를 쌓을 때 나타나는 degradation 문제를 해결하였으며, 기존 방식은 (a),(b)와 같다. 두 버전은 유사한 수의 파라미터수와 거의 유사한 정확도를 가진다. 그렇지만 bottlenect은 계산 리소스가 적게 필요하고, 깊이가 증가함에 따라 효율적으로 확장된다. 그러다 non-bottlenect의 ResNet은 bottlenect 버전보다 깊이가 더 높아지기 때문에 정확도가 더 높다. 이를통해 bottlenect은 여전히 성능 저하 문제를 가진다. ERFNet은 non-bottlenect-1D 필터로 구성하였고 이는 기존의 residual module 3X3 convolution 크기를 33%줄이며 계산 효율을 높였다.

세부 디자인은 다음과 같다.

![image](/uploads/98cac9b99332bd348c5fffe3918d63d4/image.png)

![image](/uploads/3e1bc7ed5195a3a71a5b9bc63a6de412/image.png)

## Dataset


semantic segmentation에 널리 채택되어있는 최근 도시 장면의 데이터셋 Cityscapes dataset을 이용하였다. 여기에는 2975개의 train set, 500개의 validation set, 1525개의 test set으로 이루어져있다. gtFine 데이터셋을 이용할 때는 conversor를 돌린 data set을 이용하였다.


cityscapes의 dataset에 대한 내용은 https://github.com/mcordts/cityscapesScripts 를 참고하였다.
cityscapes data의 class 분류는 다음과 같다. 기본적으로 class의 개수는 20개로 진행되어진다.

![image](/uploads/01d25cd3fd73a19a50e1887292098220/image.png)

segmentation에 이용되는 color는 6개로 나뉘어진다.

```
# Class for colors
class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

# Colored value output if colorized flag is activated.
def getColorEntry(val):
    if not isinstance(val, float):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN
```


## Modification

* 코드구현 
나뉘어져 있는 코드들을 colab에서 사용할 수 있도록 수정하였고, 시각적 기능을 주기 위해 visdom 부분의 코드를 추가하여 작성하였다. 
* 데이터 셋
Cityscapes data를 그대로 사용하기에 colab의 저장용량에 알맞지 않아 train 단계에서 학습에 쓰일 데이터는 aachen, bochum, bremen, cologne 이 4개만 이용을 하여 학습을 진행하였다. test에는 berlin, bielefeld, bonn, leverkusen, mainz, munich 이 6개를 이용하였다. validation 에서는 frankfurt, lindau, munster 이 3개를 이용하여 진행하였다. 모든 데이터는 conversor를 거치고 사용을 해야한다.
 
 ```
from __future__ import print_function, absolute_import, division
import os, glob, sys

from cityscapesscripts.helpers.csHelpers import printError
from cityscapesscripts.preparation.json2labelImg import json2labelImg

def main():
    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
        cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
    
    searchFine   = os.path.join( cityscapesPath , "gtFine"   , "*" , "*" , "*_gt*_polygons.json" )
    searchCoarse = os.path.join( cityscapesPath , "gtCoarse" , "*" , "*" , "*_gt*_polygons.json" )

    filesFine = glob.glob( searchFine )
    filesFine.sort()
    filesCoarse = glob.glob( searchCoarse )
    filesCoarse.sort()

    files = filesFine + filesCoarse

    if not files:
        printError( "Did not find any files. Please consult the README." )

    print("Processing {} annotation files".format(len(files)))

    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for f in files:
        dst = f.replace( "_polygons.json" , "_labelTrainIds.png" )

        try:
            json2labelImg( f , dst , "trainIds" )
        except:
            print("Failed to convert: {}".format(f))
            raise

        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()

if __name__ == "__main__":
    main()
```

## Report


* 3월 1~2주차

ERFNet 구현에 앞서 PyTorch와 Python에 대해 공부하는 시간을 가져 프로젝트를 원활하게 진행 할 수 있도록 하였다. Python에 대한 학습은 "나도코딩"의 강의를 참고하여 학습을 진행하였다. PyTorch에 대한 학습은 "모두를 위한 딥러닝 season2"를 참고하여 PyTorch를 이용한 딥러닝 공부를 진행하였다. Python 공부를 통해 numpy tuple 등 다양한 라이브러리 사용 및 문법을 숙지하고 class를 정의하여 활용하는 등 python을 사용하는데 있어 불편함이 없도록 학습을 진행하였다. PyTorch 공부에서는 기본적인 PyTorch에 대한 문법과 개념들을 익힘으로 코드를 읽을때 어려움이 없도록 하였다.

* 3월 3주자 
ERFNet git에 있는 논문 2가지를 읽으며 ERFNet에 대한 기본적인 모델 구조를 알아보고 공부를 하였다. bottleneck과 non-bottleneck에 대해 알아보고, 기존의 convnet과 논문에서 제시하는 convnet의 차이를 학습하여 정리를 진행하였다. Semantic Segmentation에 대한 공부를 진행하고 발표자료를 만들어 정기미팅 때 발표를 진행하였다. 해당 발표자료는 pdf 파일로 첨부하였다.

* 3월 4주차
ERFNet의 코드를 관련 git으로 부터 따와 colab에 적합하게 코드를 수정하였다. 기본적으로 데이터셋은 cityscapes의 dataset을 이용하였는데 처음 학습한 결과 dataset의 크기가 너무 커 크기를 줄여서 사용하였다. 처음 epoch를 작게 잡아 학습을 진행하여 잘 진행되는지 확인하였다.학습하는 과정을 시각화를 하기 위해서 Visdom을 이용하여 시각화하였다. input 사진과 target 사진과 학습하며 추출한 결과물 사진을 보여준다.

```
import numpy as np
from torch.autograd import Variable
from visdom import Visdom

class Dashboard:

    def __init__(self, port):
        self.vis = Visdom(port=port)

    def loss(self, losses, title):
        x = np.arange(1, len(losses)+1, 1)

        self.vis.line(losses, x, env='loss', opts=dict(title=title))

    def image(self, image, title):
        if image.is_cuda:
            image = image.cpu()
        if isinstance(image, Variable):
            image = image.data
        image = image.numpy()

        self.vis.image(image, env='images', opts=dict(title=title))
```

```
! npm install -g localtunnel
# 8097 is the port number I set myself, which can be modified to the port number I want to use
get_ipython().system_raw('python3 -m pip install visdom')
get_ipython().system_raw('python3 -m visdom.server -port 8097 >> visdomlog.txt 2>&1 &')   
get_ipython().system_raw('lt --port 8097 >> url.txt 2>&1 &')   
import time
time.sleep(5)
! cat url.txt
import visdom
time.sleep(5)
vis = visdom.Visdom(port='8097')  
print(vis)
time.sleep(3)
vis.text('testing')
! cat visdomlog.txt
```

![input](/uploads/8fbb4b18e8180a1f7ff90661eeb739db/input.PNG)


![target](/uploads/44176ba91d55673a47c2c5b66366e5a1/target.PNG)


![학습](/uploads/12eb6bb27e5a0d721355cfa1756b4f63/학습.PNG)

* 4월 1주차

학습 후 모델의 학습 결과를 판단하기에 visdom만 사용하여 과정을 표현하면 얼마나 정확하게 segmentation을 이루어 내는지 정확하게 알 수 없기에 수치화해서 텍스트파일로 저장하여 확인하는 것이 필요했다.

```
    savedir = f'{args.savedir}'
    
    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"    
    
    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))


    #TODO: reduce memory in first gpu: https://discuss.pytorch.org/t/multi-gpu-training-memory-usage-in-balance/4163/4        #https://github.com/pytorch/pytorch/issues/1893

    #optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)      ## scheduler 2

    start_epoch = 1
    if args.resume:
        #Must load weights, optimizer, epoch and best value. 
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'

        assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

```

다음과 같은 구현을 통해 학습을 진행하고 그 과정 및 결과를 수치로 나타난 텍스트 파일을 저장되게 하였다. 학습 후 드라이브에 저장되어있는걸 확인할 수 있다.


![image](/uploads/cf75ed6a65785f62de54890d94ce3ad8/image.png)

이 코드가 실행되면 학습과정마다 train loss, test loss, train iou, test iou, learning rate가 epoch 마다 정리된 txt파일이 저장되어 확인할 수 있다. 다음은 epoch 10으로 학습한 결과이다.

* 4월 4주차

연구내용 문서화 작업을 진행하여 추가할 내용과 삭제할 내용 또는 더 공부가 필요한 부분들을 정해 정리하는 시간을 가짐.

* 5월 1주차 

원래의 ERFNet 성능과 데이터 subset 사용했을때의 성능을 비교해서 정리해보기.
성능 향상을 위해 loss에 대해서 상세하게 알아보기

* 5월 2주차

ERFNet과 다른 네트워크의 차이를 분석하고, 모델에 사용한 다양한 개념들에 대해 공부를 진행하였다. ERFNet은 ENet의 구조를 기반하여 만들어진 것이므로 ENet에 대한 상세한 내용도 학습을 진행하였다.

### Downsampling

Downsampling은 이미지의 사이즈를 줄이는 것을 말하며 반대는 Upsampling이다, Downsampling을 진행하면 image의 화질이 안좋아져 자연스럽지 않고 pixel간 차이가 커서 경계가 있어보이게 된다. low pass filter를 사용하여 downsampling을 진행한다면 이미지는 blur해 보이지만 전자의 경우보다 괜찮은 결과를 보여준다. 왼쪽 사진이 원본 가운데가 전자 오른쪽이 후자이다.

![image](/uploads/105e923f46e639778e0f29b1c3a33a9e/image.png)

Downsampling 연산은 더 큰 receptive field를 가지기 때문에 더 많은 context 정보를 얻을 수 있기에 사용한다. 도로장면에서 라이더와 보행자 같은 클래스를 구분하려 할때, 사람의 외모만 학습하여 판단하기보다 맥락을 학습하는 것도 중요하다. Downdampling에는 두가지 단점이 있다. feature map 해상도를 줄이면 가장자리 부분의 공간정보를 손실하게 된고 전체 픽셀 분할은 출력이 입력과 동일한 해상도를 가져아한다
-> 인코더에 의해 생성된 feature map을 추가하여 FCN에서와 같이 max-pooling layer에서 선택된 요소의 인덱스를 저장하고 이를 사용하여 디코더에서 saprse 업 샘플링 된 map을 생성하는 방법을 이용하여 해결한다.
real-time 연산에서 가장 중요한 것은 입력 프레임에 대해 효과적으로 처리하는 것이고, ENet의 처음 두 블록은 입력 크기를 크게 줄이고 작은 특징 맵 세트를 사용하기 위해 Early downs ampling을 진행하였다.


### factorizing filters

 nxn convolution 연산을 nx1 1xn으로 나누어 수행하는데 이는 중복연산을 피할 수 있다. 병목 모듈에서 사용되는 연산은 하나의 큰 convolution layer를 낮은 순위 근사치인 더 작고 간단한 작업으로 분해하는 것으로 볼 수 있다. 이러한 분해 속도를 크게 높이고 매개변수의 수를 크게 줄여 중복성을 줄일 수 있다. 또한 layer 사이에 삽입되는 비 선형 연산 덕분에 계산 기능을 더 풍부하게 만듬


### Dilated convolution 

![image](/uploads/c663b98f366ce82bfcaea44b88d22336/image.png)

Dliated Convolution은 간단히 표현하자면 기본 Convolution 필터가 수용하는 픽셀 사이에 간격을 둔 형태이다. 입력 픽셀 수는 동일하지만 더 넓은 범위에 대한 입력을 수용할 수 있다. 즉 Convolution Layer에 또 다른 파라미터인 Dilation rate를 도입한 것이다. 이는 커널 사이의 간격을 정의하며 dilation rate가 2인 3x3 kernel은 9개의 파라미터를 사용하면서 5x5 커널과 동일한 view를 가지게 된다. 이것은 real-time segmentation 분야에서 주로 사용되는데 넓은 view가 필요하고 여러 convolution이나 큰 kernel을 사용할 여유가 없을 때 사용한다. 이는 적은 계산 비용으로 receptive field 를 늘리는 방법이다. 넓은 receptive field를 갖는 것이 매우 중요하며 이는 더 넓은 context를 고려하여 분류를 수행할 수 있다. feature map을 과도하게 down sampling하지 않고 개선하기 위해 확장된 convolution을 사용한다. 가장 작은 해상도에서 작동하는 convolution 연산을 대체, 추가 비용 없이 IOU 4% 향상되는 효과를 보았다.
 
### regularization 

대부분의 픽셀 단위 분할 데이터 셋은 상대적으로 작기 때문에 신경망과 같은 표현 모델이 빠르게 과적합이 되기 시작하여 weight decay와 stochastic를 시도하여 정확도를 높혔다.


### stochastic pooling 

Pooling이란, sub-sampling을 이용해 feature-map의 크기를 줄이고 위치나 이동에 강인한 특징을 추출하기 위한 방법이다. 가장 일반적인 pooling은 max-pooling과 average-pooling이 있지만 각각은 문제를 가지고 있는데 max-pooling은 윈도우 내의 가장 큰 값을 선택하는 방법으로 학습 데이터에 overfitting 되기 쉬운 단점이 있고 평균값을 취하는 average-pooling은 평균 연산에 의한 강한 자극이 줄어드는 현상이 있으므로 학습 결과가 좋지 않다. 
Stochastic Pooling은 max-pooling과 average-pooling의 문제를 해결하기 위한 방버으로 최대 값 혹은 평균 값 대신 확률에 따라 적절한 activation을 선택한다. 확률은 특정 activation에 전체의 activcation의 합을 나누는 방식으로 구해지며, activation의 값이 높을수록 확률은 커지고 그에 따라 관여도 커진다. 하지만 max-pooling과는 다르게 항상 가장 큰 값이 취해지는 것이 아니라 다른 의미있는 값이 취해질 수 있다는 점이 가장 큰 차이이다.

![image](/uploads/9644f16856f7e1ecb8501cc69cdf0ffe/image.png)

실제 적용을 할 때 Stochastic pooling을 사용하면 성능이 떨어지기 때문에 각각의 activation에 weighting factor로 곱해주고 그 합을 이용하는 방식으로 수행된다. average-pooling과는 다르게 자극이 높을수록 weight 또한 커지므로 단순히 평균을 이용하는 average-pooling의 단점을 보완할 수 있습니다. 또한 Pooling window의 크기만큼 다양한 조합이 가능해지기 때문에 Dropout과 같이 다양한 Network를 학습하는 듯한 model average 효과를 얻을 수 있다.


### residual learning

![image](/uploads/93ae82e82f5e8eb6bcf0e37bbf26ad51/image.png)

왼쪽이 기존의 형태로 H(x)를 최적화 시키는데 목적이 있었다면 오른쪽은 H(x)-x를 얻는것으로 목표를 수정하는 것이다 이렇게 된다면 F(x) = H(x) - x 가 되므로 H(x) = F(x) + x가 됨을 알 수있다. 그래서 입력으로부터 출력으로 바로가는 shortcut이 생긴 것인데, 이는 파라미터가 없이 바로 연결되는 구조라 연산량에서는 덧셈이 추가되는 것 외에는 차이가 없다. 이는 작은 차이로 큰 효과를 발생시키는데 최적의 경우가 되기 위해서 F(x)가 0이 되어야 하기 떄문에 학습할 방향이 미리 결정이 되고 이것이 pre conditioning 구실을 하게 된다. 그렇게 되면 입력의 작은 움직임을 쉽게 검출 할 수 있게 된다. 
정리하자면 residual learning을 통해 얻는 장점은 다음과 같다.
1. 깊은 망도 쉽게 최적화가 가능하다.
2. 늘어난 깊이로 인해 정확도를 개선할 수 있다.

### bottleneck

![image](/uploads/8fd54d47236552b03fc4b8cc801a8ff5/image.png)

3X3 연산을 1X1 3X3 1X1 구조로 차원을 줄였다가 늘림, 이렇게 구성하면 연산시간을 줄일 수 있음 

![image](/uploads/77e106916e3a6f998ab280ab324b33cf/image.png)


ENet에서는 bottleneck을 사용하여 연산량을 줄였지만 성능 저하 문제가 있어서 ERFNet에서는 non bottleenck을 사용한다. 그렇지만 연산량을 줄이기 위해서 2d가 아닌 1d로 진행하는 factorizing filter를 이용하여 중복연산을 피함으로써 연산량을 줄였다.

* 5월 3주차 

ConvNet, ENet, ERFNet 모델들에 대해 공부를 진행한 뒤 모델들의 장단점을 찾으며 성능 향상을 위해 어떠한 방법을 시도할지 공부해보기

처음 이미지 분류 작업을 위해 설계된 ConvNet은 픽셀 단위로 인식하고 End-to-End 방식으로 여러 객체 범주를 분류하면서 많은 가능성과 발전을 보여주며 가능하지 않았던 분야에서도 가능하게 하여 정확도를 높힌 모델이다. 그렇지만 실시간으로 세분화를 이뤄내는 효율성에 집중을 하게 되면 정확성이 떨어지는 현상이 발생했다. 그래서 효율성과 정확성을 모두 가져가는 새로운 Residual Block을 기반하여 Factorized Convolution을 사용하여 GPU 실시간 작동에 효율성을 주고 성능을 끌어올리며 다양한 객체 클래스를 분할하며 높은 정확도를 보여주고 Cost를 줄인 ENet 모델을 제시하였다. 
 
ENet 모델은 낮은 latency 연산으로 구성된 새로운 Deep Neural Network로 기존대비 빠르고 적은 파라미터로 좋은 정확도를 뽑아냈다. 기존대비 경량화를 진행해 빠른 연산을 진행할 수 있게 한 큰 특징이 Bottleneck 구조를 이용한 것이다. nxn layer를 계속하여 쌓는 것이 아니라 중간에 1x1 layer를 쌓으면서 layer의 깊이는 그대로 유지하고 차원을 늘리거나 줄임으로써 연산량을 nxn 그대로 쌓는 것보다 줄일 수 있다. 하지만 이 Bottleneck 구조는 성능 저하의 문제가 생겼고 이때 제시한 모델이 ERFNet이다. 
 
ERFNet은 더 정확하고 빠른 Semantic Segmentation 방법을 제안하였다. Convolution Block에 중점을 두어 해결하려고 하였고 Residual Connection 과 Factorized Convolution은 그대로 이용하였다, ERFNet과 ENet의 큰 차이는 Bottleneck 구조 가 아닌 Non-Bottleneck 구조를 사용하였다. 이로써 ENet이 해결했던 연산량 감소의 문제가 다시 생겼고 ERFNet에서는 2D- convolution이 아닌 1D-convolution을 제시하였다. 이는 nxn convol ution을 진행하는 것이 아닌 nx1 1xn으로 두번 나누어 convolution을 진행하면서 Convolution 연산을 진행할 시 발생하는 반복연산을 피하면서 연산량을 줄였다.  이는 Bottleneck 구조에서 사용되는 연산은 하나의 큰 Convolution Layer를 낮은 순위 근사치로 더 작고 간단한 작업으로 분해하는 것인데 이 분해 속도를 크게 높히고 매개변수의 수를 크게 줄여 중복성을 줄인다. 또한 층 사이에 삽입되는 비선현 연산으로 계산 기능을 더 풍부하게 만들어준다.

![image](/uploads/7ec40471b276ffba9442b0d22aa7cdab/image.png)

* 5월 4주차~6월 1주차 

ERFNet의 모델 성능 개선을 위해 ERFNet의 1D-Factorized Layer를 ENet의 2D-Factorized Layer로 변경했다. 그렇게 진행하면 Bottleneck에서 Non-Bottleneck으로 진행했을때의 연산량 감소 문제를 해결하지 못하는데 그래서 선택한 방법은 nxn Convoloution을 진행하는 것이 아닌 nxm mxn Convolution을 진행함으로써 nxn보다 중복연산을 피함으로써 연산량을 줄였다. 이렇게 진행을 하게되면 ERFNet보다는 무겁겠지만 최대한 성능을 챙기며 연산량도 줄이는 효과를 볼 수 있다. 실제 제시한 ERFNet 모델은 150 epcoh을 이용하여 학습을 진행하지만 성능 향상을 확인하기 위해 매번 150 epoch을 진행하는 것은 시간적 제약이 커 1/5을 줄인 30 epoch으로만 성능 비교를 진행하였다. 제시된 ERFNet을 30 epoch만 학습을 진행했을 때 41.9%의 정확도가 나왔으며 모델을 2D-facotirzed Convolution nxm mxn으로 구조를 바꾸고 학습을 진행한 결과 42.3%의 정확도가 나왔다. 30 epoch만 학습을 진행하고 비교했는데 0.4% 의 차이가 나는 것으로 보아 더 많은 학습을 진행하다보면 더 좋은 성능이 나올 것으로 기대가 된다.

< 결과 >
ERFNet
Best epoch is 30, with Val-IOU = 0.4192
MyNet
Best epoch is 27, with Val-IOU = 0.4232

* 6월 1주차~6월 2주차

아주대 소프트콘 대회에 자기주도프로젝트 분야에 나갈 포스터와 발표자료 및 발표 영상 제작을 하기로 회의를 진행하였다. 자율주행에 핵심이 되는 Segmentation에 대한 내용과 발전을 위한 모델인 ConvNet, ENet, ERFNet의 모델들의 장단점을 통해 구조를 변형하여 성능 향상을 목적으로 두고 진행한 것을 포스터에 작성하고 발표를 진행하였다. 해당하는 ppt 발표자료를 만들고 발표를 영상으로 찍은 뒤 소프트콘에 제출을 진행하였다. 해당 포스터와 발표자료를 같이 첨부하였다.

