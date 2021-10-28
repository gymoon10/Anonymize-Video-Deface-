## Anonymize_Video-read.md

폴더에 존재하는 모든 동영상에 대해 비식별화 작업을 수행하는 프로그램입니다.

참고 : https://github.com/ORB-HD/deface

모든 프로그램은 주어진 영상에 대해 CenterFace 얼굴 탐지 알고리즘을 적용시켜 검출된 얼굴에 대해 비식볋화하는 작업을 거칩니다. (deface와 동일) -> anonymized로 저장
이후에 얼굴을 제대로 비식별화 하지 못한 불량 케이스에 대해 추가적인 작업 (드랍, 보간법)을 수행하여 추가적으로 비식별화를 진행합니다. -> anonymized2로 저장

기존 deface 알고리즘의 usage option에 대해 --mask-scale 1.03, --replacewith solid, --boxes 옵션을 적용시키고 나머지는 default입니다.

anonymize_all_inter.py : 보간법을 사용하여 비식별화가 제대로 적용되지 않은 프레임에 대해 추가적으로 비식별화 작업을 수행 

anonymize_all_drop.py : 비식별화가 제대로 되지 않은 프레임을 드랍시킴

두 프로그램 모두 얼굴이 제대로 검출된 정상 케이스와 비정상 케이스를 분류하는데 있어, 모든 개별 프레임에서 검출된 얼굴 개수의 최빈값을 기준으로 하기 때문에 한계가 있습니다.

### Usage

파이썬 파일에서 centerface.onnx의 경로와 ipath (원본 동영상 폴더 경로), opath (비식별화된 영상이 저장될 경로)를 적절하게 설정해주세요.

$ python3 ~/anonymize_all_drop.py
