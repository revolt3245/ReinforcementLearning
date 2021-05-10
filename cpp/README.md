# Manual

1. C/C++>일반>추가 포함 디렉터리 : libtorch의 include 파일
2. C/C++>언어>언어확장 사용안함, 준수모드 : 아니요
3. 링커>일반>추가 라이브러리 디렉터리 : libtorch 의 lib파일
4. 링커>입력>추가 종속성 : torch.lib, torch.cuda.lib, caffe2_nvrtc.lib, c10.lib, c10_cuda.lib, torch.cpu.lib
5. 구성속성>디버깅>환경 : Path=3번;%PATH% 입력
