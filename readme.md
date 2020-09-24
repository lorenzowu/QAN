# Video Quality Assessment based on Quality Aggregation Networks

## Description
QAN code for the following papers:

- Wei Wu, Zhenzhong Chen, Shan Liu. Video Quality Assessment based on Quality Aggregation Networks.

## Data Generation
```
python generate_video_patchs.py
```

## Test Demo
The model weights provided in `model` are the saved weights when running a random split of LIVE VQA Quality Database. The random split is shown in [data/train_test_LIVE_VQA.txt](https://github.com/lorenzowu/QAN/blob/master/data/train_test_LIVE_VQA.txt), which contains train/test split assignment (random).
```
python test_demo.py
```
The test results are shown in [result/test_result.xlsx](https://github.com/lorenzowu/QAN/blob/master/result/test_result.xlsx).
