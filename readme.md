# Video Quality Assessment based on Quality Aggregation Networks

## Description
QAN code for the following papers:

- Wei Wu, Zhenzhong Chen, Shan Liu. Video Quality Assessment based on Quality Aggregation Networks.

## Data Generation
```
python generate_video_patchs.py
```

## Test Demo
The model weights provided in `model` are the saved weights when running a random split of LIVE VQA Quality Database. The random split is shown in [data/train_val_test_split.xlsx](https://github.com/subpic/koniq/blob/master/metadata/koniq10k_distributions_sets.csv), which contains video file names, scores, and train/validation/test split assignment (random).
```
python test_demo.py
```
The test results are shown in [result/test_result.xlsx]().