# 2026 March

## March 4 

先复习交流一下具体情况

## March 6 

- 建立一个打卡点

目前需求：

- 复现一个*完整*的工程文件

    需要寻找适合进行复现的工程文件

    以起到巩固基础知识 做到性能验证的需求

    在完成过程中解决各类工程类琐碎问题

- 验证可行性，验证文章可靠性以及理论与数据来源

- 按照论文结果得出相似的结果


## March 16 

- 开始尝试了解代码原含义

- 尝试跑通源代码

具体的代码内容可以在py文件中查看注释内容

### 现已注释的文件

#### main.py

#### models: 

##### layer: 

- local_learning_signal_generation.py; 
- stdp_layers.py

## March 19

- 当前任务是先完成对 `local_learning_signal_generation.py;stdp_layers.py` 两个文件的注释

- 完成后，对文章的结果进行复现，首先第一步先跑通当前的原有代码，保证能有结果输出

- 在此基础上，结合其他的工程文件，形成自己的新的复现工程

### 目前已经成功在本地计算机配置git环境，已经成功测试


现在已经成功训练了脚本 script_test.sh

目前目标：

1. 对文章中alpha_post部分的消融实验进行测试

2. 完成script.sh部分的测试

为了避免混淆，我们使用script_new.sh 进行测试，并且对于alpha_post测试部分我们使用script_alpha_post.sh

这两个文件我们会更改输出log的路径，这样防止混淆

今天已经开始训练。等待后续训练完成后进行分析

## March 22

对于上述两点：

1. 对文章中alpha_post部分的消融实验进行测试

2. 完成script.sh部分的测试

正在训练中，但是鉴于内存需求较大（已经占用了cuda内存导致溢出的情况），所以需要较长时间


## March 24

进行组会汇报

## March 27

检测出来问题：
```
AttributeError: module 'numpy' has no attribute 'bool'
```

`cifar10_dvs.py` 和当前 NumPy 版本不兼容

修改方案：

```
polarity = read_bits(addr, polarity_mask, polarity_shift).astype(np.bool)
```

修改`np.bool`为`np.bool_`

## March 30

更新了文件夹 **Code\Origin\1_Training_Results**，使用markdown文件进行结果总结

$\alpha_{post}$消融实验：`1_AblationStudyOnIncludingNonCausalTerms.md`

$BPTT$与$TESS$在不同图像识别任务上的比较：`2_ComparisonOfBPTTAndTESSAcrossDifferentImageRecognitionTasks.md`

关于文件和运行的细节在上述的2个markdown文件下。