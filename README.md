# OilWellStratification

### 油井分层预测



#### <u>*最新更新</u>

完成基于xgboost决策树分类的油井2分类问题。



#### 概述

对于油井探测过程中的各维度数据，经过分类器预测，可以得到8个具体的分层类别（61-68），以及不分层的情况。（暂时只支持显示分层与不分层2类）



#### 环境搭建

运行need_download.sh 脚本，下载需要的库

对于探井las类型的文件，使用了las库，如果不能正确读取，请检查是否是本地的las文件使用了utf-8编码方式，可以根据报错信息修改las库文件，这是最简单的修改方式。



#### 输入说明

输入的是las文件类型，其中必须要包括'DEPTH', 'AC', 'SP', 'COND', 'ML1', 'ML2'这8个条目，否则并不能进行分层预测。

例：

```
...
~A   DEPTH           AC              GR              SP              COND            ML1             ML2             
0.000           -9999.000       -9999.000       -9999.000       -9999.000       -9999.000       -9999.000       
0.125           -9999.000       -9999.000       -9999.000       -9999.000       -9999.000       -9999.000       
0.250           -9999.000       -9999.000       -9999.000       -9999.000       -9999.000       -9999.000       
0.375           -9999.000       -9999.000       -9999.000       -9999.000       -9999.000       -9999.000       
...
```



#### 程序功能

Predictor/predict.py为程序主体，运行后会使用指定的模型来对数据进行分层预测，最后将分层结果输出。

Tips:不是所有层都会被分层，只有上述6个维度均不是缺失值“-9999”的层才会被分类。



#### 程序使用

首先编辑predictor目录下的setting.yml

| 名称            | 用途                    |
| --------------- | ----------------------- |
| ModelPath       | 模型的路径              |
| WellLasFilePath | 需要加载的Las文件的路径 |
| ResultSavePath  | 结果保存的目录          |

接下来使用命令

```
python predict.py
```

进行运行。



#### 程序输出

生成xlsx格式的分类文件，对所有探测到数据的层进行分类，1表示该深度会被分层，0表示不分层的层。

例：

| DEPTH    | level |
| -------- | ----- |
| 1200     | 0     |
| 1200.125 | 0     |
| 1200.25  | 0     |
| 1200.375 | 0     |
| 1200.5   | 0     |
| 1200.625 | 1     |
| 1200.75  | 1     |
| 1200.875 | 1     |
| ...      |       |



#### 预设模型

预设模型有3个，是在不同类别分布下训练出的，这3个模型是所有训练模型中能够得到最好F1 Measure的。



#### 训练器（待添加）