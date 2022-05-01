# Convolutional Autoencoder for Spectral–Spatial Hyperspectral Unmixing

## 线性混合光谱模型 LMM(Linear mixing model)

参考：[高光谱遥感 原理、技术与应用.童庆禧 张兵 郑兰芬编著.高等教育出版社](https://ss.zhizhen.com/detail_38502727e7500f266bcf87befb91286a0ec30826f0d5e6681921b0a3ea25510134114c969f2eae5c46d827fd16ff83d4cdacdf48f85a5afad7ae4646ad63b0cc07a17234dce20b1b5d891e50a87c47bf?) 第6章 混合光谱理论与光谱分解

遥感器获取的光谱信号以像元为单位，是其对应的地表物质光谱信号的综合。受到遥感空间分辨力的限制，一个像元可能包含不止一种土地覆盖类型，形成混合像元(mixed pixel)。

混合像元形成主要有三个原因：

* 单一成分物质的光谱、几何结构、及在像元中的分布
* 大气传输过程中的混合效应
* 遥感仪器本身的混合效应

后两个为非线性效应，可通过大气纠正、仪器校准等方法克服，此处的混合光谱模型主要解决第一个原因。



给定假设：高光谱图像中的每个像元都可以近似认为是图像中各个端元（endmember）的线性混合。得到线性光谱混合模型如下：

![linear mixture model](./photo/linear_mixture_model.png)

线性解混就是在已知所有端元的情况下求出每个图像像元中各个端元所占的比例，从而得到反映每个端元在图像中分布情况的比例系数图。

线性光谱混合模型的矩阵表示如下：（A为端元光谱矩阵 endmember matrix，B为各端元光谱丰度 endmember abundance）

![matrix representation](./photo/matrix_representation.png)

线性光谱解混主要由两个步骤构成：

* 端元提取：提取“纯”地物的光谱
* 混合像元分解：用端元的线性组合来表示混合像元



## CNNAEU（CNN autoencoder unmixing）

paper：[Convolutional Autoencoder for Spectral–Spatial Hyperspectral Unmixing | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/9096565)

对于HU（hypterspectral unmixing）任务，大多数方法只利用了光谱信息（spectral information），未利用空间信息（spatial information）。但光谱图像与其他自然图像相同，每个像素与领域像素高度相关。提出了使用CNN的autoencoder模型，同时利用空间和光谱信息进行解混（spectral-spatial method）。

### Notation

![notation](./photo/notation.png)

![table 1](./photo/table1.png)

### Problem formulation and model

提出的spectral-spatial model形式如下：

<img src="https://latex.codecogs.com/svg.image?\boldsymbol{x}_{p}=\boldsymbol{M}&space;\boldsymbol{s}_{p}&plus;\sum_{i&space;\in&space;\mathcal{N}_{p}&space;\backslash&space;p}&space;\boldsymbol{M}_{i}&space;\tilde{\boldsymbol{s}}_{i}&plus;\boldsymbol{\epsilon}_{p},&space;\quad&space;p=1,&space;\ldots,&space;P"/>

上式中：

* 第一项反映spectra information。Sp是长度为R的向量，每个值属于[0,1]且求和为1。M为BXR的矩阵，列向量对应一个端元的光谱信息。
* 第二项反映spatial informaton。求和符号内M与S的乘积表明位置i对于点p光谱的贡献，i的使用如上面的TABLE 1所示。i取值范围覆盖点p的fXf非空领域，且f为奇数。
* 第三项为noise。

对于此问题，求解目标为上式中的 endmember matrix M 以及每个像元对应的abundances Sp。

通过一个CNN autoencoder，学习输入的低维特征表示（size不变，通道数降低至端元数R），经过softmax后（符合sum-to-one条件）视作R个端元的丰度图，同时将decoder采用1x1卷积（线性，符合LMM），将通道数变回输入的谱带数B，学习到的权重即视作矩阵M。

具体模型如下图所示：

## Reference

* [高光谱遥感 原理、技术与应用.童庆禧 张兵 郑兰芬编著.高等教育出版社](https://ss.zhizhen.com/detail_38502727e7500f266bcf87befb91286a0ec30826f0d5e6681921b0a3ea25510134114c969f2eae5c46d827fd16ff83d4cdacdf48f85a5afad7ae4646ad63b0cc07a17234dce20b1b5d891e50a87c47bf?)
* [Convolutional Autoencoder for Spectral–Spatial Hyperspectral Unmixing | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/9096565)

