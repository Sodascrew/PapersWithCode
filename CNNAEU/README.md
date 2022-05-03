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

通过一个CNN autoencoder，学习输入的低维特征表示（size不变，通道数降低至端元数R），经过softmax后（符合sum-to-one条件）视作R个端元的丰度图，同时将decoder采用kernel size为fXf的卷积（线性激活，符合LMM），将通道数变回输入的谱带数B，学习到的权重即视作矩阵M。

具体模型如下图所示：

![model](./photo/model.png)

其中Abundance maps通道i上的像素点的值对应该像元中端元i所占的比例。而最后一层网络的权重与上文spectral-spatial model中矩阵的对应关系大致如下：

![model interpretation](./photo/model_interpretation.jpg)

为了衡量autoencoder输入输出的相似性，使用了 the spectral angle distance(SAD) ，对于每一个像元有：

<img src="https://latex.codecogs.com/svg.image?J_{\mathrm{SAD}}(\boldsymbol{x},&space;\hat{\boldsymbol{x}})=\arccos&space;\left(\frac{\langle\boldsymbol{x},&space;\hat{\boldsymbol{x}}\rangle}{\|\boldsymbol{x}\|_{2}\|\hat{\boldsymbol{x}}\|_{2}}\right)"/>

对单个patch：

<img src="https://latex.codecogs.com/svg.image?\mathcal{L}^{(i)}=\frac{1}{\left|\mathcal{B}^{i}\right|}&space;\sum_{\boldsymbol{x}_{p}&space;\in&space;\mathcal{B}^{i}}&space;J_{\mathrm{SAD}}\left(\boldsymbol{x}_{p},&space;\hat{\boldsymbol{x}}_{p}\right)"/>

因此总的损失函数为N个patch求和：

<img src="https://latex.codecogs.com/svg.image?\mathcal{L}=\sum_{i=1}^{N}&space;\mathcal{L}^{(i)}"/>

此loss只衡量了两向量方向的相似性，与尺度信息无关(sclae invariant)，对于端元提取效果好，但混合像元分解即求解丰度图效果并非最佳。同时由于使用了batch normalization、learkyReLU以及softmax，丰度图中的值倾向于0或1。

> ​	This loss, although very good for endmember extraction, is not an optimal metric for data reconstruction since it is scale invariant. It is possible that SAD loss leads to **higher variance in the abundance estimation** than is explainable by considering only the variance in the quality of the extracted endmembers. Still, methods using SAD loss can achieve fairly good results for abundance estimation as the results of the MTAEU method demonstrate.(paper Ⅲ.Experiments A.Methodolody and Performance Metrics)
>
> ​	The quality of extracted endmembers weighs more heavily in our evaluation of the method than the quality of the abundance maps. This is because **the abundance maps** produced by our method tend to be very **binary**, that is abundances are either very low or very high, almost like classification maps. This happens because of batch normalization and the fact that we are using a ReLU like activation and the softmax function to enforce the ASC constraint.(paper Ⅲ.Experiments A.Methodolody and Performance Metrics)

对于此问题，可以使用上述模型用于提取端元，再用另一个的autoencoder（decoder的权重固定为上一模型得到的端元矩阵，不适用batch normalization）进行训练得到新的丰度图。(model CNNAEU2)

> As was mentioned at the start of this section, the abundance maps produced by CNNAEU are very intense looking, that is both very **sparse and binary**. It is possible to extend the method so it **refines the abundance maps after extracting the endmembers**. This could be done using an autoencoder in serial with the unmixing one, which has **its decoder’s weights set to be the extracted endmember matrix and made nontrainable so they are fixed, and does not use batch normalization layers**. By training it for a few epochs, it will produce abundance maps which are not nearly as binary looking.(paper Ⅲ.Experiments F.Abundance Maps)

## Dataset

### Samson

Samson dataset来源：[Data - rslab (ut.ac.ir)](https://rslab.ut.ac.ir/data)

读取samson_1.mat，可得到训练数据如下：

![train data](./photo/train_data.png)

'V'对应shape为(156, 9025)的训练数据

读取end3.mat，可得到标签如下：

![label](./photo/label.png)

其中，‘A’对应shape为(3, 9025)的abundance map，‘M'对应shape为(156, 3)的reference endmembers

## Code





参考：[dv-fenix/HyperspecAE: Code for the experiments on the Samson Dataset as presented in the paper: Hyperspectral Unmixing Using a Neural Network Autoencoder (Palsson et al. 2018) (github.com)](https://github.com/dv-fenix/HyperspecAE)



## Reference

* [高光谱遥感 原理、技术与应用.童庆禧 张兵 郑兰芬编著.高等教育出版社](https://ss.zhizhen.com/detail_38502727e7500f266bcf87befb91286a0ec30826f0d5e6681921b0a3ea25510134114c969f2eae5c46d827fd16ff83d4cdacdf48f85a5afad7ae4646ad63b0cc07a17234dce20b1b5d891e50a87c47bf?)
* [Convolutional Autoencoder for Spectral–Spatial Hyperspectral Unmixing | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/9096565)

