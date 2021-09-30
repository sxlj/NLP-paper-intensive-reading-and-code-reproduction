@[toc]

### 一、储备知识

#### 1. 语言模型的概念
&emsp;&emsp;语言模型是用于计算一个句子出现的概率，即语言模型可以判断某一句话从语法上是否通顺（是不是人话），从语义上是否有歧义。在很多时候，我们都要度量一句话的出现概率，一句话的出现概率等同于一句话语法的流畅程度。
#### 2. 语言模型的发展
 - 基于专家语法规则的语言模型
  语言学家企图总结出一套通用的语法规则，比如形容词后面接名词等

 - 基于统计学的语言模型
  通过概率计算来刻画语言模型
  $$
  P_{LM}(s)=P_{LM}(w_1,w_2,...,w_n)=P_{LM}(w_1)P_{LM}(w_2|w_1)...P_{LM}(w_n|w_1w_2...w_{n-1})
  $$
  基于马尔科夫假设，假设：任意一个词，它的出现概率只与前面出现的一个词(或者几个词)有关，则可以将语言模型简化如下：
  Unigram Model：
  $$
  P_{LM}(w)=\prod_{i=1}^nP_{LM}(w_i)
  $$
  Bigram Model：
  $$
  P_{LM}(s)=P_{LM}(w_1)P_{LM}(w_2|w_1)P_{LM}(w_3|w_2)...P_{LM}(w_n|w_{n-1})
  $$
  Trigram Model：
  $$
  P_{LM}(s)=P_{LM}(w_1)P_{LM}(w_2|w_1)P_{LM}(w_3|w_2，w_1)...P_{LM}(w_n|w_{n-1}，w_{n-2})
  $$
  
#### 3.语言模型的平滑操作
&emsp;&emsp;有一些词或者词组在语料中没有出现过，但是这不能代表它不可能存在。平滑操作就是给那些没有出现过的词或者词组也给一个比较小的概率。
&emsp;&emsp;平滑概念指的是试图给没有出现的N-gram分配一个比较合理的数值出来，不至于直接为0。下面介绍多种平滑策略：

 - Add-one Smoothing—拉普拉斯平滑
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625105145725.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70#pic_center)
$V$指的是词库的大小。
 - Add-K Smoothing—拉普拉斯平滑
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210625105625558.png#pic_center)
k是一个超参数，需要训练
#### 4. 语言模型的评价指标
&emsp;&emsp;语言模型实质上是一个多分类问题（这只是一种理解方式，类别是每个词）。 下面介绍一种新的评价指标—perplexity（困惑度）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210624232826449.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)
perplexity越低，表明语言模型认为这句话出现的概率越高，这句话越有可能出现。困惑度最小是1。句子概率越大，语言模型越好，困惑度越小。

### 二、研究背景
#### 1. 词的表示方式

 - one-hot编码
“话筒”表示为 [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 ...]
“麦克”表示为 [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 ...]
表示简单，
问题：词越多，维数越高（词表大小V）无法表示词和词之间关系
 - 分布式表示（稠密表示）
维度D(D<<V)
![在这里插入图片描述](https://img-blog.csdnimg.cn/0d3ed2039eab4cf9a37f467d16521b80.png)
通过词与词之间的余弦相似度来表示词和词之间的关系
### 三、研究意义

 - 衡量词向量之间的相似程度(相似单词的词向量彼此接近)
![在这里插入图片描述](https://img-blog.csdnimg.cn/f51dea09eff9422196fb250c2c11ac27.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/fe19426bca584dbc9c2c5d62e22923ec.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)

 - 作为预训练模型提升nlp任务
应用到诸如：命名实体识别、文本分类下游任务中
![在这里插入图片描述](https://img-blog.csdnimg.cn/234a02b00fdd42ee9adb33f2f42b541f.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)
也可以应用到其他NLP任务上，相当于半监督训练
![在这里插入图片描述](https://img-blog.csdnimg.cn/c38d450db93e4be88ce644a4705d4bf5.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)

 - 保留单词间的线性规则性

### 四、论文精读
&emsp;&emsp;以如下两篇word2vec文章进行精读，第二篇文章是对第一篇文章细节部分的补充。
![在这里插入图片描述](https://img-blog.csdnimg.cn/7b9f8cc72bdb4131a881eacd5bb2c6fb.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)
$$向量空间中词表示的有效估计$$
![在这里插入图片描述](https://img-blog.csdnimg.cn/e3d61d9af83543718708fd4618b64942.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)
$$单词和短语的分布式表示及其组成$$

#### 1. 摘要核心

 1. 提出了两种新颖的模型结构Skipgram和CBOW用来在大规模的预料上计算词向量
 2. 采用一种词相似度的任务来评估对比词向量质量，比以前的神经网络工作相比有非常大的提升
词对的类比实验来评估对比词向量质量
 3. 大量降低模型计算量可以提升词向量质量
 4. 进一步，在我们的语义和句法的词相似度任务上，我们的词向量是当前最好的效果

#### 2.Introduction-介绍

 - 传统NLP把词当成最小单元处理，没有词相似度的概念，并且能够在大语料上得到很好的结果，其中好的模型是N-gram模型
 - 然而，很多自然语言任务只能提供很小的语料，如语音识别、机器翻译，所以简单的扩大数据规模来提升简单模型的表现在这些任务上不再适用，所以我们必须寻找更加先进的模型
 - 数据量较大时，可以采用分布式表示方法，在大语料上，分布式表示的语言模型的效果会超过N-gram模型

#### 3. 对比模型
&emsp;&emsp;本文与前馈神经网络语言模型（NNLM）和循环神经网络语言模型（RNNLM）进行对比。
##### 3.1 前馈神经网络语言模型（NNLM）
![在这里插入图片描述](https://img-blog.csdnimg.cn/c2bac08119e44c5daa0af104e45b8e5e.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)
这个模型也就是所谓的N-gram模型。根据前n-1个单词，预测第n个位置单词的概率，使用梯度下降法优化模型，使得输出的正确的单词概率最大化。
![在这里插入图片描述](https://img-blog.csdnimg.cn/f12852c11be44c4e81bff1f3dfae3a80.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)

&emsp;&emsp;语言模型是无监督任务（不需要标注语料）。那么没有标注的语料是如何做监督学习的呢？根据前n-1个单词，预测第n个位置单词，这样就可以利用无标注语料进行监督学习。

 - 输入层
  将词映射成向量，相当于一个$1×V$的one-hot向量乘以一个$V×D$的向量得到一个$1×D$的向量
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/83bbc44b135545f1886f5f68a33e98ff.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)
  在上图 前馈神经网络语言模型中，可以使用$1×V×V×D=1×D$并行计算加速。

 - 隐藏层
  一个以tanh为激活函数的全连接层
  $$
  a=tanh(d+Wx)
  $$
  

 - 输出层
  一个全连接层，后面接一个softmax函数来生成概率分布。
  $$
  y=b+Ua
  $$
  其中y是一个$1×V$​的向量,使用softmax进行归一化
  $$
  P(w_t|w_{t-n+1},...,w_{t-1})=\frac{exp(y_{w_t})}{\sum_iexp(y_{w_t})}
  $$
  

**语言模型困惑度和Loss关系：**
多分类的交叉熵损失函数如下：T表示句子中词的个数
$$
Loss: L=-\frac{1}{T}\sum_{i=1}^Tlog(P(w_t|w_{t-n+1},...,w_{t-1}))\\
PP(s)=e^L
$$

##### 3.2 循环神经网络语言模型（RNNLM）
![在这里插入图片描述](https://img-blog.csdnimg.cn/aca7719a560e41b4899be6d9de11b38a.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)

 - 输入层
  和 NNLM一 样 ，需要将当前时间步的输入转化为词向量

 - 隐藏层
  对输入和上一时间步的隐藏输出进行全连接层操作
  $$
  s(t)=Uw(t)+Ws(t-1)+d
  $$
  

 - 输出层
  一个全连接层，后面接一个softmax函数来生成概率分布
  $$
  y(t)=b+Vs(t)
  $$
  其中$y$是一个$1×V$​的向量：
  $$
  P(w_t|w_{t-n+1},...,w_{t-1})=\frac{exp(y_{w_t})}{\sum_iexp(y_{w_t})}
  $$
  

直观展示如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2413632a7ec74ab1a139ee74b5bb7614.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)
每个时间步预测一个词，在预测第n个词时使用了前n-1个词的信息。
$$
Loss: L=-\frac{1}{T}\sum_{i=1}^Tlog(P(w_t|w_{t-n+1},...,w_{t-1}))
$$

#### 4.word2vec模型的两种模型架构
&emsp;&emsp;定义Log Linear Models：将语言模型的建立看成一个多分类问题，相当于线性分类器加上softmax
$$
Y=softmax(wx+b)
$$
&emsp;&emsp;语言模型基本思想：句子中下一个词的出现和前面的词是有关系的，所以可以使用前面的词预测下一个词。Word2vec可以看作是语言模型的简化。
&emsp;&emsp;Word2vec的基本思想：句子中相近的词之间是有联系的，比如今天后面经常出现上午、下午和晚上。所以Word2vec的基本思想是用词来预测词，skip-gram使用中心词预测周围词，cbow使用周围词预测中心词。
##### 4.1 Skip-gram
&emsp;&emsp;下面来介绍Skip-gram模型的原理，图中以window=2为例，一个中心词产生4个训练样本
![在这里插入图片描述](https://img-blog.csdnimg.cn/b65119b444d14002a8506612fe542ad1.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)
可以得到四个概率
$$p(w_{t-1}|w_t)\\p(w_{t-2}|w_t)\\p(w_{t+1}|w_t)\\p(w_{t+2}|w_t)$$
那么，Skip-gram模型是如何求概率以及如何学习词向量？
&emsp;&emsp;如果求$p(w_{t-1}|w_t)$的概率大小，输入是$w_t$，输出是$w_{t-1}$,相当于词表大小的多分类模型。
![在这里插入图片描述](https://img-blog.csdnimg.cn/c35c3dc2dd18479595ceacc05d88fb87.png)

 - $w_i$相当于将index映射成$1×V$的one-hot向量

 - $W$为**中心词的词向量矩阵**，大小为$V×D$，$w_i$与$W$矩阵相乘得到一个$1×D$的词向量。

 - $W^*$为**周围词的词向量矩阵**，将词向量矩阵与周围词矩阵$W^*$相乘得到$1×V$的向量，将向量softmax就可以得到概率。
  $$
  p(w_{i-1}|w_i)=\frac{exp(u_{w_{i-1}}^Tv_{w_{i}})}{\sum_{w=1}^Vexp(u_{w}^Tv_{w_{i}})}
  $$
  （如果按这个公式理解的话，$W^*$是$V×D$）

 - 目标是使得对应位置的概率越大越好，通过梯度反向传播训练$W^*$与$W$，$W^*$与$W$就是所谓的词向量，那么，我们要选择哪一个作为最终词向量，可以采用如下手段：
	>- $W$-中心词向量相对来说效果较好
	>- $(W+W^*)/2$（如果按这个公式理解的话，$W^*$是$V×D$）
	
 - Skip-gram的损失函数

	![在这里插入图片描述](https://img-blog.csdnimg.cn/f3db0b05860c48b094a1279ddf47dfed.png#pic_center)
$m$是窗口的大小，损失函数越大越好

##### 4.2 CBOW
&emsp;&emsp;下面来介绍CBOW模型的原理，图中以window=2为例，一个中心词只产生1个训练样本
![在这里插入图片描述](https://img-blog.csdnimg.cn/31f892399bdb4372862282a918968b14.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)
可以得到概率：
$$
p(w_t|w_{t-2},w_{t-1},w_{t+1},w_{t+2})
$$
那么，CBOW模型是如何求概率以及如何学习词向量？
&emsp;&emsp;如果求$p(w_t|w_{t-2},w_{t-1},w_{t+1},w_{t+2})$的概率大小，输入是$w_{t-2}、w_{t-1}、w_{t+1}、w_{t+2}$，输出是$w_{t}$,相当于词表大小的多分类模型。
![在这里插入图片描述](https://img-blog.csdnimg.cn/4518b38921384478a07721542493c17f.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)

 - $w_{t-2}、w_{t-1}、w_{t+1}、w_{t+2}$将index映射成$1×V$的one-hot向量

 - $W$为周围词矩阵，大小为$V×D$，将one-hot向量与中心词矩阵相乘得到4个$1×D$的词向量，将这4个词向量作求和或者平均，得到1个$1×D$的词向量

 - $W^*$为中心词矩阵，大小为$D×V$，将$1×D$的词向量与中心词矩阵相乘得到$1×V$的向量，将向量softmax就可以得到概率。

 - 目标是使得对应位置的概率越大越好，通过梯度反向传播训练$W^*$与$W$，$W^*$与$W$就是所谓的词向量，那么，我们要选择哪一个作为最终词向量，可以采用如下手段：
	>- $W$-中心词向量相对来说效果较好
	>- $(W+W^*)/2$（如果按这个公式理解的话，$W^*$是$V×D$）
	
 - 损失函数
 $$
  J(\theta)=\frac{1}{T}\sum_{t=1}^Tp(w_t|w_{t-2},w_{t-1},w_{t+1},w_{t+2})\\
    J(\theta)=\frac{1}{T}\sum_{t=1}^T\frac{exp(u_{w_{sum}}^Tv_{w_{i}})}{\sum_{w=1}^Vexp(u_{sum}^Tv_{w_{j}})}
  $$
    损失函数越大越好
##### 4.3 关键技术
&emsp;&emsp;softmax涉及到$1×D$的$U$矩阵与$D×V$的V的矩阵相乘，得做V次相乘，$V$是特别大的，所以，全连接层也是特别大的。那么，应该如何降低softmax的复杂度呢？下面介绍两种方法：层次softmax与负采样
 - Hierarchical softmax（层次softmax）
  层次softmax的**基本思想**就是将softmax的计算转化成求多个sigmoid的计算，并且少于$log_2V$
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/4e6852349d3f4de497fb96c5affca28b.png)
  转化为二叉树的结构
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/44c194fb50cf40a4ba42dbbd76512ac3.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)
  所以，如果是满二叉树，只需要计算$log_2V$个sigmoid。softmax需要做V个指数相乘，而sigmoid只需要做一次指数相乘，$$log_2V<V$$,加速了softmax的运算。
  &emsp;&emsp;满二叉树需要计算$log_2V$个sigmoid，那么构建带权重路径最短二叉树-Huffman树可以计算少于$log_2V$个sigmoid。
  &emsp;&emsp;Skip-gram的层次softmax如下：
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/0fa9d2b054e24015aa8d30a17bcbb3d4.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70#pic_center)
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/876eb1486617402fb2b65cafb370448c.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70#pic_center)
  $$p(I|c)=\sigma(\theta_0^Tv_c)\sigma(\theta_1^Tv_c)\sigma(1-\theta_2^Tv_c)=\sigma(\theta_0^Tv_c)\sigma(\theta_1^Tv_c)\sigma(-\theta_2^Tv_c)$$
  $v_c$是中心词向量
  &emsp;&emsp;推广开来，
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/d178fe39ace04bee88b30241b04c84da.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70#pic_center)
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/d518b6f4f67341378e1c2255e0f9c686.png#pic_center)
  其中，$[|x|]=1 \ or -1$
  $ch(n(w,j))=n(w,j+1)$是用来判断是否是右孩子节点
  $v_{w1}$是中心词的词向量
  $v^{'}_n(w,j)$是词$w$在树上的第$j$个节点的参数
  &emsp;&emsp;CBOW的层次softmax如下：
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/6ee9477355d946a79e37202bd0905210.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70#pic_center)
  $$
  p(I|c)=\sigma(u_o^T\theta_o)\sigma(u_o^T\theta_1)\sigma(1-u_o^T\theta_2)=\sigma(u_o^T\theta_o)\sigma(u_o^T\theta_1)\sigma(-u_o^T\theta_2)
  $$
  $u_o$​是窗口内上下文词向量的平均

 - Negative Sampling（负采样）
  &emsp;&emsp;softmax之所以慢，是因为进行了词表大小V的多分类，所以，我们尝试舍弃多分类，提升速度。一个正样本，选取$k$个负样本，对于每个词，一次要输出一个概率，总共$k+1$个，$k<<V$。负采样的优化就是增大正样本的概率，减小负样本的概率
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/fac1e3f0b8ee4507bc1d0e7b0f94803a.png)
  $$公式：Skip-gram的负采样$$
  $v_c$是中心词向量
  $u_o$是窗口内上下文词向量
  $u_k$是负采样上下文词向量
  这里还是需要每个词的上下文词向量，总的参数比HS多（每次计算量不多），经过实验结果，可以发现，负采样比层次softmax快，这是因为负采样比层次softmax需要计算更少的概率
  &emsp;&emsp;那么，应该如何负采样呢？论文中提到一种方法：减少频率大的词抽样概率，增加频率小的词抽样概率。这样做不仅能加速训练，而且能得到更好的结果。
  抽样概率计算方法如下：
  $$
  P(w)=\frac{U(w)^{\frac{3}{4}}}{Z}
  $$
  $U(w)$是词$w$在数据集中出现的概率，$Z$为归一化的参数，使得求解之后的概率和为1
  $$
  J(\theta)=log\sigma(u_o^Tv_c)+\sum_
  {i=1}^TE_{j-P(w)}[log\sigma(-u_o^Tv_j)]
  $$
  $$公式：CBOW的负采样$$
  $u_o$是窗口内上下文词向量avg
  $v_c$是正确的中心词向量
  $v_j$​是错误的中心词向量

 - 重采样（subampling of Frequent Words）
	自然语言中有这样的共识：文档或者数据集中出现频率高的词往往携带信息较少，而出现频率低的词往往携带信息多。重采样的**原因**：
	>- 想更多地训练重要的词对，比如训练“France”和“Paris”之间的关系比训练“France”和“the”之间的关系要有用。
	>- 高频词很快就训练好了，而低频次需要更多的轮次
	
	重采样的**方法**如下：
	$$P(w_i)=1-\sqrt{\frac{t}{f(w_i)}}$$
	其中，$f(w_i)$为词$w_i$在数据集中出现的概率。论文中$t$选取为$10^{-5}$，训练集中的词$w_i$会以$P(w_i)$的概率被删除，词频越大，$f(w_i)$越大，$P(w_i)$越大，那么词$w_i$就有更大的概率被删除，如果词$w_i$的词频小于等于$t$，那么$w_i$则不会被剔除，这样就会对高频词少采一些，低频词多采一些
	重采样的**优点**是：加速训练，能够得到更好的词向量

##### 4.4 关键技术对比

1. 层次softmax用词频建立一颗哈夫曼树，树的叶子节点表示词，树的非叶子节点是模型参数，每一个单词的概率用其路径上的权重乘积来表示，这样可以减少高频的搜索时间；同时，对于生僻字来说，则在Huffman树中仍需要进行复杂运算($O(logN)$)
2. 负采样是把原来的softmax的多分类问题转化为一个正例和多个负例的二分类问题。即将预测每一个单词的概率，概率最高的单词为中心词改为预测该单词是不是正样本，当负例的个数取较小常数时，负采样在每一步的梯度计算中开销都比较小。那么为什么可以使用负采样呢？这是因为目标词只与相近的词有关，没有必要使用全部的单词作为负例来更新它们的权重。

#### 5.模型复杂度
&emsp;&emsp;模型复杂度的概念：论文中以计算所需要的参数的数目来代替模型复杂度。
&emsp;&emsp;NNML的模型复杂度：
![在这里插入图片描述](https://img-blog.csdnimg.cn/057d9bff0428472fb6f38f56ac8b95b6.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)
$x$维度$N*D$
$U维度V*H$
$W维度N*D*H$
$$
Q=V*H+N*D+N*D*H
$$
&emsp;&emsp;RNNML的模型复杂度：
![在这里插入图片描述](https://img-blog.csdnimg.cn/6b09a30e30a1446aa9ba130230d614f0.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)
$$
Q=H*H+H*V
$$
&emsp;&emsp;Skip-gram的模型复杂度：
![在这里插入图片描述](https://img-blog.csdnimg.cn/64cb7853c0924e1a81666862c24ad6d1.png)
$$
HS:Q=C(D+D*log_2V)\\Neg:Q=C(D+D*(K+1))
$$
&emsp;&emsp;CBOW的模型复杂度：
![在这里插入图片描述](https://img-blog.csdnimg.cn/e8c5ee49c7094ad3afc1ce0aca388c44.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)
$$
HS:Q=N*D+D*log_2V\\Neg:Q=N*D+D*(K+1)
$$
&emsp;&emsp;模型复杂度对比来看，CBOW的模型复杂度小于Skip-gram的模型复杂度小于循环神经网路语言模型的模型复杂度小于前馈神经网络语言模型的时间复杂度。并且使用负采样要比使用层次softmax更快。
#### 6.实验结果
&emsp;&emsp;实验采用词对推理数据集，其中有5个语义类、9个语法类
![在这里插入图片描述](https://img-blog.csdnimg.cn/8d24f28a240b407faa092ff359cea641.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70)

##### 6.1 与对比模型的直接对比实验
![在这里插入图片描述](https://img-blog.csdnimg.cn/7e3a13caa43b478abee5887f97f7373c.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70#pic_center)
##### 6.2 不同模型的效率分析
![在这里插入图片描述](https://img-blog.csdnimg.cn/bc06095080174799a1c0ed31fe90913a.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NjY0OTA1Mg==,size_16,color_FFFFFF,t_70#pic_center)
#### 7.总结
&emsp;&emsp;word2vec包括skip-gram（利用中心词预测周围词）与CBOW（利用周围词预测中心词）两种模型架构，还有两种加速softmax训练的关键技术，分别是层次softmax与负采样。层次softmax基本思想就是将softmax转化为多次sigmoid，使用Huffman树的结构；负采样的基本思想是将多分类问题转化为二分类问题：一个中心词与一个周围词就是正样本，一个中心词与一个随机采样得到的词就是负样本。还有第三种技术-重采样（针对自然语言处理中的共识），基本思想就是将高频词删去一些，将低频词尽可能保留，这样就可以加速训练，并且得到更好的词向量。
&emsp;&emsp;word2vec相比于语言模型用前面的词来预测词，简化为用词预测词，简化了结构，大大减小了计算量，从而可以使用更高的维度，更大的数据量。
**关键点：**

 - 更简单的预测模型—word2vec
 - 更快的分类方案—HS和NEG

**创新点：**

 - 使用词对的预测来代替语言模型的预测
 - 使用HS和NEG降低分类复杂度
 - 使用重采样加快训练
 - 新的词对推理数据集来评估词向量的质量

&emsp;&emsp;超参数选择：利用genism做word2vec的时候，词向量的维度和单词数目有没有一个比较好的对照范围呢？

 - dim（vector_size）一般在100-500之间选择
可以按照以下标准来确定dim：初始值词典大小V的1/4次方
 - min_count一般在2-10之间选择
控制词表的大小

***
 如果对您有帮助，麻烦star！！！


