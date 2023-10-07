## ToDO
- [x] fm/deepfm
- [x] DCN
- [x] cin/xdeepfm
- [x] autoint
- [x] AFM
- [ ] fibinet
  - [x] base
  - [ ] fibinet++,senet++
  - [ ] 不同emb_dim (使用interaction模式vi@w -> (bs,1,emb_j),然后就可以了)
- [ ] PNN
- [ ] FNN
- [ ] NFM

## 特征交叉

### FMCrossLayer (deepfm)
就那样

### AutoIntLayer (autoint)

用attention做交叉
inputs: (bs,field_num,emb_size)
MultiHeadAttention(inputs,inputs,inputs)

### DCN
pass

### CIN (xdeepfm)
https://zhuanlan.zhihu.com/p/371849616
核心Cin作为wide部分显式交叉
inputs (bs,field_num,emb_size) -> (bs,cin_1+cin_2+...+cin_n)
把cin^l看作通道数
1. $x^l$每个field与$x^0$每个field点乘作为特征交互 $(bs,cin^l,field_num,emb_size)$
2. 使用shape为$[$cin^l$,field_num]$的矩阵对每个emb_size维度上的矩阵点积后求和压缩
    $(bs,cin^l,field_num,1) * [$cin^l$,field_num]$->$(bs,cin^l,field_num,1)$->(bs,1,1)
    --> 每个emb_size的结果拼接(bs,1,emb_size)
    使用$cin^(l+1)$个矩阵压缩得到$cin^(l+1)$个(bs,1,emb_size)，拼接成(bs,$cin^(l+1)$,emb_size)

inputs: (bs,field_num,emb_size)
交叉
第l层 $x^l: $(bs,cin^l,emb_size)$
第l+1层:
$x^l$每个field与$x^0$每个field点乘 --> $(bs,field_num*cin^l,emb_size)$
- 通过在emb_size切分成emb_size个tensor,然后使用 tf.matmul(a,b,transpose_b=True)实现;
- a是list: $emb_size*[bs,field_num,1]$,
- b是list: $emb_size*[bs,cin^l,1]$
- matmul 将对应位置tensor相乘然后拼接 [bs,field_num,1] @ [bs,1,$cin^l$] --> [bs,field_num,$cin^l$]
- 拼接 [emb_size,bs,field_num,$cin^l$]

压缩：
$(bs,field_num*cin^l,emb_size)$
tf.nn.conv1d(filters=$cin^(l+1)$,kernel_size=$field_num*cin^l$)
第l+1层特征
(bs,$cin^(l+1)$,emb_size))

sumpooling提取每个field的特征
(bs,$cin^(l+1)$))

### BilinearInteraction (fibinet 双线性层)
https://zhuanlan.zhihu.com/p/343572144
fibinet结构
1. SENET控制特征重要性
2. 原始inputs和senet过后的inputs分别送入一个BilinearInteraction得到2个(bs,F*(F-1),emb_size)
3. concat -> (bs,2*F*(F-1),emb_size）
