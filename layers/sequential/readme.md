## ToDO
- [x] Din
- [ ] Dien
- [ ] DSIN
- [x] BST
- [ ] MIMN
- [ ] SIM
- [ ] CAN

## 简介
### Din
att_weight=MLP([query,key,query-key,query*key]) -> (bs,seqlen,1)
然后mask加权

### Dien
1. 兴趣抽取层
使用历史点击输入gru提取兴趣H
2. 兴趣演化
计算query对hi的att_score
使用att_score改造的gru对兴趣H抽取

## BST
用户历史行为序列拼接target item
再把position embdding concat到emb_size维度
丢到multiheadattention
没啥好包装的