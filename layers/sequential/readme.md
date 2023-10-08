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

## BST
用户历史行为序列拼接target item
再把position embdding concat到emb_size维度
丢到multiheadattention
没啥好包装的