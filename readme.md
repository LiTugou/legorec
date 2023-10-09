## ToDO
### Rank
#### interaction
- [x] fm/deepfm
- [x] DCN
- [x] cin/xdeepfm
- [x] autoint
- [x] AFM
- [x] fibinet
  - [x] base
  - [ ] fibinet++,senet++
  - [ ] 不同emb_dim (使用interaction模式vi@w -> (bs,1,emb_j),然后就可以了)
- [ ] PNN
- [ ] FNN
- [x] NFM

#### MultiTarget
- [x] ESMM (esay)
- [x] MMoE
- [x] PLE

#### Sequential
- [x] Din
- [x] Dien
- [ ] DSIN
- [x] BST
- [ ] MIMN
- [ ] SIM
- [ ] CAN

#### other
- [x] senet
- [x] ppnet

### Match
- [ ] SDM
- [ ] Mind
- [ ] comirec

### other
- [ ] different training rate
- [ ] warmup
- [ ] 多目标优化
  - [ ] Uncertainty Weight
  - [ ] GradNorm
  - [ ] Dynamic Weight Average
  - [ ] Pareto-Eficient
- [ ] ragged sequence input
- [ ] uniform csv dataset loader