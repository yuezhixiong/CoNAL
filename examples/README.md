## Training the CoNAL learned architecture

### Retrain with libmtl
The retraining process of CoNAL can combine seamlessly with libmtl [[1]](#1)

- [LibMTL package](https://github.com/median-research-group/LibMTL)


Put the CoNAL.py file in the `LibMTL/architecuture` folder


```
cd nyu
```

Train the architecture on the NYUv2 dataset [[1]](#1)  by the following command:

```
python train_nyu.py --arch CoNAL --weighting RLW --dataset_path PATH/nyuv2 
```

- ``weighting``: The weighting strategy.
- ``arch``: The MTL architecture.
- ``dataset_path``: The path of the NYUv2 dataset.


### References

<a id="1">[1]</a> Baijiong Lin and Yu Zhang. LibMTL: A Python Library for Multi-Task Learning.
