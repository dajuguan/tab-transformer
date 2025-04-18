<img src="./tab.png" width="250px"></img>

## 环境
在周博Ubuntu机器上默认的环境

## 论文数据说明
14,16, 58
现在用的16,58
DE不好的采用three_d_reg_ft_de_ofd.py的第4个模型
DE好的采用three_d_reg_ft_de.py的第4个模型
注意一定要把不同的模型用不同的名字，否则把模型覆盖了
### spanwise loss distribution
- loss: three_d_model_loss.py
- loss distribution: python3 three_d_model_com.py (修改注释)
- caseI, II 不带UQ, python3 three_d_model_com.py
- caseI UQ: python3 three_d_reg_ft_de.py imgs/ft/16.png
- caseII UQ 不确定性高: python3 three_d_reg_ft_de_ofd.py imgs/ft_ofm/58.png
- caseII UQ不确定性低: python3 three_d_reg_ft_de_id58.py imgs/ft_withpaper/1.id.png (修改了start)

## Stacking line
- loss: stacking_line.py stacking_line.png
- 弯掠角度和弯掠高对比: stacking_comp.py
- CaseII域外结果预测: stacking_ft_de.case2.ofd.py stacking_line_58.png (数据index实际上是57)
- CaseII域内结果预测: stacking_ft_de.case2.id.py stacking_line_58.id.png

## 涡轮算例
- spanwise loss: three_d_reg_ft_de_withpaper.py imgs/ft_withpaper/135.withpaper.png
- stacking line: stacking_ft_de.withpaper.py stacking_line_135.withpaper.png

## Tab Transformer

Implementation of <a href="https://arxiv.org/abs/2012.06678">Tab Transformer</a>, attention network for tabular data, in Pytorch. This simple architecture came within a hair's breadth of GBDT's performance.

Update: Amazon AI claims to have beaten GBDT with Attention on <a href="https://arxiv.org/abs/2311.11694">a real-world tabular dataset (predicting shipping cost)</a>.

## Install

```bash
system python
pip install einops
$ pip install tab-transformer-pytorch
```

## Usage

```python
import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer

cont_mean_std = torch.randn(10, 2)

model = TabTransformer(
    categories = (10, 5, 6, 5, 8),      # tuple containing the number of unique values within each category
    num_continuous = 10,                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = 1,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1,                   # feed forward dropout
    mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
)

x_categ = torch.randint(0, 5, (1, 5))     # category values, from 0 - max number of categories, in the order as passed into the constructor above
x_cont = torch.randn(1, 10)               # assume continuous values are already normalized individually

pred = model(x_categ, x_cont) # (1, 1)
```

## FT Transformer

<img src="./tab-vs-ft.png" width="500px"></img>

<a href="https://arxiv.org/abs/2106.11959v2">This paper</a> from Yandex improves on Tab Transformer by using a simpler scheme for embedding the continuous numerical values as shown in the diagram above, courtesy of <a href="https://www.reddit.com/r/MachineLearning/comments/yhdqlj/project_improving_deep_learning_for_tabular_data/">this reddit post</a>.

Included in this repository just for convenient comparison to Tab Transformer

```python
import torch
from tab_transformer_pytorch import FTTransformer

model = FTTransformer(
    categories = (10, 5, 6, 5, 8),      # tuple containing the number of unique values within each category
    num_continuous = 10,                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = 1,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1                    # feed forward dropout
)

x_categ = torch.randint(0, 5, (1, 5))     # category values, from 0 - max number of categories, in the order as passed into the constructor above
x_numer = torch.randn(1, 10)              # numerical value

pred = model(x_categ, x_numer) # (1, 1)
```

## Unsupervised Training

To undergo the type of unsupervised training described in the paper, you can first convert your categories tokens to the appropriate unique ids, and then use <a href="https://github.com/lucidrains/electra-pytorch">Electra</a> on `model.transformer`.

## Todo

- [ ] consider https://arxiv.org/abs/2203.05556

## Citations

```bibtex
@misc{huang2020tabtransformer,
    title   = {TabTransformer: Tabular Data Modeling Using Contextual Embeddings},
    author  = {Xin Huang and Ashish Khetan and Milan Cvitkovic and Zohar Karnin},
    year    = {2020},
    eprint  = {2012.06678},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@article{Gorishniy2021RevisitingDL,
    title   = {Revisiting Deep Learning Models for Tabular Data},
    author  = {Yu. V. Gorishniy and Ivan Rubachev and Valentin Khrulkov and Artem Babenko},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2106.11959}
}
```
