## Discovering Latent Relationship for Temporal Knowledge Graph Reasoning

Simply run `python main.py`

*Requirments*：torch, dgl(cuda version)

## Arguments explanation

| Argument         | Default | Description                                                                   |
| ---------------- | ------- | ----------------------------------------------------------------------------- |
| --gpu            | 0       | Set cuda device                                                               |
| --dataset  -d    | ICEWS14 | Dataset used here, please choose from ICEWS14, ICEWS18, ICEWS05-15, ICEWS14s |
| --dropout        | 0.2     | Dropout rate                                                                  |
| --n-hidden       | 200     | Layer hidden dimension                                                        |
| --n-layers       | 2       | Number of layers of one GNN                                                   |
| --history-len    | 3       | history length                                                                |
| --lr             | 0.001   | learning rate                                                                 |
| --early_stop     | 5       | early stop epochs                                                             |
| --easy_copy      | 0       | Remove most print results, only keep the final output                         |

