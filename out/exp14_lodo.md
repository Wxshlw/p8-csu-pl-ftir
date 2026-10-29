# LODO 阈值验证（exp14）

| Held-out | 选中的 CNR | 选中的 γ | 训练 balanced acc | 真值 pass | 预测 pass | 正确 | 中位比 |
|---|---|---|---|---|---|---|---|
| D1 | 10 | 1 | 0.900 | pass | pass | Y | 0.899 |
| D2 | 10 | 1 | 1.000 | fail | pass | N | 1.003 |
| D3 | 10 | 1 | 0.875 | fail | fail | Y | 2.232 |
| D7w | 10 | 1 | 0.875 | fail | fail | Y | 1.266 |
| D7p | 10 | 1 | 0.875 | fail | fail | Y | 1.808 |
| D7d | 10 | 0.1 | 1.000 | pass | fail | N | 0.925 |
| D8 | 10 | 1 | 0.875 | fail | fail | Y | 1.054 |

LODO 正确 5/7；固定 10/1 正确 6/7。