# FORCE 2020 测井岩性识别可靠性评价

本目录用于研究随机深度点划分、按井划分和空间分块划分对 FORCE 2020
岩性识别性能与不确定性评价的影响。

## 当前状态

可行性试运行已完成，结论、真实指标和限制见 [REPORT.md](REPORT.md)。

## 当前阶段

1. 官方数据下载与校验
2. LAS 解析与数据审计
3. 三套划分清单
4. 基线模型与风险—覆盖率评价
5. 按真实结果确定论文题目

## 环境

```powershell
uv sync --project research/force2020-reliability
uv run --project research/force2020-reliability force2020 --help
```

## 重要约束

- 原始数据来自 Zenodo DOI `10.5281/zenodo.4351156`，不得提交到 Git。
- 所有模型拟合、缺失值处理和校准只允许在训练折内完成。
- 随机深度点划分只作为乐观上界，不作为空间泛化证据。