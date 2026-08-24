# 示例与仿真数据

- `generate_simulation.py`：按固定随机种子生成可复现的菲索干涉图；
- `plot_noll_zernike_coolwarm.py`：从 `n=1` 开始，将同阶 Noll Zernike 无标注地排成塔形；
- `simulated_data/`：参考元件、待测元件、真值图和参数说明。

从仓库根目录运行：

```powershell
python examples/generate_simulation.py
```

生成 Zernike 总览图：

```powershell
python examples/plot_noll_zernike_coolwarm.py --show
```

可使用 `--output result.pdf` 导出矢量 PDF，或通过 `--max-n 6` 增加塔形阶数。
