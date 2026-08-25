# FizeauGUI

面向本科教学与实验室算法验证的菲索干涉图处理软件。使用 Python、PySide6
和 Matplotlib 实现单帧载波相位提取、相位展开、Zernike 拟合与完整口径
波前分析。

> A PySide6 desktop application for Fizeau interferogram phase extraction,
> phase unwrapping, Zernike fitting, and full-aperture wavefront analysis.

[下载 Windows 版](https://github.com/MarvelousZK/FizeauGUI/releases/latest/download/FizeauGUI-Windows-x64.exe)
· [使用说明](docs/使用说明.md)
· [算法文档](docs/algorithms/)

## 主要功能

- 载入参考元件与待测元件干涉图，自动检测并交互调整圆形有效孔径；
- “准备 → 处理 → 结果”三段式教学流程；
- 四种单帧载波相位提取方法：
  - 经典 Fourier-transform 法（Takeda FT）：手动观察二维频谱、点击一级谱峰并设置 Gaussian 边带窗；
  - 线性 $N$ 点空间载波相移（包含圆形孔径边界约束）；
  - Luo 单载频自适应空间相移 `adapt2`；
  - 掩膜自适应 Qian 加窗傅里叶滤波 `wft2`；
- PDV 质量图引导相位展开；
- Zernike 拟合、逐项去除、RMS/PV 与完整有效口径统计；
- FT 五步诊断、相位置信度、交互式 3D 波面与两点剖面；
- 可复现的教学仿真数据和一键结果导出。

## 从源码运行

需要 Python 3.10–3.12：

```powershell
git clone https://github.com/MarvelousZK/FizeauGUI.git
cd FizeauGUI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m fizeau_gui
```

安装后也可以使用命令：

```powershell
fizeau-gui
```

## 教学仿真数据

仓库已经包含一组确定性参考图和待测图，位于
[`examples/simulated_data/`](examples/simulated_data/)。重新生成：

```powershell
python examples/generate_simulation.py
```

## 项目结构

```text
FizeauGUI/
├─ src/fizeau_gui/        Python 应用包、算法核心、主题和图标资源
├─ docs/                  使用说明、算法原理和教学配图
├─ examples/              仿真生成器与可复现示例数据
├─ tests/                 算法回归与 GUI 冒烟测试
├─ pyproject.toml         Python 包和命令行入口配置
├─ requirements.txt       开发依赖
├─ README.md
└─ LICENSE
```

## 测试

测试脚本可直接从仓库根目录运行：

```powershell
python tests/test_core.py
python tests/test_takeda_ft.py
python tests/test_luo_adapt2.py
python tests/test_masked_shift.py
python tests/test_wft2.py
python tests/test_qg_verify.py
python tests/test_unwrap.py
python tests/test_gui_smoke.py
```

## 文档与参考

本实验的数据处理流程主要由单帧相位提取、相位展开和 Zernike 分解三个计算模块组成。软件的具体操作步骤、参数设置及结果说明参见[使用说明](docs/使用说明.md)。

第一部分为单帧相位提取。经典 Fourier Transform 方法以 Takeda 教授提出的傅里叶变换条纹分析方法为理论基础 [1]，相关频谱选峰、边带滤波及相位恢复过程可结合 [FT 教学配图索引](docs/figures/figure_captions.md)学习。空间载波相移方法参考了相移干涉和线性空间载波解调的前期研究 [2–4]。该方法通过参考波前与被测波前之间的相对倾斜引入近似线性载波，并把一个条纹周期内的相邻像素作为等效相移状态；本软件另外加入圆形有效孔径的窗口约束和短弦边界处理，具体原理参见[空间载波相移方法](docs/algorithms/空间载波相移方法.md)。软件中的另一种自适应载波方法来源于四川大学李大海教授团队此前的研究工作 [5]。WFT 方法参考了南洋理工大学钱克矛教授的论文及其开源程序 [6]。对于噪声、载频变化和孔径边界条件下的进一步处理思路，可继续阅读[掩膜约束自适应稳健局部相位法](docs/algorithms/掩膜约束自适应稳健局部相位法.md)。

第二部分为相位展开。软件采用 Herráez 教授提出的质量引导相位展开思想 [7]，使相位能够按照可靠性由高到低的顺序进行展开，从而减小低质量区域对整体结果的影响。

第三部分为 Zernike 分解。Zernike 圆多项式最早由 Zernike 提出 [8]，本实验采用 Noll 编号及归一化方式 [9]，在圆形有效孔径内对波前进行最小二乘拟合。相关公式在原 MATLAB 程序中标注为《光学车间检测》第 383 页式（13.47）；可核验的英文第三版内容位于第 13 章、第 498–546 页 [10]。编号规则、归一化方式以及本项目的具体实现参见[Zernike 拟合、Noll 编号与本项目实现](docs/algorithms/Zernike拟合与Noll编号.md)。

### 完整参考文献

1. M. Takeda, H. Ina, and S. Kobayashi, “Fourier-transform method of fringe-pattern analysis for computer-based topography and interferometry,” *Journal of the Optical Society of America* **72**(1), 156–160 (1982). [DOI: 10.1364/JOSA.72.000156](https://doi.org/10.1364/JOSA.72.000156)
2. J. H. Bruning, D. R. Herriott, J. E. Gallagher, D. P. Rosenfeld, A. D. White, and D. J. Brangaccio, “Digital wavefront measuring interferometer for testing optical surfaces and lenses,” *Applied Optics* **13**(11), 2693–2703 (1974). [DOI: 10.1364/AO.13.002693](https://doi.org/10.1364/AO.13.002693)
3. B. T. Kimbrough, “Pixelated mask spatial carrier phase shifting interferometry algorithms and associated errors,” *Applied Optics* **45**(19), 4554–4562 (2006). [DOI: 10.1364/AO.45.004554](https://doi.org/10.1364/AO.45.004554)
4. J. Xu, Q. Xu, and H. Peng, “Spatial carrier phase-shifting algorithm based on least-squares iteration,” *Applied Optics* **47**(29), 5446–5453 (2008). [DOI: 10.1364/AO.47.005446](https://doi.org/10.1364/AO.47.005446)
5. P. Luo, D. Li, R. Wang, X. Zhang, X. Li, and W. Zhao, “Phase-extraction algorithm for a single-shot spatial-carrier orthogonal fringe pattern with least squares method,” *Optical Engineering* **59**(2), 024103 (2020). [DOI: 10.1117/1.OE.59.2.024103](https://doi.org/10.1117/1.OE.59.2.024103)
6. K. Qian, “Windowed Fourier transform for fringe pattern analysis,” *Applied Optics* **43**(13), 2695–2702 (2004). [DOI: 10.1364/AO.43.002695](https://doi.org/10.1364/AO.43.002695)
7. M. A. Herráez, D. R. Burton, M. J. Lalor, and M. A. Gdeisat, “Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path,” *Applied Optics* **41**(35), 7437–7444 (2002). [DOI: 10.1364/AO.41.007437](https://doi.org/10.1364/AO.41.007437)
8. F. Zernike, “Beugungstheorie des Schneidenverfahrens und seiner verbesserten Form, der Phasenkontrastmethode,” *Physica* **1**, 689–704 (1934). [DOI: 10.1016/S0031-8914(34)80259-5](https://doi.org/10.1016/S0031-8914(34)80259-5)
9. R. J. Noll, “Zernike polynomials and atmospheric turbulence,” *Journal of the Optical Society of America* **66**(3), 207–211 (1976). [DOI: 10.1364/JOSA.66.000207](https://doi.org/10.1364/JOSA.66.000207)
10. V. N. Mahajan, “Zernike Polynomial and Wavefront Fitting,” in D. Malacara (ed.), *Optical Shop Testing*, 3rd ed., pp. 498–546, Wiley (2007). [DOI: 10.1002/9780470135976.ch13](https://doi.org/10.1002/9780470135976.ch13)
11. H. Schreiber and J. H. Bruning, “Phase Shifting Interferometry,” in D. Malacara (ed.), *Optical Shop Testing*, 3rd ed., pp. 547–666, Wiley (2007). [DOI: 10.1002/9780470135976.ch14](https://doi.org/10.1002/9780470135976.ch14)

## 计量说明

- RMS/PV 始终基于完整有效口径计算，不通过缩小统计区域降低指标；
- mask 外区域不参与展开、拟合与统计，并以透明区域显示；
- 3D 预览使用轻量网格，不会改变完整分辨率数据；
- 实际计量精度取决于采集系统标定、参考面、环境稳定性与条纹质量。

本项目主要用于教学、算法验证和实验室数据分析；在未经标定与不确定度评估
时，不应直接作为商业计量结论。

## License

[MIT](LICENSE) © 2026 Zekun Zhang
