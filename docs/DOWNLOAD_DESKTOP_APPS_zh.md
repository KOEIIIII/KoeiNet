# 下载 KoeiNet 桌面应用

Windows 桌面应用下载地址：

```text
https://github.com/KOEIIIII/KoeiNet/releases
```

## 需要下载的文件

请下载所有分卷文件：

```text
KoeiNet_Windows.zip.part001
KoeiNet_Windows.zip.part002
KoeiNet_Windows.zip.part003
...
```

同时下载：

```text
Join_And_Extract.bat
README_FIRST.txt
```

所有文件必须放在同一个文件夹中。

## 如何运行

1. 双击 `Join_And_Extract.bat`。
2. 等待程序自动合并并解压分卷包。
3. 打开解压后的 `KoeiNet_Windows` 文件夹。
4. 双击 `Program01_AnalysisVisualization.exe` 运行 Program 01。
5. 双击 `Program02_ScoringProblemDetection.exe` 运行 Program 02。

不要把 exe 文件单独移出 `KoeiNet_Windows` 文件夹。`_internal` 文件夹中包含两个程序共用的运行库和资源文件。

## 维护者打包方式

使用 PyInstaller 构建 Program 01 和 Program 02 后，运行：

```powershell
python scripts\package_windows_release.py
```

输出目录为：

```text
release_packages/KoeiNet_Windows/
```

将生成的所有分卷文件、`Join_And_Extract.bat` 和 `README_FIRST.txt` 上传到 GitHub Release。
