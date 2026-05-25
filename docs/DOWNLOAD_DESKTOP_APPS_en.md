# Download KoeiNet Desktop Applications

Windows desktop applications are distributed from:

```text
https://github.com/KOEIIIII/KoeiNet/releases
```

## Files To Download

Download every package part:

```text
KoeiNet_Windows.zip.part001
KoeiNet_Windows.zip.part002
KoeiNet_Windows.zip.part003
...
```

Also download:

```text
Join_And_Extract.bat
README_FIRST.txt
```

All files must be placed in the same folder.

## How To Run

1. Double-click `Join_And_Extract.bat`.
2. Wait until the package is joined and extracted.
3. Open the extracted `KoeiNet_Windows` folder.
4. Double-click `Program01_AnalysisVisualization.exe` to run Program 01.
5. Double-click `Program02_ScoringProblemDetection.exe` to run Program 02.

Do not move the exe files away from the `KoeiNet_Windows` folder. The `_internal` folder contains the shared runtime and resources required by both applications.

## For Maintainers

After building Program 01 and Program 02 with PyInstaller, create the split release package with:

```powershell
python scripts\package_windows_release.py
```

The output is written to:

```text
release_packages/KoeiNet_Windows/
```

Upload all generated package parts, `Join_And_Extract.bat`, and `README_FIRST.txt` to a GitHub Release.
