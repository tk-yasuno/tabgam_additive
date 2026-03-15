@echo off
REM Compile Diverse GAMs LaTeX Paper

cd 1_Methodology

echo.
echo ============================================
echo Compiling Diverse GAMs Paper (pdfLaTeX)
echo ============================================
echo.

echo [1/3] First pdflatex pass...
pdflatex -interaction=nonstopmode diverse_gams_bridge_deterioration.tex

echo.
echo [2/3] Second pdflatex pass (resolve references)...
pdflatex -interaction=nonstopmode diverse_gams_bridge_deterioration.tex

echo.
echo [3/3] Third pdflatex pass (finalize)...
pdflatex -interaction=nonstopmode diverse_gams_bridge_deterioration.tex

if exist diverse_gams_bridge_deterioration.pdf (
    echo.
    echo ============================================
    echo Compilation successful!
    echo Output: diverse_gams_bridge_deterioration.pdf
    echo ============================================
    start diverse_gams_bridge_deterioration.pdf
) else (
    echo.
    echo ============================================
    echo Compilation failed! Check the .log file.
    echo ============================================
)

cd ..
pause
