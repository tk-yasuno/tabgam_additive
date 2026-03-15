# Compile Diverse GAMs LaTeX Paper

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "Compiling Diverse GAMs Paper (pdfLaTeX)" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

Set-Location "1_Methodology"

Write-Host "[1/3] First pdflatex pass..." -ForegroundColor Yellow
pdflatex -interaction=nonstopmode diverse_gams_bridge_deterioration.tex | Out-Null

Write-Host "[2/3] Second pdflatex pass (resolve references)..." -ForegroundColor Yellow
pdflatex -interaction=nonstopmode diverse_gams_bridge_deterioration.tex | Out-Null

Write-Host "[3/3] Third pdflatex pass (finalize)..." -ForegroundColor Yellow
pdflatex -interaction=nonstopmode diverse_gams_bridge_deterioration.tex | Out-Null

if (Test-Path "diverse_gams_bridge_deterioration.pdf") {
    Write-Host "`n============================================" -ForegroundColor Green
    Write-Host "Compilation successful!" -ForegroundColor Green
    Write-Host "Output: diverse_gams_bridge_deterioration.pdf" -ForegroundColor Green
    Write-Host "============================================`n" -ForegroundColor Green
    
    Start-Process "diverse_gams_bridge_deterioration.pdf"
} else {
    Write-Host "`n============================================" -ForegroundColor Red
    Write-Host "Compilation failed! Check the .log file." -ForegroundColor Red
    Write-Host "============================================`n" -ForegroundColor Red
    
    if (Test-Path "diverse_gams_bridge_deterioration.log") {
        Write-Host "Last 20 lines of log file:" -ForegroundColor Yellow
        Get-Content "diverse_gams_bridge_deterioration.log" | Select-Object -Last 20
    }
}

Set-Location ..
