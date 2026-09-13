@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem Smart Tennis Analytics - Integracion automatica Sackmann/TML
rem Ejecutar desde cualquier carpeta del repositorio.
rem ============================================================

cd /d "%~dp0\..\.."
set "PROJECT_ROOT=%CD%"
set "PYTHON=%PROJECT_ROOT%\.venv-upcoming\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] No existe el Python del entorno virtual:
    echo         %PYTHON%
    exit /b 1
)

set "BUILDER=%PROJECT_ROOT%\ingestion\v2\build_jeff_sackmann_with_odds_tml.py"
set "RECONCILER=%PROJECT_ROOT%\ingestion\v2\reconcile_tml_integration.py"
set "PARQUET=%PROJECT_ROOT%\data\processed\jeff_sackmann_with_odds.parquet"
set "REPORT_DIR=%PROJECT_ROOT%\data\processed\tml_integration_reports"
set "TML_DIR=%PROJECT_ROOT%\data\raw\tml_v2\current_season"
set "SACKMANN_DIR=%PROJECT_ROOT%\data\raw\tennis_atp-master"
set "ODDS_DIR=%PROJECT_ROOT%\data\raw\tennis_data_co_uk_v2"

if not exist "%BUILDER%" (
    echo [ERROR] No existe el constructor: %BUILDER%
    exit /b 1
)

if not exist "%RECONCILER%" (
    echo [ERROR] No existe el reconciliador: %RECONCILER%
    exit /b 1
)

for %%F in ("2026.csv" "2026_atp_quali.csv" "2026_challenger.csv") do (
    if not exist "%TML_DIR%\%%~F" (
        echo [ERROR] Falta: %TML_DIR%\%%~F
        exit /b 1
    )
)

if not exist "%SACKMANN_DIR%\atp_players.csv" (
    echo [ERROR] Falta: %SACKMANN_DIR%\atp_players.csv
    exit /b 1
)

set "ATP_DATABASE=%TML_DIR%\ATP_Database.csv"
if not exist "%ATP_DATABASE%" set "ATP_DATABASE=%PROJECT_ROOT%\data\raw\tml_v2\ATP_Database.csv"
if not exist "%ATP_DATABASE%" (
    echo [ERROR] No se encuentra ATP_Database.csv en current_season ni en tml_v2.
    exit /b 1
)

if not exist "%ODDS_DIR%" (
    echo [ERROR] No existe el directorio de cuotas: %ODDS_DIR%
    exit /b 1
)

if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"

set "LOG=%REPORT_DIR%\run_tml_integration.log"
echo Inicio: %DATE% %TIME% > "%LOG%"
echo Proyecto: %PROJECT_ROOT% >> "%LOG%"

rem Evita que Streamlit o Python mantengan el Parquet abierto.
taskkill /F /IM streamlit.exe >nul 2>nul
taskkill /F /IM pythonw.exe >nul 2>nul

call :run_builder
if errorlevel 1 goto :failed

call :run_reconciliation
if errorlevel 1 goto :review_required

call :run_validation
if errorlevel 1 goto :failed

echo.
echo [OK] Integracion completada correctamente.
echo [OK] Parquet: %PARQUET%
echo [OK] Informes: %REPORT_DIR%
echo [OK] Log: %LOG%
exit /b 0

:run_builder
echo.
echo ============================================================
echo [1/3] Construyendo Sackmann + TML + cuotas
echo ============================================================
"%PYTHON%" "%BUILDER%" ^
  --sackmann-dir "%SACKMANN_DIR%" ^
  --odds-dir "%ODDS_DIR%" ^
  --tml-main "%TML_DIR%\2026.csv" ^
  --tml-qualifying "%TML_DIR%\2026_atp_quali.csv" ^
  --tml-challenger "%TML_DIR%\2026_challenger.csv" ^
  --tml-player-database "%ATP_DATABASE%" ^
  --atp-players "%SACKMANN_DIR%\atp_players.csv" ^
  --unmatched-player-policy synthetic ^
  --output "%PARQUET%" ^
  --report-dir "%REPORT_DIR%" >> "%LOG%" 2>&1
exit /b %ERRORLEVEL%

:run_reconciliation
echo.
echo ============================================================
echo [2/3] Reconciliando identidades y calidad de datos
echo ============================================================
"%PYTHON%" "%RECONCILER%" ^
  --parquet "%PARQUET%" ^
  --report-dir "%REPORT_DIR%" ^
  --max-synthetic-players 15 >> "%LOG%" 2>&1
exit /b %ERRORLEVEL%

:run_validation
echo.
echo ============================================================
echo [3/3] Validando resultado final
echo ============================================================
"%PYTHON%" -c "import json,pandas as pd; p=r'%PARQUET%'; x=pd.read_parquet(p,columns=['tourney_id','competition_type','match_num','winner_id','loser_id','tourney_date','source_origin']); assert not x.duplicated(['tourney_id','competition_type','match_num']).any(), 'Claves duplicadas'; assert not x[['winner_id','loser_id']].isna().any().any(), 'IDs nulos'; print('Filas:',len(x)); print('Fecha maxima:',pd.to_numeric(x.tourney_date,errors='coerce').max()); print(x.groupby(['source_origin','competition_type'],dropna=False).size().to_string())" >> "%LOG%" 2>&1
exit /b %ERRORLEVEL%

:review_required
echo.
echo [REVISION REQUERIDA] El constructor termino, pero quedan demasiados IDs sinteticos.
echo Revisa:
echo   %REPORT_DIR%\remaining_synthetic_players.csv
echo   %REPORT_DIR%\synthetic_player_candidates.csv
echo   %REPORT_DIR%\player_id_canonical_map.csv
echo.
echo El Parquet original no se reemplazo durante la reconciliacion.
echo Log: %LOG%
exit /b 2

:failed
echo.
echo [ERROR] El proceso ha fallado. Revisa el log:
echo         %LOG%
exit /b 1
