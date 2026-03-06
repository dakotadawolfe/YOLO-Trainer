@echo off
cd /d "%~dp0.."
if "%~1"=="" (set TARGET=mound_dharok) else (set TARGET=%~1)
if "%TARGET:~0,2%"=="--" set TARGET=%TARGET:~2%
python "YOLO Trainer/yolo_workflow.py" full %TARGET%
pause
