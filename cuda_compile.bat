@echo off

REM Compile all .cu files in src/cuda directory to .ptx files

cd .\src\cuda\

for %%i in (*.cu) do nvcc -ptx %%i -g