@echo off

cd .\cuda\

for %%i in (*.cu) do nvcc -ptx %%i