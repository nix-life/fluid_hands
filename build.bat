@echo off
set GXX=C:\raylib\w64devkit\bin\g++.exe
set RAYLIB=C:\raylib\raylib\src
set FLAGS=-O2 -std=c++17 -static -I%RAYLIB% -L%RAYLIB%
set LIBS=-lraylib -lws2_32 -lopengl32 -lgdi32 -lwinmm

%GXX% %FLAGS% -fopenmp main.cpp net.cpp -o fluid_hands.exe %LIBS% 2>build.log
if errorlevel 1 (
    echo OpenMP build failed, retrying without -fopenmp...
    %GXX% %FLAGS% main.cpp net.cpp -o fluid_hands.exe %LIBS% 2>build.log
)
if errorlevel 1 (
    echo BUILD FAILED - see build.log
    type build.log
    exit /b 1
)
if exist build.log (
    for %%A in (build.log) do if %%~zA gtr 0 (
        echo --- warnings ---
        type build.log
    ) else del build.log
)
echo BUILD OK: fluid_hands.exe
