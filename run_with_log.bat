@echo off
REM CEVAE 实验结果自动记录脚本
REM 用法: run_with_log.bat [数据集] [其他参数]

setlocal enabledelayedexpansion

REM 获取当前时间戳（格式：YYYYMMDD_HHMMSS）
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set timestamp=%datetime:~0,8%_%datetime:~8,6%

REM 解析参数
set DATASET=%1
set MODE=standard

REM 判断数据集和模式
if "%DATASET%"=="ihdp" (
    set FILENAME=ihdp_standard_%timestamp%.txt
) else if "%DATASET%"=="ihdp1000" (
    REM 检查是否使用分离模式
    echo %2 | findstr /C:"separate_reps" >nul
    if !errorlevel!==0 (
        set MODE=separate
    ) else (
        set MODE=combined
    )
    set FILENAME=ihdp1000_%MODE%_%timestamp%.txt
) else if "%DATASET%"=="twins" (
    set FILENAME=twins_standard_%timestamp%.txt
) else (
    echo 用法: run_with_log.bat [数据集] [其他参数]
    echo 数据集选项: ihdp, ihdp1000, twins
    exit /b 1
)

REM 创建 record 目录（如果不存在）
if not exist record mkdir record

REM 显示实验信息
echo ========================================
echo CEVAE 实验结果自动记录
echo ========================================
echo 数据集: %DATASET%
echo 模式: %MODE%
echo 时间戳: %timestamp%
echo 输出文件: record\%FILENAME%
echo ========================================
echo.

REM 运行训练并保存输出
echo 开始训练...
python cevae_ihdp.py %* | tee record\%FILENAME%

echo.
echo ========================================
echo 实验完成！
echo 结果已保存到: record\%FILENAME%
echo ========================================

endlocal
