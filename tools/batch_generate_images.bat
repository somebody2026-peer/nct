@echo off
chcp 65001 >nul
echo ============================================
echo 批量生成 AI大模型Agent内容创作专栏配图
echo ============================================
echo.

set OUTPUT_DIR=D:\python_projects\NCT\docs\AI大模型Agent内容创作与自动化发布实战专栏\figures

echo 正在处理第 3-18 篇...
echo.

for %%f in (
    "temp/article_03_context_management.md"
    "temp/article_04_quality_evaluation.md"
    "temp/article_05_playwright_async.md"
    "temp/article_06_ui_interaction_challenges.md"
    "temp/article_07_glm_image_cover.md"
    "temp/article_08_intelligent_degradation.md"
    "temp/article_09_batch_publish_system.md"
    "temp/article_10_streamlit_dashboard.md"
    "temp/article_11_academic_paper_writer.md"
    "temp/article_12_api_documentation_generator.md"
    "temp/article_13_marketing_copy_generator.md"
    "temp/article_14_exercise_generator.md"
    "temp/article_15_cross_language_writing.md"
    "temp/article_16_personal_ip_content_matrix.md"
    "temp/article_17_monetization_saas.md"
    "temp/article_18_future_gpt5_era.md"
) do (
    echo [处理] %%f
    python tools/md_image_generator.py %%f --output-dir "%OUTPUT_DIR%"
    echo.
)

echo ============================================
echo 批量处理完成！
echo ============================================
pause
