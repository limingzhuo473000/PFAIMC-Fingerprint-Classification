"""
使用Playwright将HTML流程图转换为PNG/JPG图片
"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

async def convert_html_to_image(html_path: str, output_png: str, output_jpg: str):
    """
    使用Playwright将HTML文件转换为PNG和JPG图片
    """
    html_path = Path(html_path).resolve()
    
    if not html_path.exists():
        print(f"❌ HTML文件不存在: {html_path}")
        return False
    
    async with async_playwright() as p:
        # 启动浏览器
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1400, 'height': 2000})
        
        # 加载HTML文件
        file_url = f'file:///{html_path.as_posix()}'
        print(f"正在加载HTML文件...")
        await page.goto(file_url, wait_until='networkidle', timeout=30000)
        
        # 等待页面完全渲染
        await asyncio.sleep(3)
        
        # 获取页面实际高度
        page_height = await page.evaluate('document.documentElement.scrollHeight')
        page_width = await page.evaluate('document.documentElement.scrollWidth')
        
        print(f"页面尺寸: {page_width} x {page_height}")
        
        # 调整视口以匹配内容
        await page.set_viewport_size({'width': page_width, 'height': page_height})
        await asyncio.sleep(1)
        
        # 截图PNG
        print(f"正在生成PNG: {output_png}")
        await page.screenshot(path=str(output_png), full_page=True, type='png')
        print(f"✓ PNG生成成功")
        
        # 截图JPG (quality=95)
        print(f"正在生成JPG: {output_jpg}")
        await page.screenshot(path=str(output_jpg), full_page=True, type='jpeg', quality=95)
        print(f"✓ JPG生成成功")
        
        await browser.close()
    
    return True


if __name__ == "__main__":
    import sys
    
    # HTML文件路径
    html_file = r"F:\预处理\预处理\网页端 - 副本(全多进程+计时)\网页端 - 副本(全多进程+计时)\algorithm_flowchart.html"
    
    # 输出路径
    output_dir = Path(html_file).parent
    png_output = output_dir / "algorithm_flowchart.png"
    jpg_output = output_dir / "algorithm_flowchart.jpg"
    
    print("=" * 60)
    print("HTML流程图转图片工具 (Playwright)")
    print("=" * 60)
    print(f"\n输入文件: {html_file}")
    print(f"输出PNG: {png_output}")
    print(f"输出JPG: {jpg_output}")
    print()
    
    try:
        success = asyncio.run(convert_html_to_image(html_file, png_output, jpg_output))
        
        print("\n" + "=" * 60)
        if success:
            print("✓ 转换完成！")
            print(f"  PNG: {png_output}")
            print(f"  JPG: {jpg_output}")
        else:
            print("⚠ 转换失败")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
