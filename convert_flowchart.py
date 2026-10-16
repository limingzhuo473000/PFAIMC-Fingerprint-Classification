"""
将HTML流程图转换为PNG/JPG图片
"""
import os
import sys
from pathlib import Path

# 添加backend目录到路径
backend_dir = Path(__file__).parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

def convert_html_to_image(html_path: str, output_path: str, fmt: str = "png"):
    """
    将HTML文件转换为图片
    
    参数:
        html_path: HTML文件路径
        output_path: 输出图片路径
        fmt: 输出格式 (png 或 jpg)
    """
    try:
        # 尝试使用 imgkit (需要 wkhtmltoimage)
        import imgkit
        
        options = {
            'format': fmt,
            'quality': 95 if fmt == 'jpg' else None,
            'width': 1200,
            'enable-local-file-access': None,
        }
        
        imgkit.from_file(html_path, output_path, options=options)
        print(f"✓ 成功生成图片: {output_path}")
        return True
        
    except ImportError:
        print("imgkit 未安装，尝试其他方法...")
    except Exception as e:
        print(f"imgkit 转换失败: {e}")
    
    try:
        # 尝试使用 weasyprint + cairosvg
        from weasyprint import HTML
        import cairosvg
        
        # 先转为PDF
        pdf_path = output_path.replace('.png', '.pdf').replace('.jpg', '.pdf')
        html = HTML(filename=html_path)
        html.write_pdf(pdf_path)
        
        # PDF转PNG
        cairosvg.svg2png(
            url=pdf_path,
            write_to=output_path,
            output_width=1200
        )
        print(f"✓ 成功生成图片: {output_path}")
        return True
        
    except ImportError:
        print("weasyprint/cairosvg 未安装")
    except Exception as e:
        print(f"weasyprint 转换失败: {e}")
    
    try:
        # 尝试使用 selenium (需要安装 Chrome/Edge 和 webdriver)
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        import time
        
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument(f'--window-size=1400,1000')
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(f'file:///{os.path.abspath(html_path)}')
        time.sleep(3)  # 等待渲染
        
        # 截图
        driver.save_screenshot(output_path)
        driver.quit()
        
        print(f"✓ 成功生成图片: {output_path}")
        return True
        
    except ImportError:
        print("selenium 未安装")
    except Exception as e:
        print(f"selenium 转换失败: {e}")
    
    # 如果所有方法都失败，提供手动方案
    print("\n⚠ 自动转换失败，请使用以下手动方法:")
    print(f"1. 在浏览器中打开: {html_path}")
    print(f"2. 按 Ctrl+P 打印")
    print(f"3. 选择'另存为PDF'或直接截图")
    print(f"4. 或使用在线工具转换HTML为图片")
    return False


if __name__ == "__main__":
    # HTML文件路径
    html_file = r"F:\预处理\预处理\网页端 - 副本(全多进程+计时)\网页端 - 副本(全多进程+计时)\algorithm_flowchart.html"
    
    # 输出路径
    output_dir = Path(__file__).parent
    png_output = output_dir / "algorithm_flowchart.png"
    jpg_output = output_dir / "algorithm_flowchart.jpg"
    
    print("=" * 60)
    print("HTML流程图转图片工具")
    print("=" * 60)
    print(f"\n输入文件: {html_file}")
    print(f"输出PNG: {png_output}")
    print(f"输出JPG: {jpg_output}")
    print()
    
    # 检查HTML文件是否存在
    if not Path(html_file).exists():
        print(f"❌ 错误: HTML文件不存在: {html_file}")
        sys.exit(1)
    
    # 尝试生成PNG
    print("正在生成PNG...")
    success_png = convert_html_to_image(str(html_file), str(png_output), "png")
    
    # 尝试生成JPG
    print("\n正在生成JPG...")
    success_jpg = convert_html_to_image(str(html_file), str(jpg_output), "jpg")
    
    print("\n" + "=" * 60)
    if success_png or success_jpg:
        print("✓ 转换完成！")
    else:
        print("⚠ 自动转换失败，请参考上述手动方法")
    print("=" * 60)
