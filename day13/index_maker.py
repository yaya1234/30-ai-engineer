import os
import pandas as pd
import pdfplumber
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment, Font

def select_files():
    """
    弹出文件选择框，支持多选
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    print("👉 正在弹出文件选择框，请选中你要索引的 PDF 文件...")

    file_paths = filedialog.askopenfilenames(
        title="请选择你的法考资料",
        filetypes=[("PDF 文件", "*.pdf")],
    )

    return root.tk.splitlist(file_paths)

def extract_preview_text(filepath, max_chars=1500):
    """
    读取 PDF 前几页内容。
    如果读不到文字（扫描件），会尝试读取第一页的文本块信息作为提示。
    """
    text_content = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            if len(pdf.pages) == 0:
                return "（空文件）"

            # 策略：循环读取前3页，直到提取到足够的文字
            pages_to_read = min(3, len(pdf.pages))

            for i in range(pages_to_read):
                page = pdf.pages[i]
                # 尝试提取文字
                page_text = page.extract_text(x_tolerance=1, y_tolerance=1)

                if page_text:
                    text_content += page_text + "\n"
                    if len(text_content) > max_chars:
                        text_content = text_content[:max_chars] + "..."
                        break
                else:
                    # 如果是扫描件，extract_text 会返回 None
                    # 我们尝试获取一些元数据或者提示用户这是扫描件
                    text_content += f"\n[第{i+1}页似乎是扫描图片，无法直接提取文字] "

    except Exception as e:
        return f"读取错误: {str(e)}"

    return text_content.strip()

def format_excel(writer, df):
    """
    调整 Excel 格式：自动列宽、自动换行
    """
    worksheet = writer.sheets['Sheet1']

    # 设置列宽
    # A列(文件名): 30, B列(大小): 10, C列(预览): 80, D列(路径): 50
    column_widths = {
        'A': 30,
        'B': 15,
        'C': 80,
        'D': 60
    }

    for col_letter, width in column_widths.items():
        worksheet.column_dimensions[col_letter].width = width

    # 设置自动换行和垂直居中
    wrap_alignment = Alignment(wrap_text=True, vertical='center')

    # 遍历所有行进行格式调整
    for row in worksheet.iter_rows():
        for cell in row:
            cell.alignment = wrap_alignment
            # 给表头加粗
            if cell.row == 1:
                cell.font = Font(bold=True)

    # 自动调整行高（openpyxl 没有直接的 auto_fit_row_height，
    # 但设置 wrap_text=True 后，Excel 打开时通常会自动适应，
    # 或者我们可以手动设一个默认高度）
    worksheet.row_dimensions[1].height = 30 # 表头高一点
    for i in range(2, len(df) + 2):
        worksheet.row_dimensions[i].height = 60 # 内容行默认高度

def main():
    # 1. 选择文件
    files = select_files()
    if not files:
        print("❌ 未选择任何文件，程序退出。")
        return

    # 2. 准备数据容器
    data = []
    output_file = "法考资料智能索引表.xlsx"

    print(f"🚀 开始处理 {len(files)} 个文件...")

    # 3. 循环处理（带进度条）
    for file_path in tqdm(files, desc="正在提取内容"):
        file_name = os.path.basename(file_path)
        file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)

        # 提取前1500字
        preview = extract_preview_text(file_path)

        data.append({
            "文件名": file_name,
            "大小 (MB)": file_size_mb,
            "内容预览 (前1500字)": preview,
            "完整路径": file_path
        })

    # 4. 写入 Excel
    df = pd.DataFrame(data)

    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='索引表')
            # 调用格式化函数
            format_excel(writer, df)

        print("-" * 30)
        print(f"✅ 成功！索引表已生成：【{output_file}】")
        print("💡 提示：如果内容显示不全，请在 Excel 中双击单元格下边框。")
        print("-" * 30)

    except PermissionError:
        print(f"❌ 错误：请先关闭正在打开的 Excel 文件：【{output_file}】")

if __name__ == "__main__":
    main()