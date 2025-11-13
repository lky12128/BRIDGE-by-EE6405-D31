import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import zipfile
import tarfile
import gzip
import shutil 
# 
TARGET_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/"

DOWNLOAD_DIR = "amazon_reviews_2023"

FILE_EXTENSIONS = [".jsonl.gz"]

DELETE_ARCHIVE_AFTER_EXTRACTION = True


def get_dataset_links(page_url, extensions):
    """
    访问目标网页，解析HTML，并找出所有指向目标文件类型的链接。
    """
    print(f"正在访问页面: {page_url}")
    links = []
    try:
        response = requests.get(page_url)
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if any(href.endswith(ext) for ext in extensions):
                full_url = urljoin(page_url, href)
                links.append(full_url)
    
    except requests.exceptions.RequestException as e:
        print(f"错误：无法访问页面 {page_url}。请检查URL和网络连接。")
        print(e)
        return []

    print(f"成功找到 {len(links)} 个数据集链接。")
    return links

def download_and_extract(url, dest_folder):
    """
    下载单个文件并根据其类型进行解压。
    """
    try:
        filename = url.split('/')[-1]
        filepath = os.path.join(dest_folder, filename)

        print(f"  -> 正在下载 {filename}...")
        
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print(f"  -> 下载完成。文件保存在: {filepath}")

        # --- 解压文件 ---
        print(f"  -> 正在解压 {filename}...")
        
        if filename.endswith(".jsonl.gz"):
            # .gz文件不是归档文件，只包含一个文件。直接解压到目标目录。
            # 输出文件名： All_Beauty.jsonl.gz -> All_Beauty.jsonl
            output_filepath = os.path.splitext(filepath)[0] 
            with gzip.open(filepath, 'rb') as f_in:
                with open(output_filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"  -> 解压完成，文件在: {output_filepath}")

        elif filename.endswith(".zip"):
            extract_path = os.path.join(dest_folder, os.path.splitext(filename)[0])
            os.makedirs(extract_path, exist_ok=True)
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"  -> 解压完成，文件在: {extract_path}")
            
        elif filename.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tbz")):
            extract_path = os.path.join(dest_folder, os.path.splitext(filename)[0])
            os.makedirs(extract_path, exist_ok=True)
            with tarfile.open(filepath, 'r:*') as tar_ref:
                tar_ref.extractall(extract_path)
            print(f"  -> 解压完成，文件在: {extract_path}")
        else:
            print(f"  -> 未知或不支持的压缩格式: {filename}")
            return

        if DELETE_ARCHIVE_AFTER_EXTRACTION:
            os.remove(filepath)
            print(f"  -> 已删除压缩包: {filename}")

    except Exception as e:
        print(f"  -> 处理文件 {url} 时发生错误: {e}")

def main():

    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        print(f"已创建目录: {DOWNLOAD_DIR}")

    dataset_urls = get_dataset_links(TARGET_URL, FILE_EXTENSIONS)

    if not dataset_urls:
        print("未找到任何数据集链接，脚本退出。")
        return

    for i, url in enumerate(dataset_urls, 1):
        print(f"\n--- 处理第 {i}/{len(dataset_urls)} 个文件 ---")
        download_and_extract(url, DOWNLOAD_DIR)
        
    print("\n所有任务已完成！")

if __name__ == "__main__":

    main()