import os
import requests
def download_file(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"Downloaded: {save_path}")
        else:
            print(f"Failed to download: {url}")
    else:
        print(f"File already exists: {save_path}")

from bs4 import BeautifulSoup

def extract_hrefs_from_html(html_file):
    """
    Extracts all href attributes from <a> tags in an HTML file.
    
    Args:
        html_file (str): Path to the HTML file.
        output_file (str, optional): Path to save the extracted hrefs. If None, no file is saved.
    
    Returns:
        list: A list of href strings extracted from the HTML file.
    """
    # Load the HTML file
    with open(html_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')

    # Extract all href attributes from <a> tags
    hrefs = [a['href'] for a in soup.find_all('a', href=True)]

    return hrefs
    
dataset = {
    ("train","sat"):'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html',
    ("train","map"):'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html',
    ("valid","sat"):'https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/index.html',
    ("valid","map"):'https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/index.html',
    ("test", "sat"):'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat/index.html',
    ("test", "map"):'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/map/index.html',
  
} 
BASE = "dataset"
for folders, url in dataset.items():
  os.makedirs(BASE, exist_ok=True)
  f1 = os.path.join(BASE, folders[0]) 
  f2 = os.path.join(f1, folders[1])
  os.makedirs(f1, exist_ok=True) 
  os.makedirs(f2, exist_ok=True)
  index = os.path.join(f2, "index.html")
  download_file(url, index)
  hrefs = extract_hrefs_from_html(index)
  for href in hrefs:
    download_file(href, os.path.join(f2, href.split('/')[-1]))
