from pandas import read_csv, read_excel
import os

# read a file depending on his extension
async def read_file(filename):
    if filename.endswith(".csv"):
        data = read_csv(filename)
    elif filename.endswith((".xls", ".xlsx")):
        data = read_excel(filename)
    else:
        raise ValueError("Format de fichier non pris en charge.")

    return data

# calculate number of elements and duplicates
def calculate_duplicates(data, columns):
    duplicates = data.duplicated(subset=columns, keep=False).sum()
    total_elements = len(data)
    unique_elements = total_elements - duplicates
    return total_elements, duplicates, unique_elements

# add a "is_seed" col to the dataset
def add_is_seed_column(data):
    data["is_seed"] = 1
    return data

# verify "title" and "abstract" cols in the file
def check_columns(data,object=""):
    required_columns = ["title", "abstract"]
    if not all(column in data.columns for column in required_columns):
        raise ValueError(f"Les colonnes 'title' et 'abstract' doivent être présentes dans le fichier des ."+{object})

# save an uploaded file into the uploads file and return the file_path
async def save_uploaded_file(uploaded_file):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

    uploads_dir = os.path.join(parent_dir, "uploads")

    os.makedirs(uploads_dir, exist_ok=True)

    file_path = os.path.join(uploads_dir, uploaded_file.filename.lower())
    with open(file_path, 'wb') as f:
        content = await uploaded_file.read()
        f.write(content)
    
    return file_path
