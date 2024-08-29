import os
import zipfile

path = r""


def extract_files(directory):
    files = os.listdir(directory)

    for file in files:
        file_path = os.path.join(directory, file)
        # most files are zipped in that case it must be extracted
        if file.endswith('.zip'):
            folder_name = os.path.splitext(file)[0]
            extract_path = os.path.join('raw', folder_name)
            # create new directory to keep the data structure
            os.makedirs(extract_path, exist_ok=True)

            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)


extract_files(path)
