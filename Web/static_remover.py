import os

def clear_folder(folder_path):
    """
    Clears all video files from the specified folder.

    Args:
        folder_path (str): The path to the folder containing video files.
    """
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) : 
                os.unlink(file_path)
    else:
        print(f"The folder {folder_path} does not exist.")