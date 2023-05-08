from pathlib import Path

def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    path = Path(path)
    files = list(path.glob("*"))
    return files[0]  