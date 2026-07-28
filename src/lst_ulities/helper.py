import glob


def glob_files(path: str | list[str], pattern: str) -> list[str]:
    """
    Glob files in the given path using the given pattern.
    """
    if isinstance(path, str):
        files = glob.glob(f"{path}/{pattern}")
        return sorted(files)
    elif isinstance(path, list):
        files = []
        for p in path:
            files.extend(glob.glob(f"{p}/{pattern}"))
        return sorted(files)
