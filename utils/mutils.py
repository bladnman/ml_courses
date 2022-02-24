import os

def get_data_path():
    """Will search upward for first directory with a /data folder"""
    def get_dir(parts, num):
        return '/'.join(parts[:(-1*num)]) + '/data'

    cwd = os.getcwd()
    parts = cwd.split('/')
    cnt = len(parts)
    while cnt > 0:
        dir_test = f"{'/'.join(parts)}/data"
        parts.pop()
        cnt = len(parts)
        exists = os.path.isdir(dir_test)
        if exists: return dir_test
    return None

def read_csv(filename, header=True, inferSchema=True, data_path=None):
    data_path = get_data_path() if data_path == None else data_path
    return spark.read.csv(f"file:{data_path}/{filename}", header=header, inferSchema=inferSchema)
