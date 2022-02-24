import os

class SparkUtils:
    def __init__(self, spark):
        self.spark = spark
        
    def get_data_path(self):
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

    def read_csv(self, filename, header=True, inferSchema=True, data_path=None):
        data_path = self.get_data_path() if data_path == None else data_path
        return self.spark.read.csv(f"file:{data_path}/{filename}", header=header, inferSchema=inferSchema)

    def read_json(self, filename, header=True, inferSchema=True, data_path=None):
        data_path = self.get_data_path() if data_path == None else data_path
        return self.spark.read.csv(f"file:{data_path}/{filename}", header=header, inferSchema=inferSchema)

    def read_svm(self, filename, data_path=None):
        data_path = self.get_data_path() if data_path == None else data_path
        return self.spark.read.format('libsvm').load(f"file:{data_path}/{filename}")
        