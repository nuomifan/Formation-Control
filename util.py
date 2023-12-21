import shutil
from pathlib import Path


def del_file(path):
    # 删除路径下的所有文件
    for elm in Path(path).glob('*'):
        print("删除", elm)
        elm.unlink() if elm.is_file() else shutil.rmtree(elm)
