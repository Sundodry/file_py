# -*- coding: utf-8 -*-

import wget
import zipfile
import tarfile
import os
from Zrl_uncompressfile import un_zip

# 网络地址
DATA_URL = 'http://labfile.oss.aliyuncs.com/courses/782/data.zip'
# 本地硬盘文件
#  DATA_dir = 'E:/Zrl_python/file_py/Zrl_project/NBA\Data'

out_fname = 'data.zip'

wget.download(DATA_URL, out=out_fname)

# 提取压缩包
un_zip(out_fname)

# 删除下载文件
os.remove(out_fname)
