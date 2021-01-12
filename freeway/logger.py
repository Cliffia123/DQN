#!/usr/bin/env python
# -*- coding: utf-8 -*

class Logger:
    """
    创建日志文件，将训练和测试的数据写入日志文件中，方便后期画图
    """
    def __init__(self, filename):
        self.filename = filename

        f = open(f"{self.filename}.csv", "w")
        f.close()

    def log(self, msg):
        f = open(f"{self.filename}.csv", "a+")
        f.write(f"{msg}\n")
        f.close()