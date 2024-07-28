import sys
import os
import time

class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def setup_paths(program_name):
    start_time = time.time()
    # start_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_time))
    start_time_str = time.strftime("%Y%m%d_%H%M", time.localtime(start_time))
    log_file = f'/home/zhangzhan/GCN_train_2/Model/{program_name}_{start_time_str}_model_log.txt'
    model_file = f'/home/zhangzhan/GCN_train_2/Model/{program_name}_{start_time_str}_model.pth'
    logfile = open(log_file, 'w')
    tee = Tee(sys.stdout, logfile)
    sys.stdout = tee
    return log_file, model_file, logfile

def log_only(logfile, message):
    logfile.write(message + '\n')
    logfile.flush()
