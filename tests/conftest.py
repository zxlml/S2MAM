import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Windows 下避免 OpenMP 重复加载
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
