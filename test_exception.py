import sys

from src.exception.exception import CustomException

try:
    x = 10 / 0
except Exception as e:
    raise CustomException(e, sys)