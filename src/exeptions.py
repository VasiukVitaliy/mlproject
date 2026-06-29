import sys
from src.logger import logging

class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        super().__init__(str(error))
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        self.error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            file_name, exc_tb.tb_lineno, str(error)
        )

    def __str__(self):
        return self.error_message
    
    
if __name__=="__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.error("error")
        raise CustomException(e, sys)