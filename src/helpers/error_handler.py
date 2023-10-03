"""
Custom errors handlers for the app
Path: helpers/error_handler.py
"""
import traceback


class CustomException(Exception):
    """
    Custom exception
    """
    def __init__(self, message, logger):
        self.message = message
        super().__init__(self.message)
        _, line_number, function_name, _ = traceback.extract_stack()[-2]
        stack = traceback.extract_stack()
        logger.error(f"Error in {function_name} at line {line_number}: {message}, {stack}")
