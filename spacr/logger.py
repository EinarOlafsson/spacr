import logging
import functools
import os

# Automatically configure logging
def configure_logger(log_file_name='spacr.log'):
    """
    Configure the global logging system to log INFO-level messages to a file.

    This function sets up logging to write messages to a file located in the user's
    home directory. The log format includes timestamp, module name, log level, and message.

    Args:
        log_file_name (str): Name of the log file. The file will be saved in the user's
                             home directory. Default is 'spacr.log'.

    Example:
        >>> configure_logger()
        # Logs will be saved to ~/spacr.log

        >>> configure_logger('custom.log')
        # Logs will be saved to ~/custom.log
    """
    # Determine a safe location for the log file
    home_dir = os.path.expanduser("~")  # Get the user's home directory
    log_file_path = os.path.join(home_dir, log_file_name)  # Save log file in home directory

    # Setup logging configuration
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Call the function to configure the logger
configure_logger()

# Create a logger instance for this module
logger = logging.getLogger(__name__)

# Decorator to log function calls
def log_function_call(func):
    """
    Decorator that logs the function call, its arguments, return value, and any exceptions.

    Logs:
        - Function name and arguments on call
        - Return value on successful completion
        - Exception traceback if an error is raised

    Args:
        func (Callable): The target function to decorate.

    Returns:
        Callable: The wrapped function with logging enabled.

    Example:
        >>> @log_function_call
        ... def multiply(a, b):
        ...     return a * b
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]  # Arguments passed to the function
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # Keyword arguments
        signature = ", ".join(args_repr + kwargs_repr)  # Construct the signature for logging
        logger.info(f"Calling {func.__name__}({signature})")  # Log function call
        try:
            result = func(*args, **kwargs)  # Execute the function
            logger.info(f"{func.__name__} returned {result!r}")  # Log the result
            return result
        except Exception as e:
            logger.exception(f"Exception occurred in {func.__name__}")  # Log any exceptions
            raise  # Re-raise the exception after logging it
    return wrapper