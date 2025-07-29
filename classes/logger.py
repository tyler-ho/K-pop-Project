import logging

def create_logger(log_location):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_location),
            logging.StreamHandler()
        ]
    )
