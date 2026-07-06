import logging


def create_logger(logfile):

    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s : %(message)s",
        filemode="w",
    )

    return logging.getLogger()
