import logging
import posixpath
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
import os


def get_logger(general_config):
	logging_path = general_config['logging_path']
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	rootLogger = logging.getLogger('Text_Affinity')


	if not os.path.exists(logging_path):
		print (logging_path)
		os.mkdir(logging_path)
		print("Directory /logs created")

	fileHandler = TimedRotatingFileHandler(logging_path + "/" + "{:%Y-%m-%d}.log".format(datetime.now()),when="D",
	                                       interval=1,backupCount=5)

	fileHandler.setFormatter(formatter)

	rootLogger.addHandler(fileHandler)

	consoleHandler = logging.StreamHandler()
	consoleHandler.setFormatter(formatter)
	rootLogger.addHandler(consoleHandler)

	if general_config['logging_level'] == "INFO":
		rootLogger.setLevel(logging.INFO)
	elif general_config['logging_level'] == "WARNING":
		rootLogger.setLevel(logging.WARNING)
	else:
		rootLogger.setLevel(logging.ERROR)

	return rootLogger
