import argparse
import json
import logging
from logging.handlers import TimedRotatingFileHandler


def main_run(main_func, DEFAULT_CONFIG):
    parser = argparse.ArgumentParser(description='Protocol configuration:')
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG, type=str,
                        help='config file (default: ./<sript_filename>.json)')
    parser.add_argument('-ch', '--checkpoint', default=None, type=str,
                        help='Path to a model checkpoint (.pth file)')
    parser.add_argument('-rsd', '--resource_dir', default=None, type=str,
                        help='Path to a directory with resource files')
    parser.add_argument('-sv', '--save_dir', default=None, type=str,
                        help='Path to save folder: save model states and images generated during val phase')
    parser.add_argument('-chlr', '--change_lr', default=False, type=bool,
                        help='In case checkpoint is loaded. If False - continue with resumed parameters '
                             'of optim and lr_sched or else use the ones defined in train.py')
    arguments = parser.parse_args()
    config = json.load(open(arguments.config))
    assert config is not None, 'Default configuration file is not accessible. Please define custom configuration file.'
    main_func(config, arguments)


def default_log_config(file_output=None, silence=False, console_level=logging.INFO, file_level=logging.DEBUG):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = "%Y-%m-%d %H:%M:%S"
    ch, fh = logging.NullHandler(), logging.NullHandler()
    if not silence:
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
    if file_output:
        fh = logging.handlers.RotatingFileHandler(filename=file_output, maxBytes=1048576, backupCount=5)
        fh.setLevel(file_level)
    logging.basicConfig(format=log_format, datefmt=date_format, level=logging.INFO, handlers=[ch, fh])