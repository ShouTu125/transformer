import json


class Config(object):
    def __init__(self, logger, args):
        self.logger = logger
        self.config = vars(args)
        
    def save_config(self, path):
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
        self.logger.debug(f'Config saved to file {path}')

    def load_config(self, path):
        with open(path) as f:
            self.config = json.load(f)

        self.logger.debug(f'Config loaded from file {path}')

    def print_config(self):
        debug = 'Running with the following configs:\n'
        for k,v in self.config.items():
            debug += f'\t{k} : {str(v)}\n'

        self.logger.debug('\n' + debug + '\n')
        