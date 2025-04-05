from ruamel.yaml import YAML
import os

class YamlHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.indent(mapping=2, sequence=4, offset=2)
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                self.data = self.yaml.load(f)
        else:
            raise FileNotFoundError(f"File {self.file_path} does not exist.")

    def save(self):
        if self.data is not None:
            with open(self.file_path, 'w') as f:
                self.yaml.dump(self.data, f)

    def update_config(self, key_path, value):
        """
        Update the value of a nested key in the YAML data.
        
        :param key_path: List of keys representing the path to the target key.
        :param value: The new value to set.
        """
        data = self.data
        for key in key_path[:-1]:
            data = data.get(key, {})
        data[key_path[-1]] = value


