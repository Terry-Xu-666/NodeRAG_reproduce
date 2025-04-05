import importlib
import sys

class LazyImport:
    def __init__(self, module_name, class_name=None):
        self.module_name = module_name
        self.class_name = class_name
        self._module = None
        self._class = None

    def _import(self):
        if self._module is None:
            if self.module_name in sys.modules:
                self._module = sys.modules[self.module_name]
            else:
                self._module = importlib.import_module(self.module_name)
            
        if self.class_name and self._class is None:
            self._class = getattr(self._module, self.class_name)
        return self._class if self.class_name else self._module

    def __call__(self, *args, **kwargs):
        cls = self._import()
        return cls(*args, **kwargs)

    def __getattr__(self, item):
        cls = self._import()
        return getattr(cls, item)