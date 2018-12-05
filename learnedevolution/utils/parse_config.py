import os
class ParseConfig(object):
    # Parses configuration
    config_required = set()
    config_required_from_obj = ['type']
    config_defaults = {};
    default_topic = ""

    @classmethod
    def from_config(cls, config, key = None, new = False):
        if key is None:
            key = cls.default_topic
        obj_config = _get_own_config(config, key)
        if not new and '[OBJ]' in obj_config:
            return obj_config['[OBJ]'];
        kwargs = cls._get_kwargs(config, key)
        obj = cls(**kwargs)
        obj_config['[OBJ]'] = obj
        return obj

    @classmethod
    def _get_kwargs(cls, config, key=""):
        obj_config = _get_own_config(config, key);
        kwargs = dict();
        for param in cls.config_required:
            if param in cls.config_required_from_obj:
                value = obj_config[param] if param in obj_config else None;
                if isinstance(value, dict):
                    value['key'] = key+"."+param
            else:
                # Find in provided config
                value = _get_in_config(config, param, key);

            if value is None:
                # Param not found
                assert param in cls.config_defaults, "\"{}\" should be provided for {} at \"{}\"".format(param, cls.__name__, key);
                # add param to config
                obj_config[param] = cls.config_defaults[param];
                value = cls.config_defaults[param];
                if isinstance(value, dict):
                    value['key'] = key+"."+param
            kwargs[param]  = value;
        obj_config.update(kwargs);
        return kwargs

    @classmethod
    def _config_required(cls, *new):
        cls.config_required = cls.config_required | set(new);

    @classmethod
    def _config_required_from_obj(cls, *new):
        cls.config_required_from_obj = cls.config_required_from_obj | set(new);

    @classmethod
    def _config_defaults(cls, **new):
        cls.config_defaults = dict(cls.config_defaults);
        cls.config_defaults.update(new);

    @classmethod
    def from_config_file(cls, config_file, replace=None, overwrite_config_file= None):
        assert os.path.exists(config_file)
        if replace is None:
            replace = dict()
        config = cls.load_config(config_file)
        if overwrite_config_file is not None:
            assert os.path.exists(overwrite_config_file)
            overwrite_config = cls.load_config(overwrite_config_file, "overwrite_config")
            cls.config_merge(config, overwrite_config)
        cls.config_merge(config, replace)
        obj = cls.from_config(config)
        return obj

    @staticmethod
    def config_merge(base_config, overwrite_config):
        # overwrites the entries in base_config with the entries in overwrite_config
        b = base_config
        o = overwrite_config
        for k in o:
            if k in base_config and isinstance(o[k], dict) and isinstance(b[k], dict):
                ParseConfig.config_merge(b[k], o[k])
            else:
                b[k] = o[k]





    @staticmethod
    def load_config(path, module_name = "config"):
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        assert hasattr(config, "config")
        assert isinstance(config.config, dict), "Configuration should be a dictionary"
        return config.config

# helpers
def _get_in_config(config, param, key, current_key = ""):
    assert isinstance(key, str);
    assert isinstance(param, str);
    if key != "":
        # Calc next key
        idx = key.find('.');
        new_key = key[idx+1:] if idx > -1 else "";
        key = key[:idx] if idx > -1 else key;

        # See if we can find the param closer to the key
        if isinstance(config, list):
            new_config = config if key == "" else config[int(key)]
        else:
            new_config = config if key == "" else config[key]
        current = _get_in_config(new_config, param, new_key, current_key+"."+key);
        if current is not None:
            return current;

    # Return own result
    current  = config[param] if param in config else None;
    if isinstance(current, dict):
        current['key'] = current_key+"."+param;
    return current;

def _get_own_config(config, key):
    current = config;
    for item in key.split("."):
        if item  == "":
            continue;
        if isinstance(current, list):
            item = int(item)
            if item >= len(current):
                current[item] = dict()
        elif item not in current:
            current[item] = dict()
        current = current[item];
    return current;

def config_factory(classes, config, key = ""):
    obj_config = _get_own_config(config,key);
    if "type" not in obj_config:
        print(obj_config)
    assert 'type' in obj_config, '"type" should be specified in {}'.format(key)
    type = obj_config['type'];
    classes = classes();
    assert type in classes, '"{}" does not exist for {}'.format(type, key)

    return classes[type].from_config(config, key)
