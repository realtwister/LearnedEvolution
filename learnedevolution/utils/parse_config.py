class ParseConfig(object):
    # Parses configuration
    config_required = set()
    config_required_from_obj = ['type']
    config_defaults = {};

    @classmethod
    def from_config(cls, config, key = "", new = False):
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
            else:
                # Find in provided config
                value = _get_in_config(config, param, key);

            if value is None:
                # Param not found
                assert param in cls.config_defaults, "\"{}\" should be provided for {} at \"{}\"".format(param, cls.__name__, key);
                # add param to config
                obj_config[param] = cls.config_defaults[param];
                value = cls.config_defaults[param];
            kwargs[param]  = value;
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



    @staticmethod
    def load_config(path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        assert hasattr(config, "config")
        return config.config

# helpers
def _get_in_config(config, param, key):
    assert isinstance(key, str);
    assert isinstance(param, str);
    if key != "":
        # Calc next key
        idx = key.find('.');
        new_key = key[idx+1:] if idx > -1 else "";
        key = key[:idx] if idx > -1 else key;

        # See if we can find the param closer to the key
        new_config = config if key == "" else config[key]
        current = _get_in_config(new_config, param, new_key);
        if current is not None:
            return current;

    # Return own result
    return config[param] if param in config else None;

def _get_own_config(config, key):
    current = config;
    for item in key.split("."):
        if item  == "":
            continue;
        if item not in config:
            current[item] = dict();
        current = current[item];
    return current;

def config_factory(classes, config, key = ""):
    obj_config = _get_own_config(config,key);
    assert 'type' in obj_config, '"type" should be specified in {}'.format(key)
    type = obj_config['type'];
    classes = classes();
    assert type in classes, '"{}" does not exist for {}'.format(type, key)

    return classes[type].from_config(config, key)
