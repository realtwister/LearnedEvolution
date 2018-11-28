from ..utils.random import RandomGenerator;
from ..utils.parse_config import ParseConfig;




class ProblemSuite(RandomGenerator):
    @classmethod
    def _get_kwargs(cls, config, key = ""):
        cls._config_required(
            "dimension"
        )
        kwargs =  super()._get_kwargs(config, key=key)
        from . import problem_classes
        problem_classes = problem_classes()
        for i, cls_list in enumerate(kwargs['clss']):
            current = None
            for class_name in cls_list[::-1]:
                if current is None:
                    current = problem_classes[class_name]
                else:
                    current = problem_classes[class_name](current)
            kwargs['clss'][i] = current
        return kwargs;
