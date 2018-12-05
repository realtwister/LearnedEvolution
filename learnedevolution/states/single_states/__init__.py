def single_state_classes():
    from .translation_scale_invariant import TranslationScaleInvariant
    from .translation_scale_rotation_invariant import TranslationScaleRotationInvariant
    return locals()
