DECOMPOSITION_TYPES = (
    'stiefel_projected_lambdas',
    'stiefel_projected_directions_only',
    'vv_bounded_volumes',
    'unprojected_lambdas',
    'diagonal_only',
)


ABLATION_PRESETS = {
    'reference_stiefel_projected_lambdas': {
        'decomposition_type': 'stiefel_projected_lambdas',
        'description': 'Reference model: Stiefel-projected directions with learned lambdas.',
    },
    'stiefel_projected_directions_only': {
        'decomposition_type': 'stiefel_projected_directions_only',
        'description': 'Ablates learned lambdas while keeping Stiefel-projected directions.',
    },
    'vv_bounded_volumes': {
        'decomposition_type': 'vv_bounded_volumes',
        'description': 'Uses only VV decomposition, represented as normalized directions with bounded column volumes.',
        'min_variance': 1e-5,
        'max_variance': 10.0,
    },
    'unprojected_lambdas': {
        'decomposition_type': 'unprojected_lambdas',
        'description': 'Ablates the Stiefel projection while keeping learned lambdas.',
    },
    'diagonal_only': {
        'decomposition_type': 'diagonal_only',
        'description': 'Removes low-rank decomposition and keeps only bounded diagonal covariance.',
        'min_variance': 1e-5,
        'max_variance': 10.0,
    },
}


def apply_ablation_preset(args):
    preset_name = getattr(args, 'ablation_preset', 'none')
    if preset_name == 'none':
        return args

    preset = ABLATION_PRESETS[preset_name]
    for key, value in preset.items():
        if key != 'description':
            setattr(args, key, value)

    if getattr(args, 'ablation_name', '') == '':
        args.ablation_name = preset_name

    return args


def format_ablation_presets():
    lines = []
    for name, preset in ABLATION_PRESETS.items():
        lines.append(f"{name}: {preset['description']}")
    return "\n".join(lines)
