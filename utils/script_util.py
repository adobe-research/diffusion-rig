import argparse
import inspect

from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.model import DiffusionRig

def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        p2_weight=False,
        p2_gamma=1.0,
        p2_k=1.0,
    )

def model_and_diffusion_defaults():

    """
    Defaults for image training.
    """
    res = dict(
        image_size=256,
        num_channels=128,
        num_res_blocks=2,
        num_heads=1,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16",
        channel_mult="",
        dropout=0.1,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
        latent_dim=64,
        in_channels=12,
        encoder_type='resnet18',
    )
    res.update(diffusion_defaults())
    return res

def create_model_and_diffusion(
    image_size,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_new_attention_order,
    latent_dim,
    in_channels,
    encoder_type,
    p2_weight,
    p2_gamma,
    p2_k,
):

    model = DiffusionRig(        
        image_size,
        learn_sigma,
        num_channels,
        num_res_blocks,
        channel_mult,
        num_heads,
        num_head_channels,
        num_heads_upsample,
        attention_resolutions,
        dropout,
        use_checkpoint,
        use_scale_shift_norm,
        resblock_updown,
        use_new_attention_order,
        latent_dim,
        in_channels,
        encoder_type,
    )

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        p2_weight=p2_weight,
        p2_gamma=p2_gamma,
        p2_k=p2_k,
    )

    return model, diffusion


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    p2_weight=False,
    p2_gamma=1.,
    p2_k=1.,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]

    if not predict_xstart:
        model_mean_type = gd.ModelMeanType.EPSILON
    else:
        model_mean_type = gd.ModelMeanType.START_X
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        p2_weight=p2_weight,
        p2_gamma=p2_gamma,
        p2_k=p2_k,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
