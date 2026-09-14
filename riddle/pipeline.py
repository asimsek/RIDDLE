import numpy as np

from riddle.storage import atomic_write, save_npz, write_json
from riddle.recovery import EpochRecovery
from riddle.acceleration import install_tensor_batches, execution_report
from .latent import prepare, Mapper
from .campaign import train_campaign, ensemble_predict, fractions, PROTOCOL
from .runtime import ordered_map
from .settings import input_features
from .resume import resume_policy
from .production import evaluation_rows, region_acceptance, PRODUCTION_POLICY
from .integrity import require_finite


def run(args, contract):
    fraction_values = fractions(args.fractions)
    acceleration = install_tensor_batches()
    output = args.output
    latent_root = output / "background"
    recovery = EpochRecovery(latent_root, contract, args.resume, **resume_policy(args))
    settings = args.settings
    selection = prepare(
        args.data, latent_root, args.seed, args.device, recovery, settings=settings["background"]
    )
    training, validation = ordered_map(
        np.load,
        [latent_root / name for name in ("training_latents.npy", "validation_latents.npy")],
        args.io_workers,
    )
    result = train_campaign(
        training,
        validation,
        output / "density",
        epochs=args.epochs,
        runs=args.runs,
        seed=args.seed,
        device=args.device,
        fraction_values=fraction_values,
        initialization=settings["riddle"]["initialization"],
        workers=args.workers,
        io_workers=args.io_workers,
        settings=settings["riddle"],
    )
    mapper = Mapper(args.data, latent_root, selection["inference_mapping_epoch"], args.device)
    acceptance = {}
    for partition, suffix in (("validation", "val"), ("test", "test"), ("signal_region", None)):
        names = (
            (f"innerdata_{suffix}.npy", f"outerdata_{suffix}.npy")
            if suffix
            else ("innerdata_test.npy", "innerdata_extrabkg_test.npy", "innerdata_extrasig.npy")
        )
        rows, region = evaluation_rows(
            ordered_map(np.load, [args.data / name for name in names], args.io_workers), names
        )
        z, mask = mapper.map(rows)
        scores = ensemble_predict(output / "density", z, args.device)
        require_finite(scores, "RIDDLE ensemble scores")
        acceptance[partition] = region_acceptance(rows[:, -1], mask, region)
        aligned = np.full(len(rows), np.nan, dtype=scores.dtype)
        aligned[mask] = scores
        arrays = dict(
            mass=rows[:, 0],
            is_signal_region=region,
            labels=rows[:, -1].astype(np.int8),
            mask=mask,
            scores=aligned,
            physical=rows[:, 1:-1],
            latent=z,
        )
        atomic_write(output / f"{partition}_scores.npz", lambda p: save_npz(p, **arrays))
    write_json(output / "mapping_acceptance.json", acceptance)
    write_json(
        output / "protocol.json",
        {
            **PROTOCOL,
            "production_policy": PRODUCTION_POLICY,
            "input_features": input_features(settings, contract["inputs"]),
            "features": training.shape[1] - 3,
            "layers": settings["riddle"]["flow"]["layers"],
            "blocks": settings["riddle"]["flow"]["num_blocks"],
            "hidden_features": settings["riddle"]["flow"]["hidden_features"],
            **settings["riddle"]["training"],
            "gradient_clip": f"flow parameters only; norm {settings['riddle']['training']['gradient_clip_norm']}",
            "ensemble": f"equal-weight mean signal density over all requested fits and {settings['riddle']['training']['selected_checkpoints']} validation-selected epochs per fit",
            "settings": settings,
            "flow_selection": selection,
            "background_epochs": settings["background"]["epochs"],
            "background_configuration": {
                **settings["background"]["configuration"],
                "num_inputs": training.shape[1] - 3,
            },
            "reference_samples": settings["background"]["reference_samples"],
            "epochs": args.epochs,
            "runs": args.runs,
            "initialization": settings["riddle"]["initialization"],
            "fractions": args.fractions,
            "selected_checkpoints": result["selected_checkpoints"],
            "acceleration": execution_report(acceleration),
        },
    )
