"""Opt-in CPU mapping interventions; ordinary production never enters this module.

Mapping optimization remains in mapping.prepare. Residual fitting, safeguards,
calibration and exports remain in the actual production worker/pipeline.
"""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .storage import file_digest, write_json, atomic_write, save_npz, save_array, digest
from .enhancements import load_torch, save_torch, chunks


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, expected):
    if file_digest(Path(path)) != expected:
        raise ValueError('Mapping experiment artifact changed: '+str(path))


def ledger(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def environment(path):
    from scripts.riddle_protocol_environment import snapshot
    value = snapshot()


    stable = {k:v for k,v in value.items() if k != 'hostname'}
    path = Path(path)
    if path.exists() and read(path)['numerical'] != stable:
        raise ValueError('Mapping experiment numerical environment changed')
    if not path.exists():
        write_json(path, dict(numerical=stable, first_process=value))
    return stable


def mapping_inputs(spec, roles, sources, device):
    if str(device) != 'cpu':
        raise ValueError('Mapping-component study is CPU-only')
    checked(spec['preprocessing'], spec['preprocessing_sha256'])
    checked(spec['partitions'], spec['partitions_sha256'])
    indices = ledger(spec['partitions'])
    roles = {k:dict(v) for k,v in roles.items()}
    for name, source in [('map_train','outerdata_train'), ('map_val','outerdata_val')]:
        ix = indices[name]
        if ix.dtype.kind not in 'iu' or len(np.unique(ix)) != len(ix) or (ix < 0).any() or (ix >= len(sources[source])).any():
            raise ValueError('Invalid experimental map role '+name)
        roles[name] = dict(source=source, indices=ix)
    return roles, load_torch(spec['preprocessing'])


def observe_epoch(spec, output, model, epoch, clean, sources, fit, mass_parameters, device, primary_loss):
    """Record a second validation loss on the same trained checkpoint, no RNG use."""
    from .preprocessing import load_dataset
    from scripts.riddle_protocol_cohort import assert_eval, buffers, assert_buffers_unchanged
    checked(spec['partitions'], spec['partitions_sha256'])
    ix = ledger(spec['partitions'])['alternate_map_val']
    rows = sources['outerdata_val'][ix].copy(); rows[:, -1] = 0
    prepared = load_dataset(rows, external_datadict=fit)
    assert_eval(model); before = buffers(model)
    mm, ms = mass_parameters
    with torch.random.fork_rng(devices=[]):
        loss = -float(chunks(model.log_probs, prepared['tensor2'],
            (prepared['labels']-mm)/ms, device=device).double().mean())
    assert_buffers_unchanged(model, before)
    if not np.isfinite(loss):
        raise ValueError('Nonfinite alternate mapping-validation loss')
    save_torch(Path(output)/'epoch_checkpoints'/f'epoch_{epoch:03d}.pt',
        dict(epoch=epoch, model=model.state_dict(), validation_nll=primary_loss,
             alternate_validation_nll=loss, alternate_events=len(prepared['tensor2'])))


def validate_worker(args):
    from .options import effective_features
    if (args.method != 'riddle' or args.device != 'cpu' or args.workers != 1 or args.runs != 1
            or not all(effective_features(args.settings['riddle']).values())
            or args.settings['riddle'].get('data_policy','production_v1') != 'production_v1'):
        raise ValueError('Mapping experiment requires the complete CPU RIDDLE baseline, one fit/worker')
    spec = args.mapping_experiment
    checked(spec['roles'], spec['roles_sha256'])
    checked(Path(spec['bank'])/'bank.json', spec['bank_sha256'])
    checked(spec['manifest'], spec['manifest_sha256'])
    manifest = read(spec['manifest'])
    if args.seed != manifest['seed'] or str(args.data) != manifest['data']:
        raise ValueError('Mapping experiment data/seed changed')
    from .data import diagnostic_profile
    inputs = read(args.data/'inputs.json')
    if diagnostic_profile(inputs) is None and not inputs.get('synthetic_smoke_fixture'):
        raise ValueError('Mapping experiments require a declared diagnostic dataset')


def prepare_frozen(data, output, seed, device, *, background, options, data_policy,
                   residual_batch_size, experiment):
    """Map persisted source roles with a verified model, without fitting a map."""
    from .mapping import Mapper, split_roles
    from .roles import attach_source_ids, validate_member_split
    spec = experiment
    output, bank = Path(output), Path(spec['bank'])
    checked(bank/'bank.json', spec['bank_sha256'])
    checked(spec['roles'], spec['roles_sha256'])
    receipt = read(bank/'bank.json')
    for name,h in receipt['files'].items(): checked(bank/name,h)
    output.mkdir(parents=True, exist_ok=True)
    state = output/'experiment.json'
    if state.exists() and read(state) != spec:
        raise ValueError('Frozen-map experimental contract changed')
    write_json(state, spec)
    for name in ('model.pt','preprocessing.pt','mapping_settings.json','flow_selection.json','history.json','background_settings.json'):
        target = output/name
        if target.exists(): checked(target, receipt['files'][name])
        else: shutil.copy2(bank/name, target)
    sources, _ = split_roles(data, seed, calibration=True)
    parts = ledger(spec['roles'])
    map_parts = ledger(bank/'source_partitions.npz')
    parts.update(map_train=map_parts['map_train'], map_val=map_parts['map_val'])
    source_names = dict(map_train='outerdata_train', map_val='outerdata_val',
        signal_train='innerdata_train', signal_val='innerdata_train',
        calibration_train='outerdata_train', calibration_val='outerdata_val', closure='outerdata_val',
        mixture_validation='innerdata_val', evidence='innerdata_val')
    roles = {k:dict(source=source_names[k], indices=v) for k,v in parts.items() if k in source_names}
    for k,v in roles.items():
        ix = v['indices']; n = len(sources[v['source']])
        if ix.dtype.kind not in 'iu' or len(ix)<2 or len(np.unique(ix))!=len(ix) or (ix<0).any() or (ix>=n).any():
            raise ValueError('Invalid fixed source assignment: '+k)
    if not np.array_equal(np.sort(np.r_[parts['signal_train'],parts['signal_val']]),np.arange(len(sources['innerdata_train']))):
        raise ValueError('Fixed residual roles must cover the source exactly once')
    mapper = Mapper(output, device); mapped = {}
    for name, role in roles.items():
        rows = sources[role['source']][role['indices']]
        z, mask = mapper.map(rows)
        mapped[name] = dict(z=z, mass=rows[mask,0], rows=rows, mask=mask)
    attach_source_ids(data, roles, mapped)
    train, val = mapped['signal_train'], mapped['signal_val']
    if len(train['z']) % residual_batch_size == 1:
        raise ValueError('Fixed residual roles produce a singleton batch; refusing silent reassignment')
    joined = {k:np.concatenate((train[k],val[k])) for k in ('z','mass','ids','source_ids','source_indices','rows','mask')}
    joined['source'] = 'innerdata_train'
    mapped['residual_train'] = joined
    split = dict(train=np.arange(len(train['z'])), validation=np.arange(len(train['z']),len(joined['z'])))
    validate_member_split(split,len(joined['z'])); mapped['member_splits'] = {0:split}
    roles['residual_train'] = dict(source='innerdata_train',indices=np.r_[parts['signal_train'],parts['signal_val']])
    atomic_write(output/'source_partitions.npz',lambda p:save_npz(p,**{k:v['indices'] for k,v in roles.items()}))
    write_json(output/'data_roles.json',{k:dict(source=v['source']+'.npy',events=len(v['indices']),
        indices_sha256=digest(v['indices']),truth_labels_used=False) for k,v in roles.items()})
    ids = {name+'__'+key:item[key] for name,item in mapped.items() if name!='member_splits'
           for key in ('ids','source_ids','source_indices','mask') if key in item}
    atomic_write(output/'event_roles.npz',lambda p:save_npz(p,**ids))
    write_json(output/'role_policy.json',dict(policy='mapping_component_diagnostic_v1',
        residual_assignment=spec['residual_assignment'], split_before_mapping=True, refill_after_rejection=False,
        truth_labels_used=False, original_production_v1_unchanged=True))
    for name, role in [('training_latents.npy','residual_train'),('validation_latents.npy','evidence'),
                       ('mixture_validation_latents.npy','mixture_validation')]:
        item = mapped[role]
        value = np.column_stack((item['mass'],item['z'],np.ones(len(item['z'])),np.zeros(len(item['z'])))).astype(np.float32)
        atomic_write(output/name,lambda p,v=value:save_array(p,v))
    return read(output/'flow_selection.json'), mapper, mapped
