"""Versioned, truth-independent source-role policies for protocol reproduction."""
from pathlib import Path
import numpy as np

POLICIES = {
    'production_v1': ('production', 'production'),
    # HC production baseline: validated study mapping policy with the normal
    # production residual/member policy.  Unlike the one-fit HC diagnostic,
    # this retrains the study-policy map for each run and preserves the normal
    # per-fit production ensemble behavior.
    'production_v2': ('study', 'production'),
    'study_replay_v1': ('study', 'study'),
    'mapping_hybrid_v1': ('production', 'study'),
    'residual_hybrid_v1': ('study', 'production'),
}

DEFAULT_POLICY = 'production_v2'
PRODUCTION_POLICIES = frozenset(('production_v1', 'production_v2'))
DIAGNOSTIC_POLICIES = frozenset(set(POLICIES) - set(PRODUCTION_POLICIES))
REPLAY_SCORING_POLICIES = frozenset(('study_replay_v1', 'mapping_hybrid_v1', 'residual_hybrid_v1'))


def policy_parts(policy):
    if policy not in POLICIES:
        raise ValueError('Unknown data-role policy: '+str(policy))
    return POLICIES[policy]


def validate_replay_batch(policy, batch_size):
    if policy_parts(policy)[1] == 'study' and batch_size != 256:
        raise ValueError('Historical residual-role replay requires batch_size=256')


def override_roles(sources, roles, seed, policy, batch_size):
    """Preserve production_v1 exactly; production_v2 promotes the HC study-map/production-residual combination."""
    mapping, residual = policy_parts(policy)
    validate_replay_batch(policy, batch_size)
    roles = {k: dict(source=v['source'], indices=v['indices'].copy()) for k,v in roles.items()}
    def put(name, source, indices):
        roles[name] = dict(source=source, indices=np.asarray(indices, dtype=np.int64))
    if mapping == 'study':
        order = np.random.default_rng([seed,501]).permutation(len(sources['outerdata_train']))
        n = int(.7*len(order))
        put('map_train','outerdata_train',order[:n])
        put('correction_train','outerdata_train',order[n:])
        val = np.random.default_rng([seed,502]).permutation(len(sources['outerdata_val']))
        n = len(val)
        put('map_val','outerdata_val',val[:n//2])
        put('correction_val','outerdata_val',val[n//2:3*n//4])
        put('closure','outerdata_val',val[3*n//4:])
        roles.pop('calibration_train'); roles.pop('calibration_val')
    if residual == 'study':
        from .campaign import member_split
        # The historical study used this source split before mapping rejection.
        a,b,_ = member_split(len(sources['innerdata_train']),seed,0,batch_size=256)
        if len(a)%512 == 1:
            b,a = np.r_[a[-1],b],a[:-1]
        put('signal_train','innerdata_train',a)
        put('signal_val','innerdata_train',b)
        roles.pop('residual_train')
    if policy == 'study_replay_v1':
        put('evidence','innerdata_val',np.arange(len(sources['innerdata_val'])))
        roles.pop('mixture_validation')
    return roles


def attach_source_ids(data, roles, mapped):
    path = Path(data)/'event_ids.npz'
    if not path.exists():
        return  # Legacy/synthetic callers may not provide physical event identities.
    with np.load(path,allow_pickle=False) as ids:
        for name,role in roles.items():
            item = mapped[name]
            full = ids[role['source']+'.npy'][role['indices']]
            item['source_ids'] = full
            item['ids'] = full[item['mask']]
            item['source_indices'] = role['indices']
            item['source'] = role['source']


def finalize_mapped(mapped, seed, policy, batch_size):
    """Mirror study post-mapping adjustments without silently repartitioning."""
    mapping,residual = policy_parts(policy)
    validate_replay_batch(policy, batch_size)
    if mapping == 'study':
        for source,target,offset in [('correction_train','calibration_train',9101),
                                     ('correction_val','calibration_val',9102)]:
            item=mapped[source]
            ix=np.random.default_rng([seed,offset]).permutation(len(item['z']))
            ix=ix[len(ix)//2:]
            accepted=np.flatnonzero(item['mask'])[ix]
            mapped[target] = dict(z=item['z'][ix], mass=item['mass'][ix], rows=item['rows'][accepted],
                                 mask=np.ones(len(ix),bool))
            for key in ('ids',):
                if key in item: mapped[target][key]=item[key][ix]
            if 'ids' in item: mapped[target]['source_ids']=mapped[target]['ids']
            if 'source' in item:
                mapped[target].update(source=item['source'],source_indices=item['source_indices'][accepted])
    if residual == 'study':
        train,val = mapped['signal_train'],mapped['signal_val']
        # Historical robustness shared-input preparation moves a final singleton
        # to the END of validation, after mapping acceptance (distinct from source split).
        if len(train['z'])%batch_size == 1:
            # Move the corresponding source row as well as its accepted latent.
            # Rejected rows keep their original order and remain auditable.
            if 'mask' in train:
                source_index=np.flatnonzero(train['mask'])[-1]
                for key in ('rows','mask','source_ids','source_indices'):
                    if key in train:
                        val[key]=np.concatenate((val[key],train[key][source_index:source_index+1]))
                        train[key]=np.delete(train[key],source_index,axis=0)
            for key in ('z','mass','ids'):
                if key in train:
                    val[key]=np.concatenate((val[key],train[key][-1:]))
                    train[key]=train[key][:-1]
        joined={k:np.concatenate((train[k],val[k])) for k in ('z','mass')}
        if 'ids' in train: joined['ids']=np.concatenate((train['ids'],val['ids']))
        mapped['residual_train']=joined
        mapped['member_splits']={0:dict(train=np.arange(len(train['z'])),
            validation=np.arange(len(train['z']),len(joined['z'])))}
        if policy == 'study_replay_v1': mapped['mixture_validation']=val
    return mapped


def validate_member_split(split, n):
    values=[]
    for key in ('train','validation'):
        ix=np.asarray(split[key])
        if ix.ndim!=1 or ix.dtype.kind not in 'iu' or len(ix)<2 or (ix<0).any() or (ix>=n).any():
            raise ValueError('Invalid explicit member '+key+' indices')
        values.append(ix.astype(np.int64))
    if not np.array_equal(np.sort(np.concatenate(values)),np.arange(n)):
        raise ValueError('Member split must partition every accepted development event exactly once')
    return values
