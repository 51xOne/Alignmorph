"""Select pairs without changing the ordered VAE sampling stream."""


def select_pairs(pairs, ids=None):
    if ids is None:
        return pairs
    unknown = set(ids) - {pair["exp_id"] for pair in pairs}
    if unknown:
        raise ValueError(f"Unknown pair IDs: {sorted(unknown)}")
    return [pair for pair in pairs if pair["exp_id"] in ids]


def generate_sequence(pipeline, pairs, selected, generate):
    pending = {pair["exp_id"] for pair in selected}
    for pair in pairs:
        if not pending:
            break
        if pair["exp_id"] in pending:
            generate(pair)
            pending.remove(pair["exp_id"])
        else:
            pipeline.encode(pair, None)
