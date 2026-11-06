"""Share the existing aligned reference trajectory without shrinking its batch.

The aligned operands already contain both native endpoints. Only independent
vanilla reference inference is shared; sequence attention and gates are unchanged.
"""
import torch

from .attention import ReferenceBank, invert_references


def reference_indices(query, query_text, bank, bank_text):
    """Map exact image AND conditioning bytes; fail if any input is absent."""
    for a, b in ((query, bank), (query_text, bank_text)):
        if a.dtype != b.dtype or a.shape[1:] != b.shape[1:]:
            raise ValueError('Reference operand shapes and dtypes must agree')
    if query.shape[0] != query_text.shape[0] or bank.shape[0] != bank_text.shape[0]:
        raise ValueError('Reference and conditioning batch sizes differ')

    def rows(images, text):
        images = images.detach().cpu().contiguous().view(torch.uint8)
        text = text.detach().cpu().contiguous().view(torch.uint8)
        return [(images[i].numpy().tobytes(), text[i].numpy().tobytes())
                for i in range(images.shape[0])]

    lookup = {}
    for i, key in enumerate(rows(bank, bank_text)):
        lookup.setdefault(key, i)
    keys = rows(query, query_text)
    if any(key not in lookup for key in keys):
        raise ValueError('Aligned reference bank does not contain every native image/text pair')
    return torch.tensor([lookup[key] for key in keys], device=bank.device, dtype=torch.long)


class SharedReferencePool:
    def __init__(self, states, text):
        self.bank = ReferenceBank(states, text, None)
        self.cache_key = None

    def capture(self, unet, timestep):
        key = (id(unet), int(timestep))
        if key != self.cache_key:
            self.bank.capture(unet, timestep)
            self.cache_key = key
        return self.bank.memory

    def clear(self):
        self.bank.memory.clear()
        self.cache_key = None


class SharedReferenceBank:
    def __init__(self, pool, reliability, indices=None):
        self.pool, self.reliability, self.indices = pool, reliability, indices
        self.memory = {}

    @torch.no_grad()
    def capture(self, unet, timestep):
        self.memory.clear()
        for name, (key, value) in self.pool.capture(unet, timestep).items():
            if self.indices is not None:
                key, value = key.index_select(0, self.indices), value.index_select(0, self.indices)
            self.memory[name] = (key, value)


@torch.no_grad()
def prepare_shared_references(unet, collector, texts, frames, scheduler, timesteps):
    if unet.training:
        raise ValueError('Shared reference inference requires evaluation mode')
    text = torch.cat([t.expand(frames, -1, -1) for t in texts])
    aligned = collector.aligned.latents()
    indices = reference_indices(collector.original.latents(), text, aligned, text)
    # Preserve the exact original aligned batch, including row order and size.
    states = invert_references(unet, aligned, text, scheduler, timesteps)
    pool = SharedReferencePool(states, text)
    return (SharedReferenceBank(pool, collector.original.reliability(), indices),
            SharedReferenceBank(pool, collector.aligned.reliability()))
