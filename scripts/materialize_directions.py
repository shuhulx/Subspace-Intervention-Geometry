"""Save the recorded pilot direction values as a new CPU PyTorch file.

The values come from the archived JSON. The historical tensor hash refers
to the earlier pilot file; this command creates a new serialization.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/recovered_direction_vectors.json'
SOURCE_SHA256 = 'ab04c16c30c61c5bbfb2c1f2560d2998cad10d082149081aeceec5cde8cf03fb'
SOURCE_GIT_BLOB_SHA1 = 'e618abf662a78dae629516069f3e55397b55b63d'
HISTORICAL_TENSOR_SHA256 = 'a3878eb7072a77fe62f7fd16a6be6027a30038daa20254f7be9fbd699dfe0c1d'
MODEL_REVISION = '299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8'


def validate_source(payload):
    """Validate the numerical values and the metadata used by read_directions."""
    if (payload.get('provenance') != 'regenerated' or
            payload.get('layers') != [10, 15] or
            payload.get('model_revision') != MODEL_REVISION or
            payload.get('dtype') != 'bfloat16' or
            payload.get('tensor_dtypes') != {'s1': 'torch.float32', 's2': 'torch.float32'} or
            payload.get('tensor_file_sha256') != HISTORICAL_TENSOR_SHA256):
        raise ValueError('Recovered direction metadata does not match the recorded pilot.')
    cosine = payload.get('s1_cosine_to_released_l10')
    if not isinstance(cosine, (int, float)) or not math.isfinite(cosine) or not .98 < cosine <= 1:
        raise ValueError('Missing or invalid recorded S1 alignment.')
    for name in ('s1', 's2'):
        values = payload.get(name)
        if not isinstance(values, list) or len(values) != 2304:
            raise ValueError(f'{name} must contain 2304 values.')
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} contains a non-finite or non-numeric value.')
            try:
                recovered = struct.unpack('!f', struct.pack('!f', value))[0]
            except (OverflowError, struct.error) as exc:
                raise ValueError(f'{name} contains a value outside float32 range.') from exc
            if recovered != value:
                raise ValueError(f'{name} contains a value that is not exact in float32.')
        if not any(value != 0 for value in values):
            raise ValueError(f'{name} must be nonzero.')


def load_source(path=SOURCE):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('Recovered direction JSON does not match its pinned SHA256.')
    blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
    if blob != SOURCE_GIT_BLOB_SHA1:
        raise ValueError('Recovered direction JSON does not match its source Git blob.')
    payload = json.loads(data)
    validate_source(payload)
    return payload


def materialize(output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite {output}')
    source = load_source()
    try:
        torch = importlib.import_module('torch')
    except ImportError as exc:
        raise RuntimeError('PyTorch is required to write the tensor file; no model download is needed.') from exc

    payload = {key: value for key, value in source.items()
               if key not in ('s1', 's2', 'tensor_file_sha256')}
    for name in ('s1', 's2'):
        payload[name] = torch.tensor(source[name], dtype=torch.float32, device='cpu')
    payload.update({
        'historical_tensor_file_sha256': HISTORICAL_TENSOR_SHA256,
        'recovered_from_json_sha256': SOURCE_SHA256,
        'recovered_from_json_git_blob_sha1': SOURCE_GIT_BLOB_SHA1,
        'materialized_with_torch': str(torch.__version__),
        'serialization_note': 'Recorded pilot float32 values saved in a new .pt file. '
                              'The historical tensor hash refers to the earlier pilot file.',
    })

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.tmp', dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # The hard link fails if another process already created the output.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        'output': str(output),
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'source_json_sha256': SOURCE_SHA256,
        'historical_tensor_file_sha256': HISTORICAL_TENSOR_SHA256,
        'tensor_dtype': 'float32',
        'recorded_extraction_dtype': source['dtype'],
        'note': 'New .pt serialization of recorded pilot values; '
                'the historical tensor hash refers to the earlier pilot file.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, metavar='OUTPUT.pt',
                        help='new tensor file; parent directories are created and overwrites are refused')
    args = parser.parse_args()
    try:
        report = materialize(args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
