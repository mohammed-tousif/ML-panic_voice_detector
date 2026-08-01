# Design and Security Notes

## Model pipeline

`audio_pipeline.py` is the single source of truth for audio decoding and MFCC
feature extraction. Training, CLI prediction, and web prediction import the same
function so preprocessing cannot drift between environments. The included model
was originally trained with direct three-second MFCC extraction, so the shared
pipeline intentionally preserves that behavior.

Training deduplicates all recordings by SHA-256 content. Speakers are isolated
across training, validation, and test splits. Gain, background-noise, and
time-shift augmentation is applied only after splitting and only to development
recordings, so synthetic variants cannot leak into evaluation.

Validation predictions select the lowest useful threshold that maximizes recall
while maintaining the configured precision floor. The classifier, selected
threshold, and feature contract are stored together in a versioned artifact.
`training_metrics.json` records speaker splits, the precision-recall curve,
confusion matrix, and untouched-test metrics.

Custom data enters through `prepare_dataset.py`. Its CSV manifest requires a
relative file path, `Normal` or `Distress` label, and stable speaker identifier.
The importer confines sources to the manifest directory and outputs to the
project dataset directory, decodes each recording, and writes a validated
training manifest with content hashes.

## Upload security

The browser and Flask application enforce a 10 MiB upload limit. The server is
the authoritative boundary: it rejects missing, empty, oversized, and unsupported
files, stores uploads under random temporary names, deletes them after each
request, and returns generic errors instead of decoder internals.

The frontend and API are served from the same origin, so cross-origin access is
disabled. Audio contents remain untrusted and are decoded only for the first
three seconds. Compressed-media decoding still consumes compute; deployment-level
rate limiting should be added if the endpoint is exposed to untrusted public use.

`model.pkl` is a trusted, repository-controlled build artifact. `joblib` model
files must never be replaced with files supplied by users because loading an
untrusted serialized model can execute arbitrary code.

## Product boundary

The classifier maps acted RAVDESS anger and fear to "Distress." It is not a
medical device or validated emergency detector. User-facing results must be
treated as experimental classifications rather than emergency determinations.
