# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=5,<6", "jsonschema>=4,<5", "Pillow>=12,<13"]
# ///
"""Prepare reviewed Harvard headers as a benchmark config; no inference or upload."""
import argparse
import json
import shutil
from pathlib import Path

from datasets import Dataset, Features, Image, Value
from dataset_contract import file_hash, json_text, sha256, validate_config
from scorer import SCORER_VERSION
from version import __version__

CONFIG = 'harvard-botany-headers'
BENCHMARK = 'small-models-for-glam/glam-extraction-benchmark'


def export(review_file, images, audit_dir, source_card, output):
    review = json.loads(review_file.read_text())
    schema = review['target_schema']
    schema_text = json_text(schema)
    config_dir = output / CONFIG
    config_dir.mkdir(parents=True, exist_ok=False)
    rows, pending = [], []
    for item in review['items']:
        pixels = (images / f"row{item['source_row']}.jpg").read_bytes()
        if sha256(pixels) != item['image_sha256']:
            raise ValueError(f"Image checksum mismatch: {item['id']}")
        provenance = {k: item[k] for k in (
            'source_id', 'source_row', 'source_page', 'source_url', 'image_sha256',
            'printed_fields_reviewed_by_human', 'correction_reviewed_by_human',
            'schema_applicable', 'scoring_eligible', 'review_status',
        )}
        provenance.update(source_dataset=review['source_dataset'],
                          source_revision=review['source_revision'],
                          annotation_method='Visual model draft; printed fields human-reviewed. '
                          'Correction absence visually audited; replacement review recorded per row.')
        row = {'id': item['id'], 'image': {'bytes': pixels, 'path': f"{item['id']}.jpg"},
               'target_schema': schema_text,
               'expected_output': json_text(item['expected_output']),
               'provenance': json_text(provenance)}
        if item['scoring_eligible']:
            rows.append(row)
        else:
            provenance['reason'] = 'Handwritten replacement transcription awaits review'
            provenance['expected_output_is_draft'] = True
            row['provenance'] = json_text(provenance)
            pending.append(row)
    audit = json.loads((audit_dir / 'findings.json').read_text())
    for item in audit['cards']:
        if item['schema_assessment'] == 'fits':
            continue
        metadata = json.loads((audit_dir / 'metadata' / f"row{item['source_row']}.json").read_text())
        pixels = (audit_dir / 'images' / f"row{item['source_row']}.jpg").read_bytes()
        if sha256(pixels) != metadata['image_sha256']:
            raise ValueError('Audit image checksum mismatch')
        provenance = {
            'source_dataset': review['source_dataset'], 'source_revision': review['source_revision'],
            'source_id': item['source_id'], 'source_row': item['source_row'],
            'source_page': item['page'], 'image_sha256': sha256(pixels),
            'schema_applicable': False, 'scoring_eligible': False,
            'reason': item['schema_assessment'], 'review_status': 'visual_schema_audit_only',
        }
        pending.append({'id': f"{item['source_id']}:{item['page']:04d}",
                        'image': {'bytes': pixels, 'path': f"row{item['source_row']}.jpg"},
                        'target_schema': schema_text, 'expected_output': None,
                        'provenance': json_text(provenance)})
    features = Features({'id': Value('string'), 'image': Image(),
                         'target_schema': Value('string'), 'expected_output': Value('string'),
                         'provenance': Value('string')})
    for split, records in [('test', rows), ('review', pending)]:
        Dataset.from_list(records, features=features).to_parquet(config_dir / f'{split}.parquet')
    (config_dir / 'target-schema.json').write_text(json.dumps(schema, indent=2) + '\n')
    shutil.copyfile(source_card, config_dir / 'source-card.md')
    manifest = {
        'contract_version': '1', 'benchmark_id': BENCHMARK, 'config': CONFIG, 'split': 'test',
        'title': 'Harvard Botany Libraries: printed taxon headings and handwritten replacements',
        'institution': {'name': 'Harvard University Botany Libraries',
                        'url': 'https://library.harvard.edu/libraries/botany'},
        'source': {'repo_id': review['source_dataset'], 'revision': review['source_revision'],
                   'config': 'default', 'split': 'train', 'source_card_sha256': file_hash(source_card)},
        'license': 'public-domain',
        'rights_note': 'Source card reports Harvard NOT_IN_COPYRIGHT; preserve Harvard attribution.',
        'converter': {'name': 'harvard-headers', 'version': '1'}, 'harness_version': __version__,
        'item_count': len(rows), 'item_ids_sha256': sha256(json_text([r['id'] for r in rows]).encode()),
        'target_schema_sha256': sha256(schema_text.encode()), 'data_file': f'{CONFIG}/test.parquet',
        'data_sha256': file_hash(config_dir / 'test.parquet'),
        'scoring': {'id': 'typed-kie', 'version': SCORER_VERSION, 'exclude': [],
                    'schema_projection': 'nullable-to-value-v1', 'null_policy': 'Absent value, not unreadable text'},
        'label_production': {'draft': 'Visual assistant transcription, not source OCR',
                             'printed_fields': 'Human reviewed',
                             'corrections': 'Replacement strings human reviewed; absence visually audited',
                             'review_metadata': 'Per-field human review flags in provenance'},
        'selection': {'type': 'curated proof of concept', 'reviewed_candidate_count': 30,
                      'note': 'Not a representative sample of all Harvard cards'},
        'review_split': {'data_file': f'{CONFIG}/review.parquet', 'item_count': len(pending),
                         'scored': False, 'note': 'Unsupported or pending cards retained; no gold for unannotated cards'},
        'task_instructions': 'Extract the original printed taxon name and authority, and any explicit '
                             'handwritten taxonomic replacement attached to the heading. Preserve literal '
                             'spelling and abbreviations; do not update taxonomy. Return the three fields '
                             'as JSON. Use null for absent values. Ignore body notes, initials, editing '
                             'marks and rulings; a strike-through alone is not a replacement.',
        'registration': {'status': 'not_registered', 'proposed_task_id': CONFIG},
    }
    (config_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = validate_config(output, CONFIG)
    (config_dir / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['valid']:
        raise ValueError(report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['review', 'images', 'audit', 'source-card', 'output']:
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.review, args.images, args.audit, args.source_card, args.output), indent=2))
