"""Canonical extraction handoff must preserve provenance-based missingness."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
STEP = ROOT / 'scripts/measurement/04_prepare_role_inputs.py'


def fingerprint(title):
    sources = {'title': title, 'abstract': 'A stated technical effect.', 'claim_1': 'A sensor measures pressure.'}
    return hashlib.sha256(json.dumps(sources, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class CanonicalRoleHandoff(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        text_dir = self.root / 'work/data/data'
        text_dir.mkdir(parents=True)
        records = []
        for pid in range(1, 6):
            records.append({'patent_id': str(pid), 'grant_year': 2005, 'is_focal_candidate': True,
                            'patent_title': 'A technical task.',
                            'patent_abstract': 'A stated technical effect.',
                            'independent_claim_text': 'A sensor measures pressure.', 'full_input_available': True})
        pq.write_table(pa.Table.from_pylist(records), text_dir / 'a61b_text_base_1995_2020.parquet')
        role_dir = self.root / 'work/extraction'
        role_dir.mkdir(parents=True)
        roles = []
        for pid in range(1, 6):
            role = {'patent_id': str(pid), 'role': 'P', 'present': True, 'source_field': 'title',
                    'evidence_span': 'A technical task.', 'role_status': 'VALID_ORIGINAL_SPAN',
                    'current_input_sources_sha256': fingerprint('A technical task.'),
                    'input_hash_verified': True, 'confidence': 0.9, 'extraction_model': 'fixture'}
            if pid == 1:
                # Correct current snapshot, but an explicitly approved legacy import.
                role['input_hash_verified'] = False
            elif pid == 2:
                # A present, locatable span cannot repair an extraction from changed inputs.
                role['role_status'] = 'INPUT_CHANGED'
            elif pid == 3:
                role.update(present=None, source_field=None, evidence_span=None, role_status='NO_OUTPUT')
            elif pid == 4:
                role['evidence_span'] = 'This text never occurs in the source.'
            elif pid == 5:
                # Source content changed after the canonical export.
                role['current_input_sources_sha256'] = fingerprint('An earlier source title.')
            roles.append(role)
        pq.write_table(pa.Table.from_pylist(roles), role_dir / 'role_inputs_primary.parquet')
        (self.root / 'broken_parsed.jsonl').write_text('deliberately invalid parsed response', encoding='utf-8')
        self.config = self.root / 'config.toml'
        self.config.write_text('[paths]\nwork_dir="work"\n[inputs]\nrole_outputs="broken_parsed.jsonl"\n', encoding='utf-8')
        self.env = os.environ.copy()
        self.env['TFSC_CONFIG'] = str(self.config)
        self.env['PYTHONPATH'] = str(ROOT) + os.pathsep + self.env.get('PYTHONPATH', '')

    def run_step(self, *args):
        return subprocess.run([sys.executable, str(STEP), *args], env=self.env,
                              capture_output=True, text=True, check=False)

    def test_canonical_file_overrides_raw_response_setting_and_preserves_failures(self):
        result = self.run_step()
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = pq.read_table(self.root / 'work/measurement/data/role_inputs_primary.parquet').to_pylist()
        problem = {row['patent_id']: row for row in rows if row['role'] == 'P'}
        self.assertEqual(problem['1']['role_status'], 'VALID_ORIGINAL_SPAN')
        self.assertFalse(problem['1']['input_hash_verified'])
        self.assertEqual(problem['1']['embedding_text'], 'A technical task.')
        for pid, expected in [('2', 'INPUT_CHANGED'), ('3', 'NO_OUTPUT'),
                              ('4', 'INVALID_OR_NONVERBATIM_SPAN'), ('5', 'INPUT_CHANGED')]:
            self.assertEqual(problem[pid]['role_status'], expected)
            self.assertIsNone(problem[pid]['embedding_text'])
            self.assertFalse(problem[pid]['usable_for_primary_embedding'])
        self.assertEqual(len(rows), 15)

    def test_real_exporter_to_measurement_through_runner(self):
        predictions = []
        for pid in [1, 2, 4, 5]:
            prediction = {'patent_id': str(pid), 'model': 'fixture',
                          'input_sources_sha256': fingerprint('A technical task.'),
                          'elements': {
                              'P': {'present': True, 'source_field': 'title', 'evidence_span': 'A technical task.'},
                              'M': {'present': False},
                              'E': {'present': True, 'source_field': 'abstract', 'evidence_span': 'A stated technical effect.'}}}
            if pid == 2:
                prediction['input_sources_sha256'] = '0' * 64
            if pid == 4:
                prediction['elements']['P']['evidence_span'] = 'A generated paraphrase.'
            if pid == 5:
                prediction.pop('input_sources_sha256')
            predictions.append(prediction)
        parsed = self.root / 'parsed.jsonl'
        parsed.write_text(''.join(json.dumps(row) + '\n' for row in predictions), encoding='utf-8')
        self.config.write_text('[paths]\nwork_dir="work"\n[inputs]\nrole_outputs="parsed.jsonl"\n', encoding='utf-8')
        runner = [sys.executable, str(ROOT / 'run.py'), 'extract', '--step', 'prepare_roles']
        exported = subprocess.run(runner, env=self.env, capture_output=True, text=True, check=False)
        self.assertEqual(exported.returncode, 0, exported.stderr)
        measured = self.run_step()
        self.assertEqual(measured.returncode, 0, measured.stderr)
        output = self.root / 'work/measurement/data/role_inputs_primary.parquet'
        problem = {row['patent_id']: row for row in pq.read_table(output).to_pylist() if row['role'] == 'P'}
        for pid, status in [('1', 'VALID_ORIGINAL_SPAN'), ('2', 'INPUT_CHANGED'), ('3', 'NO_OUTPUT'),
                            ('4', 'INVALID_OR_NONVERBATIM_SPAN'), ('5', 'INPUT_HASH_MISSING')]:
            self.assertEqual(problem[pid]['role_status'], status)
        # The exporter's explicit legacy-import decision must survive the handoff.
        exported = subprocess.run([*runner, '--', '--allow-missing-input-hash'], env=self.env,
                                  capture_output=True, text=True, check=False)
        self.assertEqual(exported.returncode, 0, exported.stderr)
        measured = self.run_step()
        self.assertEqual(measured.returncode, 0, measured.stderr)
        problem = {row['patent_id']: row for row in pq.read_table(output).to_pylist() if row['role'] == 'P'}
        self.assertEqual(problem['5']['role_status'], 'VALID_ORIGINAL_SPAN')
        self.assertFalse(problem['5']['input_hash_verified'])
        self.assertEqual(problem['2']['role_status'], 'INPUT_CHANGED')

    def test_direct_raw_jsonl_is_rejected(self):
        result = self.run_step('--roles', str(self.root / 'broken_parsed.jsonl'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('canonical role Parquet', result.stderr)


if __name__ == '__main__':
    unittest.main()
