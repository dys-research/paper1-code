import json
import unittest

from scripts.extraction.core import parse_response, span_check


class ExtractionTests(unittest.TestCase):
    def test_only_contiguous_source_text_is_valid(self):
        sources = {"title": "A sensor", "abstract": "Detect pressure accurately.", "claim_1": "A sensor comprising a valve."}
        good = {"present": True, "source_field": "abstract", "evidence_span": "pressure accurately"}
        self.assertTrue(span_check(good, sources)[0])
        paraphrase = {**good, "evidence_span": "accurate pressure detection"}
        self.assertFalse(span_check(paraphrase, sources)[0])
        wrong_field = {**good, "source_field": "title"}
        self.assertFalse(span_check(wrong_field, sources)[0])
        self.assertFalse(span_check({"present": False}, sources)[0])

    def test_invalid_role_schema_is_rejected(self):
        valid = {role: {"present": True, "confidence": .8} for role in ["P", "M", "E"]}
        self.assertEqual(set(parse_response(json.dumps(valid))), {"P", "M", "E"})
        invalid = {**valid, "P": {"present": "yes"}}
        with self.assertRaises(ValueError):
            parse_response(json.dumps(invalid))
        invalid = {**valid, "E": {"present": True, "confidence": 1.5}}
        with self.assertRaises(ValueError):
            parse_response(json.dumps(invalid))


if __name__ == "__main__":
    unittest.main()
