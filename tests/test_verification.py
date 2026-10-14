import unittest
from fraudvision.verification import verify_amounts_text, combined_decision

class OCRVerificationTests(unittest.TestCase):
    def test_consistent(self):
        check=verify_amounts_text('Subtotal $271.53\nTax $19.01\nTOTAL $290.54')
        self.assertEqual(check.status,'CONSISTENT')
        self.assertEqual(check.difference,'+0.00')
    def test_mismatch(self):
        check=verify_amounts_text('Subtotal $271.53\nTax $19.01\nTOTAL $351.54')
        self.assertEqual(check.status,'MISMATCH')
        self.assertEqual(check.difference,'+61.00')
    def test_invoice_with_tax_percentage(self):
        check=verify_amounts_text('Subtotal $1,200.00\nTax (7%) $84.00\nTOTAL $1,284.00')
        self.assertEqual(check.status,'CONSISTENT')
    def test_missing_value_unreadable(self):
        self.assertEqual(verify_amounts_text('Subtotal $100.00\nTotal $110.00').status,'UNREADABLE')
    def test_ambiguous_total_unreadable(self):
        self.assertEqual(verify_amounts_text('Subtotal $100.00\nTax $7.00\nTotal $107.00\nTotal $108.00').status,'UNREADABLE')
    def test_missing_tesseract(self):
        self.assertEqual(verify_amounts_text('Tesseract not installed or not found.').status,'OCR_UNAVAILABLE')
    def test_combined_rule(self):
        ok=verify_amounts_text('Subtotal $100.00\nTax $7.00\nTotal $107.00')
        bad=verify_amounts_text('Subtotal $100.00\nTax $7.00\nTotal $109.00')
        self.assertEqual(combined_decision(False,ok),'NO FLAG')
        self.assertEqual(combined_decision(False,bad),'REVIEW')
        self.assertEqual(combined_decision(True,ok),'REVIEW')
        self.assertEqual(combined_decision(False,None),'NO FLAG')
