import csv
from pathlib import Path

from api import app

out = Path('sample_output.csv')
expected = Path(__file__).resolve().parent / 'Unihack_ExpectedOutput-DeliveryFormat.csv'
if not expected.exists():
    expected = Path('Unihack_ExpectedOutput-DeliveryFormat.csv')
with out.open(newline='', encoding='utf-8') as handle:
    out_headers = next(csv.reader(handle))
with expected.open(newline='', encoding='utf-8-sig') as handle:
    expected_headers = next(csv.reader(handle))
with out.open(encoding='utf-8') as handle:
    output_rows = sum(1 for _ in handle) - 1
print({
    'routes': sorted(route.path for route in app.routes),
    'output_rows': output_rows,
    'output_columns': len(out_headers),
    'headers_match': out_headers == expected_headers,
    'output_bytes': out.stat().st_size,
})
assert out_headers == expected_headers
assert output_rows == 1000
