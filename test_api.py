from pathlib import Path

from fastapi.testclient import TestClient

from api import app
from schema import EXPECTED_HEADERS

client = TestClient(app)
sample = Path(__file__).resolve().parent / 'Unihack_SampleDataset-Input.csv'
if not sample.exists():
    sample = Path('Unihack_SampleDataset-Input.csv')
with sample.open('rb') as handle:
    response = client.post('/process', files={'file': ('sample.csv', handle, 'text/csv')})
assert response.status_code == 200, response.text
body = response.json()
assert len(body['records']) == 1000
assert body['delivery_headers'] == EXPECTED_HEADERS
assert len(body['traceability'][0]['fields']) + len(body['traceability'][0]['missing_data']) == 252

with sample.open('rb') as handle:
    csv_response = client.post('/process/csv', files={'file': ('sample.csv', handle, 'text/csv')})
assert csv_response.status_code == 200, csv_response.text
assert csv_response.headers['content-type'].startswith('text/csv')
print({'json_status': response.status_code, 'csv_status': csv_response.status_code, 'json_records': len(body['records']), 'traceability_fields_per_record': len(body['traceability'][0]['fields'])})
