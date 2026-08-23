import csv
from pathlib import Path

from fastapi.testclient import TestClient

from agents import Orchestrator
from api import app
from evaluate import evaluate
from schema import EXPECTED_HEADERS
from source_research import LocalDocumentRAG, SourceDocument

sample = Path(__file__).resolve().parent / 'Unihack_SampleDataset-Input.csv'
expected = Path(__file__).resolve().parent / 'Unihack_ExpectedOutput-DeliveryFormat.csv'
if not sample.exists():
    sample = Path('Unihack_SampleDataset-Input.csv')
if not expected.exists():
    expected = Path('Unihack_ExpectedOutput-DeliveryFormat.csv')

with expected.open(newline='', encoding='utf-8-sig') as handle:
    expected_headers = next(csv.reader(handle))
assert len(EXPECTED_HEADERS) == 252
assert EXPECTED_HEADERS == expected_headers

custom_rows = [{
    'Mfg_Part_Num': 'CUSTOM-ABC-123',
    'Part_Desc': 'Custom ABC-123 6 in Ceramic Cut-Off Disc 10 pcs',
    'E1_Brand': '-- Unbranded --',
    'Unilog_Brand': '-- No Unilog Brand --',
    'DIB_Brand': '-- No DIB Brand --',
    'Part_Manuf': 'Example Manufacturer (EXMPL)',
}]
custom = Orchestrator(use_llm=False).process(rows=custom_rows)
assert custom['records'][0]['Mfg_Part_Num'] == 'CUSTOM-ABC-123'
assert custom['records'][0]['MANUFACTURER_NAME'] == 'Example Manufacturer'
assert custom['records'][0]['MANUFACTURER_PART_NUMBER'] == 'CUSTOM-ABC-123'
assert custom['validation']['valid']
assert len(custom['traceability'][0]['fields']) + len(custom['traceability'][0]['missing_data']) == 252

local_doc = SourceDocument(url='https://example.test/product', title='Example Product', text='Custom ABC-123 is a 6 inch ceramic cut-off disc. Includes 10 pieces.')
rag = LocalDocumentRAG([local_doc])
retrieved = rag.search('Custom ABC-123 ceramic cut-off disc', k=1)
assert retrieved and 'Custom ABC-123' in retrieved[0][2]

with sample.open('rb') as handle:
    response = TestClient(app).post('/process', files={'file': ('sample.csv', handle, 'text/csv')})
assert response.status_code == 200, response.text
body = response.json()
assert len(body['records']) == 1000
assert body['delivery_headers'] == EXPECTED_HEADERS
metrics = evaluate(body)
assert metrics['schema_valid'] and metrics['records'] == 1000

print({
    'schema_columns': len(EXPECTED_HEADERS),
    'custom_dynamic_input': True,
    'local_rag_retrieval': True,
    'api_status': response.status_code,
    'sample_records': len(body['records']),
    'sample_coverage': metrics['populated_cell_coverage'],
})
