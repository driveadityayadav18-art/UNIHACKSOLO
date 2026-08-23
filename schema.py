"""Pydantic schema for the UniHack expected delivery format.

The delivery CSV is intentionally kept flat and exactly matches the supplied
ExpectedOutput-DeliveryFormat.csv header order. Traceability is carried beside
that flat record in the API/UI envelope so the delivery CSV remains compatible.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

EXPECTED_HEADERS: List[str] = [
    "MFR URL", "Ref URL 1", "Ref URL 2", "Ref URL 3", "Ref URL 4", "Ref URL 5",
    "PART_NUMBER", "Dept", "Class", "Fine", "SKU - MY_PART_NUMBER", "Mfg_Part_Num",
    "Part_Desc", "E1_Brand", "Unilog_Brand", "DIB_Brand", "Part_Manuf",
    "MANUFACTURER_NAME", "BRAND_NAME", "TRADE_NAME", "MANUFACTURER_PART_NUMBER",
    "ALTERNATE_PART_NUMBER", "Classpath", "MOBILE_DESC", "INVOICE_DESC", "SHORT_DESC",
    "LONG_DESC1", "RETAIL_DESC", "MARKETING_DESCRIPTION", *[f"ITEM_FEATURES_{i}" for i in range(1, 21)],
    "With", "Standard/Approvals", "Prop 65", "Application", "Includes", "Product Name",
    *[part for i in range(1, 51) for part in (f"ATTRIBUTE_LABEL {i}", f"ATTRIBUTE_VALUE {i}", f"ATTRIBUTE_UOM {i}")],
    "UPC", "EAN", "GTIN", "UNSPSC", "Warranty", "List Price", "Selling Qty", "Selling UOM",
    "Standard Packaging Information", "LENGTH", "LENGTH_UOM", "HEIGHT", "HEIGHT_UOM", "WIDTH",
    "WIDTH_UOM", "WEIGHT", "WEIGHT_UOM", "VOLUME", "VOLUME_UOM", "Product Image",
    "Alternate Image 1", "Alternate Image 2", "Alternate Image 3", "Alternate Image 4", "SDS",
    "SDS_1", "Warranty Information", "Catalog", "Specification Sheet", "Instruction/Installation Manual",
    "Service Manual", "Owners/User Manual", "Line Drawing", "MTR", "RoHS", "Full Engineering Drawing",
    "Energy Star Guide", "Technical Bulletin", "Submittal", "Compatibility Chart", "Size Chart",
    "Product Label/Insert", "Video Link", "Video Link 1", "Country Of Origin", "Discontinued",
    "Actual Image (Yes/No)",
]

INPUT_HEADERS = ["Mfg_Part_Num", "Part_Desc", "E1_Brand", "Unilog_Brand", "DIB_Brand", "Part_Manuf"]
PLACEHOLDER_VALUES = {"", "--", "-", "n/a", "na", "none", "null", "-- unbranded --", "-- no unilog brand --", "-- no dib brand --"}

if len(EXPECTED_HEADERS) != 252:
    raise RuntimeError(f"Expected exactly 252 delivery columns, found {len(EXPECTED_HEADERS)}")

# Pydantic field aliases preserve headers containing spaces, slashes, and hyphens.
_PRODUCT_FIELDS = {
    header.replace(" ", "_").replace("/", "_").replace("-", "_"): (Optional[str], Field(default=None, alias=header))
    for header in EXPECTED_HEADERS
}
ProductRecord = create_model("ProductRecord", __base__=BaseModel, **_PRODUCT_FIELDS)
ProductRecord.model_config = ConfigDict(populate_by_name=True, extra="forbid")


class FieldTrace(BaseModel):
    """Evidence and confidence for one output field."""

    value: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    source: str
    source_excerpt: str = ""
    reasoning: str
    status: str = "observed"


class AIFieldSuggestion(BaseModel):
    """Structured model returned by the optional LLM enrichment step."""

    value: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    source_url: Optional[str] = None


class AIEnrichmentResponse(BaseModel):
    """Schema-constrained field suggestions; unknown fields are rejected by the caller."""

    fields: Dict[str, AIFieldSuggestion] = Field(default_factory=dict)


class ProductTraceability(BaseModel):
    """Per-field traceability for a product record."""

    fields: Dict[str, FieldTrace]
    missing_data: List[str] = Field(default_factory=list)
    hitl_fields: List[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    hitl_required: bool = False


class ProcessResponse(BaseModel):
    """API response: exact delivery records plus traceability metadata."""

    records: List[Dict[str, Optional[str]]]
    traceability: List[ProductTraceability]
    validation: ValidationReport
    delivery_headers: List[str] = Field(default_factory=lambda: EXPECTED_HEADERS.copy())


def model_to_delivery_dict(record: Any) -> Dict[str, Optional[str]]:
    """Serialize a ProductRecord using the original CSV header names and order."""
    if hasattr(record, "model_dump"):
        data = record.model_dump(by_alias=True, exclude_none=False)
    else:
        data = record.dict(by_alias=True, exclude_none=False)
    return {header: data.get(header) for header in EXPECTED_HEADERS}


def is_missing(value: Any) -> bool:
    """Treat the source's placeholder conventions as missing values."""
    if value is None:
        return True
    return str(value).strip().lower() in PLACEHOLDER_VALUES
