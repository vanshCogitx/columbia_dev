import pytest
from models.product import Product

def test_price_conversion_from_numeric_with_currency_meta():
    # Simulate raw data where price_raw is a numeric value (no currency string)
    raw = {
        "price": 4499.0,
        "name": "Test Product",
        "brand": "TestBrand",
        "description": "A test product",
        "image": "http://example.com/image.jpg",
        "offers": {},
    }
    # serp_meta provides currency information
    serp_meta = {"priceCurrency": "INR"}
    product = Product.from_raw(raw, "http://example.com/product", serp_meta)
    assert product is not None
    # Expected conversion: INR to USD using rate 0.012
    expected_price = round(4499.0 * 0.012, 2)
    assert product.price == expected_price
