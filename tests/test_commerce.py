import json

from cortex.tools import commerce, web


def test_multi_product_lookup_searches_each_item_and_filters_bad_offers(monkeypatch):
    queries = []

    def search(query, _limit):
        queries.append(query)
        if "Alpha Pro" in query and "Beta Slim" not in query:
            return [
                {
                    "title": "Protective Stand for Alpha Pro Console",
                    "url": "https://www.noon.com/uae-en/alpha-pro-stand/A1/p/",
                    "snippet": "AED 100.00",
                },
                {
                    "title": "Alpha Pro Console",
                    "url": "https://www.noon.com/uae-en/alpha-pro/A2/p/",
                    "snippet": "AED 199.00",
                },
                {
                    "title": "Alpha Pro Console International Version",
                    "url": "https://www.noon.com/uae-en/alpha-pro/A3/p/",
                    "snippet": "AED 3699.00",
                },
                {
                    "title": "Alpha Pro Console UAE Version",
                    "url": "https://www.noon.com/uae-en/alpha-pro/A4/p/",
                    "snippet": "AED 3599.00",
                },
            ]
        if "Beta Slim" in query:
            return [
                {
                    "title": "Beta Slim Console",
                    "url": "https://www.noon.com/uae-en/beta-slim/B1/p/",
                    "snippet": "AED 2315.00",
                }
            ]
        return []

    monkeypatch.setattr(web, "_provider_search", search)

    result = json.loads(
        commerce.product_prices.invoke(
            {
                "product": "Alpha Pro and Beta Slim",
                "region": "AE",
                "retailer": "noon.com",
            }
        )
    )

    assert len(queries) == 2
    assert {offer["product"] for offer in result["offers"]} == {
        "Alpha Pro",
        "Beta Slim",
    }
    assert {offer["price_value"] for offer in result["offers"]} == {
        2315.0,
        3599.0,
        3699.0,
    }
