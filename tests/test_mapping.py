from privfusion.utils import find_mapping


def test_find_mapping() -> None:
    response = find_mapping("Horse")

    assert response is not None
    assert isinstance(response, list)
    assert len(response) > 0
    assert "http://dbpedia.org/resource/Horse" in response
