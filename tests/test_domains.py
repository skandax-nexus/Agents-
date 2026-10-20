from omniresearch.domains import DOMAINS, categories, get_domain, route_domains
from omniresearch.domains.router import validate_domain_ids


def test_registry_is_broad():
    ids = {d.id for d in DOMAINS}
    for must in ["physics", "chemistry", "biology", "mathematics", "climate_science", "oceanography", "hydrology",
                 "ecology", "soil_science", "forestry", "air_quality", "waste_management", "renewable_energy",
                 "environmental_policy", "medicine", "economics", "law", "history", "philosophy", "arts", "general"]:
        assert must in ids, must
    assert len(DOMAINS) >= 60
    assert "Environmental sciences" in categories()
    assert all(d.description and d.methods and d.sources for d in DOMAINS)


def test_unique_ids():
    ids = [d.id for d in DOMAINS]
    assert len(ids) == len(set(ids))


def test_routing_environmental():
    picked = [d.id for d in route_domains("How does ocean acidification affect coral reef fisheries?")]
    assert "oceanography" in picked


def test_routing_multi_domain():
    picked = [d.id for d in route_domains("What is the economic cost of air pollution from coal power plants?", max_domains=4)]
    assert "air_quality" in picked
    assert "energy_systems" in picked or "economics" in picked


def test_routing_never_empty():
    picked = route_domains("xyzzy plugh")
    assert picked and picked[0].id == "general"


def test_unknown_domain_falls_back():
    assert get_domain("not-a-domain").id == "general"
    out = validate_domain_ids(["bogus", "Climate-Science"], "anything about glaciers", 3)
    assert out[0].id == "climate_science"
