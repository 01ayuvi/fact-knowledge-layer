"""entities.py tests. The Morparia case is drawn straight from
docs/CASE_DOSSIER.md §2 -- the prospectus and the annual report refer to
her with two different surface forms, and resolving them to the same
canonical id is a precondition for the SUPERSEDES/TEMPORAL_STATE_CHANGE
relation the dossier calls the "sophistication beat" to even be possible:
without a shared canonical_id, claim_key() (src/fkl/store/models.py) never
groups the two facts together in the first place."""

from src.fkl.normalize.entities import canonicalize


def test_delhivery_name_variants_share_canonical_id():
    ids = {
        canonicalize("Delhivery Limited"),
        canonicalize("Delhivery Ltd"),
        canonicalize("Delhivery Ltd."),
        canonicalize("Delhivery"),
    }
    assert len(ids) == 1


def test_the_company_resolves_via_doc_subject():
    assert canonicalize("the Company", doc_subject="Delhivery Limited") == canonicalize(
        "Delhivery Limited"
    )


def test_your_company_resolves_via_doc_subject():
    assert canonicalize("your Company", doc_subject="Delhivery Limited") == canonicalize(
        "Delhivery Limited"
    )


def test_self_reference_without_doc_subject_does_not_crash():
    # No doc context available -- should degrade gracefully, not raise.
    assert canonicalize("the Company") == "unknown_self_reference"


def test_morparia_name_variants_share_canonical_id_critical():
    """docs/CASE_DOSSIER.md §2: 'Prospectus 2022, p88: Kalpana Jaisingh
    Morparia is a Non-Executive Independent Director...' vs 'Annual
    Report FY24, p91: Ms. Kalpana Jaisingh Morparia Non Executive -
    Independent Director (resigned w.e.f. February 11, 2023)'."""
    prospectus_form = canonicalize("Kalpana Jaisingh Morparia")
    ar_form = canonicalize("Ms. Kalpana Jaisingh Morparia")
    assert prospectus_form == ar_form


def test_other_directors_from_dossier():
    """docs/CASE_DOSSIER.md §2: 'Corroborating instances of the same
    pattern in the same document (AR p91): Munish Ravinder Varma
    (resigned 29 Jun 2022), Agus Tandiono (resigned 8 Apr 2022) -- both
    listed as active in the 2022 prospectus.'"""
    assert canonicalize("Munish Ravinder Varma") == canonicalize("Mr. Munish Ravinder Varma")
    assert canonicalize("Agus Tandiono") == canonicalize("Mr. Agus Tandiono")


def test_unknown_entity_gets_stable_deterministic_id():
    a = canonicalize("Vinculum Solutions Private Limited")
    b = canonicalize("Vinculum Solutions Private Limited")
    assert a == b
    assert a != canonicalize("Falcon Autotech Private Limited")


def test_case_and_whitespace_insensitive():
    assert canonicalize("delhivery limited") == canonicalize("  Delhivery   Limited  ")
