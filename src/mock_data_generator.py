"""
Synthetic data generation for the KYC Triage demo.

Everything in this file is FAKE. No real customers, documents, or transactions
are ever involved. We use the `Faker` library to generate realistic-looking
JSON so the rest of the system (agents, guardrails, UI) has something
believable to work with, without needing a real bank's data or a real
Document AI subscription.

Determinism: every generator function accepts an optional `seed`. If two
calls use the same seed, they produce the same output. This matters for a
portfolio demo — you want to be able to say "watch what happens with this
exact applicant" and get the same result every time, instead of random data
making your demo unreproducible.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from faker import Faker

DocumentType = Literal["passport", "drivers_license", "utility_bill", "national_id"]


def _seed_to_int(seed: str | int | None) -> int | None:
    """Turn any seed (string, int, or None) into an int Faker/random can use.

    Why this exists: we want callers to be able to pass a seed derived from
    an uploaded file's bytes (a hash, which is a hex string) OR a plain int
    OR nothing at all (fully random). Faker/random.seed() want an int or
    None, so this is the one place that conversion happens.
    """
    if seed is None:
        return None
    if isinstance(seed, int):
        return seed
    # Hash the string deterministically to an int (stable across runs,
    # unlike Python's built-in hash() which is randomized per-process).
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


@dataclass
class ExtractedDocument:
    """What a (simulated) Document AI call would hand back for one document."""

    document_type: DocumentType
    full_name: str
    date_of_birth: str  # ISO format, e.g. "1990-05-14"
    document_number: str
    issuing_country: str
    address: str
    expiration_date: str | None
    extraction_confidence: float  # 0.0–1.0, mimics a real OCR confidence score

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CustomerHistoryRecord:
    """A synthetic 'what we already know about this person' record.

    In a real bank this would come from a core banking system or a KYC
    vendor. Here, the Fraud & Anomaly Detection Agent queries this via an
    MCP tool (see src/mcp_tools_server.py) instead of calling this function
    directly — that's what makes the MCP usage real rather than decorative.
    """

    customer_id: str
    full_name_on_file: str
    date_of_birth_on_file: str
    address_on_file: str
    account_open_date: str
    prior_flags: list[str]
    average_monthly_transactions: int
    is_on_watchlist: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ApplicantCase:
    """A full simulated KYC case: the document just 'scanned', plus whatever
    history already exists on file for that name. The two are generated
    together so we can control how well (or badly) they match — that
    mismatch is exactly what the Fraud Agent is supposed to catch.
    """

    case_id: str
    scenario: Literal["clean", "fraud_mismatch", "watchlist_hit", "incomplete"]
    document: ExtractedDocument
    history: CustomerHistoryRecord | None  # None = brand new customer, no file on record
    submitted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def generate_extracted_document(
    seed: str | int | None = None,
    document_type: DocumentType | None = None,
) -> ExtractedDocument:
    """Simulate what Google Document AI's `process_document` response would
    contain after OCR + entity extraction on an ID or utility bill.

    We do NOT call any real OCR/Document AI API here — see the module
    docstring in guardrails_and_logging.py and the README for why (cost +
    credentials would break a public zero-cost demo).
    """
    fake = Faker()
    if (seeded := _seed_to_int(seed)) is not None:
        Faker.seed(seeded)
        random.seed(seeded)

    doc_type = document_type or random.choice(
        ["passport", "drivers_license", "utility_bill", "national_id"]
    )
    dob = fake.date_of_birth(minimum_age=18, maximum_age=85)
    issued_recently = doc_type in ("passport", "drivers_license", "national_id")

    return ExtractedDocument(
        document_type=doc_type,
        full_name=fake.name(),
        date_of_birth=dob.isoformat(),
        document_number=fake.bothify(text="??######").upper(),
        issuing_country=fake.country(),
        address=fake.address().replace("\n", ", "),
        expiration_date=(
            (date.today() + timedelta(days=365 * random.randint(1, 8))).isoformat()
            if issued_recently
            else None
        ),
        extraction_confidence=round(random.uniform(0.72, 0.99), 2),
    )


def generate_customer_history(
    seed: str | int | None = None,
    matching_name: str | None = None,
    matching_dob: str | None = None,
    matching_address: str | None = None,
    force_watchlist: bool = False,
) -> CustomerHistoryRecord:
    """Simulate a lookup against an existing customer/KYC database.

    `matching_name` / `matching_dob` / `matching_address`, when given, seed
    the record so those fields line up with a document we already
    generated — used to build a genuinely "clean" scenario where EVERY
    identity field matches, not just the name. (An earlier version of this
    function only aligned the name, which meant a real LLM reasoning over
    the still-random, still-mismatched date-of-birth/address fields could
    flag a "clean" case as risky — that's what the fraud_detection_agent's
    prompt explicitly asks it to do. Leave any of the three out to build a
    fraud/mismatch scenario for that specific field instead.)
    """
    fake = Faker()
    if (seeded := _seed_to_int(seed)) is not None:
        Faker.seed(seeded)
        random.seed(seeded)

    flags_pool = [
        "previous_chargeback",
        "address_change_last_30_days",
        "multiple_failed_logins",
        "large_cash_deposit_flagged",
        "sanctions_screening_hit",
    ]
    prior_flags = random.sample(flags_pool, k=random.choice([0, 0, 0, 1, 2]))

    return CustomerHistoryRecord(
        customer_id=fake.bothify(text="CUST-#######"),
        full_name_on_file=matching_name or fake.name(),
        date_of_birth_on_file=matching_dob or fake.date_of_birth(minimum_age=18, maximum_age=85).isoformat(),
        address_on_file=matching_address or fake.address().replace("\n", ", "),
        account_open_date=fake.date_between(start_date="-10y", end_date="-1d").isoformat(),
        prior_flags=prior_flags,
        average_monthly_transactions=random.randint(2, 60),
        is_on_watchlist=force_watchlist or ("sanctions_screening_hit" in prior_flags),
    )


def generate_applicant_case(
    seed: str | int | None = None,
    scenario: Literal["clean", "fraud_mismatch", "watchlist_hit", "incomplete", "random"] = "random",
) -> ApplicantCase:
    """Build one complete, internally-consistent demo case.

    This is the single entry point the UI's "Generate Random Applicant"
    button and the file-upload simulation both call. `scenario` lets the
    demo deliberately produce each of the three final decisions (approve /
    manual review / reject) on demand, which is much more useful for an
    interview walkthrough than pure randomness.
    """
    seeded_int = _seed_to_int(seed)
    if seeded_int is not None:
        random.seed(seeded_int)
    if scenario == "random":
        scenario = random.choice(["clean", "clean", "fraud_mismatch", "watchlist_hit", "incomplete"])

    fake = Faker()
    case_id = fake.bothify(text="KYC-????-####").upper()

    if scenario == "clean":
        document = generate_extracted_document(seed=seeded_int)
        history = generate_customer_history(
            seed=(seeded_int or 0) + 1,
            matching_name=document.full_name,
            matching_dob=document.date_of_birth,
            matching_address=document.address,
        )
    elif scenario == "fraud_mismatch":
        # Name on the "scanned" document deliberately does NOT match the
        # name already on file for that customer ID — a classic identity
        # inconsistency the Fraud Agent should catch.
        document = generate_extracted_document(seed=seeded_int)
        history = generate_customer_history(seed=(seeded_int or 0) + 1, matching_name=None)
    elif scenario == "watchlist_hit":
        # Identity fields all match (this scenario is specifically testing
        # the watchlist hard-stop, not identity mismatch) — only the
        # watchlist flag is forced on.
        document = generate_extracted_document(seed=seeded_int)
        history = generate_customer_history(
            seed=(seeded_int or 0) + 1,
            matching_name=document.full_name,
            matching_dob=document.date_of_birth,
            matching_address=document.address,
            force_watchlist=True,
        )
    elif scenario == "incomplete":
        document = generate_extracted_document(seed=seeded_int)
        document.extraction_confidence = round(random.uniform(0.3, 0.55), 2)
        document.expiration_date = None
        history = None  # brand-new customer, nothing on file yet
    else:  # pragma: no cover - defensive, Literal already constrains callers
        raise ValueError(f"Unknown scenario: {scenario}")

    return ApplicantCase(case_id=case_id, scenario=scenario, document=document, history=history)
