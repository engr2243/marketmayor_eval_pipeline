"""
What the model is allowed to say about a contact.

The model never returns a score or a ranking. It returns dated, quoted,
attributable claims, and anything downstream that wants a priority derives it
from those deterministically. That split is the point of the design: a ranking
change can always be traced to a specific sentence a specific person wrote on a
specific day, and re-running on the same inputs gives the same answer.

Schema constraints worth knowing before editing, because structured outputs are
stricter than plain JSON mode:

- every object needs `extra="forbid"`, which becomes `additionalProperties: false`
- numeric bounds, string lengths and regex patterns are not supported; anything
  needing a bounded value becomes an enum, anything needing a length limit is
  trimmed after validation instead
- no recursive models
- there is a ceiling on how many optional fields will compile into a grammar, so
  each one is rent — a claim needing three related values pays for one combined
  field rather than three
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Strict = ConfigDict(extra="forbid")

ConversationState = Literal[
    "active",                # exchange in flight, both sides engaged
    "awaiting_their_reply",  # we sent last, no response yet
    "awaiting_my_reply",     # they sent last, we have not responded
    "concluded_won",         # completed successfully
    "concluded_lost",        # they went elsewhere, withdrew, or said no
    "dormant",               # was a real exchange, has gone quiet
    "no_conversation",       # no substantive two-way exchange on record
]

FactKind = Literal[
    "commitment",            # a dated future obligation by either party
    "intent",                # stated intent to transact
    "timeline",              # a horizon without a firm date
    "objection",             # a stated blocker
    "life_event",            # a move, a job change, a family change
    "preference",            # a durable fact worth remembering
    "business_opportunity",  # recruiting, partnership or investment in US
    "relationship",          # another person is named and related
    "referred_by",           # someone sent this contact to us
    "referral_made",         # they sent us somebody
    "engagement",            # they transacted with us, or referred someone
    "endorsement",           # they said they would recommend us
    "contact_request",       # they asked to be contacted at or around a time
    "contact_detail",        # where they live, work, or a personal date
]

Confidence = Literal["low", "medium", "high"]

RelationshipKind = Literal[
    "spouse", "partner", "child", "parent", "family", "business", "other"
]
EngagementKind = Literal["direct", "referral"]
IntentState = Literal[
    "active_buying", "active_selling", "future_buying", "future_selling",
    "not_looking",
]

# The plain facts that belong in columns rather than prose. A note is often the
# only place they are ever written down, so extraction offers them as the fields
# they are instead of burying them in a summary.
DetailField = Literal[
    "street_address", "city", "state", "postal_code",
    "birthday", "home_anniversary",
    "company", "job_title",
    "price_range", "property_note",
]


class Fact(BaseModel):
    """One attributable claim, tied to the message it came from."""

    model_config = Strict

    kind: FactKind
    summary: str = Field(description="One sentence, written for a person to read.")

    evidence: str = Field(
        description=(
            "A VERBATIM quote from exactly one message in the transcript. Copy "
            "it character for character. Do not paraphrase, join quotes from "
            "different messages, correct spelling, or clean it up."
        )
    )
    source_message_id: str = Field(
        description=(
            "The bracketed id of the message the evidence came from. Must be "
            "one of the ids shown in the transcript."
        )
    )
    occurred_on: str = Field(description="ISO date (YYYY-MM-DD) of that message.")
    actor: Literal["contact", "us"] = Field(description="Who said or committed this.")

    due_on: str | None = Field(
        default=None,
        description=(
            "Absolute ISO date (YYYY-MM-DD) when this becomes actionable. "
            "Resolve relative language against the message's OWN date: 'call me "
            "in 3 months' in a message sent 2026-03-04 is 2026-06-04. Null when "
            "there is no resolvable future date."
        ),
    )

    person_name: str | None = Field(
        default=None,
        description=(
            "The OTHER person this fact names, spelled exactly as the message "
            "spells them: a family member, the referrer, or the person who was "
            "referred. Null when the fact names nobody. Required for "
            "relationship and referred_by facts."
        ),
    )
    relationship_kind: RelationshipKind | None = Field(
        default=None, description="For relationship facts only: how they are related."
    )
    engagement_kind: EngagementKind | None = Field(
        default=None,
        description=(
            "For engagement facts only. direct = they transacted with us; "
            "referral = they sent us somebody."
        ),
    )
    intent: IntentState | None = Field(
        default=None,
        description=(
            "The buying or selling position this fact establishes, when it does. "
            "Null when the fact says nothing about a transaction."
        ),
    )
    event_on: str | None = Field(
        default=None,
        description=(
            "ISO date when a PAST event happened. Use the first of the month or "
            "year when that is all the message gives ('bought in 2022' is "
            "2022-01-01). Never after the message's own date. Null when unstated."
        ),
    )

    detail_field: DetailField | None = Field(
        default=None,
        description="For contact_detail only: which plain fact the message states.",
    )
    detail_value: str | None = Field(
        default=None,
        description=(
            "For contact_detail only: the value exactly as the message gives it. "
            "An ISO date for a birthday or anniversary, using the first of the "
            "month when only a month is stated; a price range as written."
        ),
    )

    # One field rather than three, and not for tidiness: the schema compiles
    # into a grammar and had outgrown what the API would accept. A fact that
    # needs three related values pays for one.
    transaction: str | None = Field(
        default=None,
        description=(
            "For a direct engagement: what the message says the transaction was, "
            "on one line, in this order and only the parts stated — bought or "
            "sold, the property, and what it closed at: 'bought 118 Sycamore Ave "
            "for $362,500', 'sold 12 Candlewick Ct.'. A past price, never what "
            "they are looking to spend next."
        ),
    )
    confidence: Confidence = Field(
        description=(
            "How plainly this transcript states the fact. high — one message "
            "says it outright about this contact, and reading it back to them "
            "would sound like quoting them. medium — stated, but you had to join "
            "two messages, or it was said about someone else in their household. "
            "low — implied by tone or circumstance, never actually said. High is "
            "acted on without asking, so use it only when the quote carries the "
            "fact on its own."
        )
    )


class ContactAnalysis(BaseModel):
    """The model's complete read on one contact."""

    model_config = Strict

    conversation_state: ConversationState
    state_evidence: str = Field(
        description="A verbatim quote supporting the conversation_state."
    )
    state_source_message_id: str | None = None
    facts: list[Fact]
    notes: str | None = Field(
        default=None,
        description="At most two sentences of context that is not a fact. Null if none.",
    )
