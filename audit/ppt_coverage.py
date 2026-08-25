"""
audit/ppt_coverage.py
Checks whether a counsellor covered the key sections of the
Kalvium B.Tech CSE presentation (160-slide AY26 deck).

Each PPT_SECTION defines:
  - id          : machine key
  - label       : display name
  - slides      : which slides cover it
  - priority    : "must" | "should" | "can"
  - keywords    : regex patterns to detect in English transcript
  - description : what covering this section means
  - how_to_explain : ideal way to explain this section
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Literal

from config.models import Utterance, Speaker


# ─────────────────────────────────────────────────────────────────────────────
# PPT Section definitions
# ─────────────────────────────────────────────────────────────────────────────

PPT_SECTIONS: list[dict] = [

    # ── MUST COVER ────────────────────────────────────────────────────────────

    {
        "id": "what_is_kalvium",
        "label": "What is Kalvium?",
        "slides": "1–4",
        "priority": "must",
        "keywords": [
            r"\bkalvium\b", r"\bb\.?tech\b.*\bcse\b", r"\b4.year\b.*\bdegree\b",
            r"\bresidential\b", r"\bon.campus\b", r"\bUGC\b", r"\bAICTE\b",
            r"\b2[,\s]*228\s*students\b", r"\brecognized.*universit",
            r"\bpartner.*universit", r"\bsoftware.*product.*engineering\b",
        ],
        "description": "Counsellor explained what Kalvium is: 4-year residential B.Tech CSE, AICTE/UGC recognised, 2,228+ students.",
        "how_to_explain": (
            "Start with one sentence: 'Kalvium is a 4-year, residential, on-campus B.Tech CSE programme "
            "offered in collaboration with UGC and AICTE approved universities across India.' "
            "Then show the student count (2,228+) as social proof. "
            "Name the specific university you're discussing for the student's location."
        ),
    },

    {
        "id": "practical_vs_theory",
        "label": "80/20 Practical vs Theory",
        "slides": "5",
        "priority": "must",
        "keywords": [
            r"\b80.?%.*practical\b", r"\b20.?%.*theor\b",
            r"\bpractical.*vs.*theor\b", r"\breal.world.*learning\b",
            r"\btheor.*80\b", r"\bpractical.*first\b",
            r"\btraditional.*college.*20\b", r"\bunlike.*regular.*college\b",
            r"\bindustry.*ready\b", r"\bhands.on\b",
        ],
        "description": "Counsellor explained the 80% practical / 20% theory inversion vs traditional 80% theory / 20% practical.",
        "how_to_explain": (
            "Use the contrast: 'In a regular engineering college, 80% is theory and 20% practical. "
            "Kalvium flips this — 80% practical, 20% theory. "
            "You graduate with 2 years of real project experience baked into your degree, not just a certificate.'"
        ),
    },

    {
        "id": "four_pillars",
        "label": "The 4 Pillars of Kalvium",
        "slides": "8",
        "priority": "must",
        "keywords": [
            r"\blove.*computer.*science\b", r"\bdiscipline.*ethics\b",
            r"\bautonomous.*learning\b", r"\breal.world.*readiness\b",
            r"\b4.*pillar\b", r"\bfour.*pillar\b",
            r"\bpillar.*kalvium\b", r"\bhow.*kalvium.*work\b",
        ],
        "description": "Counsellor introduced all 4 pillars: Love for CS, Discipline & Ethics, Autonomous Learning, Real-World Readiness.",
        "how_to_explain": (
            "Introduce the framework: 'Kalvium is built on 4 pillars that work together. "
            "1) Love for CS — deep coding skills from Day 1. "
            "2) Discipline & Ethics — professional-grade habits. "
            "3) Autonomous Learning — you own your learning, not wait to be taught. "
            "4) Real-World Readiness — simulated work, FOSS, internships from semester 1.' "
            "Use the PPT slide to visually anchor each pillar."
        ),
    },

    {
        "id": "placement_stats",
        "label": "Placement Stats (82%, 10–34 LPA, PPOs)",
        "slides": "57–63",
        "priority": "must",
        "keywords": [
            r"\b82.?%\b.*plac", r"\bplac.*82.?%\b",
            r"\b10.*lpa\b", r"\b34.*lpa\b", r"\b10.*34.*lpa\b",
            r"\bsalary.*range\b", r"\bppo\b", r"\bpre.placement.*offer\b",
            r"\b13.*ppo\b", r"\b3rd.*year.*placed\b",
            r"\bfortune.*500\b", r"\binternship.*stat\b",
        ],
        "description": "Counsellor quoted specific placement stats: 82% placed, 10–34 LPA range, 13 PPOs in 3rd year.",
        "how_to_explain": (
            "Lead with specifics: 'Our 2022 batch — the first to graduate — has 82% placed so far "
            "with 6 months still to go. Salary range: 10 LPA to 34 LPA. "
            "13 students got PPOs (pre-placement offers) in their 3rd year itself. "
            "These are real numbers from our first batch, not projections.'"
        ),
    },

    {
        "id": "no_100_percent_guarantee",
        "label": "Honest: No 100% Placement Guarantee",
        "slides": "55–56",
        "priority": "must",
        "keywords": [
            r"\bno.*guarantee\b.*plac", r"\bnot.*guarantee\b.*plac",
            r"\bno.*free.*lunch\b", r"\bgurarantee.*100\b",
            r"\bkalvium.*does.*not.*guarantee\b",
            r"\bdepend.*on.*student\b", r"\byou.*have.*to.*work\b",
            r"\bhonest.*plac\b",
        ],
        "description": "Counsellor was honest that Kalvium does NOT guarantee 100% placement — depends on student effort.",
        "how_to_explain": (
            "Be upfront: 'I want to be honest — Kalvium does not guarantee 100% placement. "
            "No legitimate college does. What we guarantee is that if you put in the work, "
            "you will be more employable than 90% of engineers graduating in India. "
            "There is no free lunch.' This builds trust and pre-empts the competition's hollow promises."
        ),
    },

    {
        "id": "admission_process",
        "label": "Admission Process & KNET Test",
        "slides": "65",
        "priority": "must",
        "keywords": [
            r"\bknet\b", r"\badmission.*process\b", r"\b4.*step\b",
            r"\bpsychometric.*test\b", r"\baptitude.*test\b",
            r"\badmission.*portal\b", r"\bin.person.*interview\b",
            r"\bknet.*score\b", r"\bstep.*1\b.*\bsign.*up\b",
            r"\binterview.*centre\b",
        ],
        "description": "Counsellor explained the 4-step admission process including the KNET test.",
        "how_to_explain": (
            "Walk them through the 4 steps: "
            "1) Sign up on the Kalvium admissions portal. "
            "2) Take the KNET — a psychometric + aptitude test (not JEE difficulty). "
            "3) Attend an in-person interview at a nearby centre. "
            "4) Visit your chosen university and complete admission. "
            "Reassure them the KNET is about potential, not just marks."
        ),
    },

    # ── SHOULD COVER ──────────────────────────────────────────────────────────

    {
        "id": "dojo_coding_platform",
        "label": "DOJO — Coding Platform",
        "slides": "17–20",
        "priority": "should",
        "keywords": [
            r"\bdojo\b", r"\bcoding.*platform\b", r"\blearn.*cod.*right.*way\b",
            r"\btech.*leader.*design\b", r"\blesson.*design.*top\b",
            r"\btextbook.*not.*enough\b", r"\biit.*madras.*textbook\b",
        ],
        "description": "Counsellor mentioned DOJO — the coding platform with lessons designed by top tech leaders.",
        "how_to_explain": (
            "Show the DOJO: 'We don't use generic textbooks. Our coding curriculum is designed "
            "by engineers from top tech companies. The DOJO platform gives you structured, "
            "progressive coding challenges — the same textbook used at IIT-Madras, but taught "
            "the way real engineers actually code.'"
        ),
    },

    {
        "id": "autonomous_learning_faq",
        "label": "Why Autonomous Learning? (FAQ #2)",
        "slides": "52",
        "priority": "should",
        "keywords": [
            r"\btake.*charge.*learn\b", r"\bown.*learning\b",
            r"\bnot.*spoon.fed\b", r"\bhow.*to.*learn\b",
            r"\bbeyond.*marks\b", r"\bautonomous.*tough\b",
            r"\bautonomous.*beginning\b", r"\blifelong.*confidence\b",
        ],
        "description": "Counsellor explained why autonomous learning (not spoon-feeding) builds lifelong confidence.",
        "how_to_explain": (
            "Answer the 'why': 'Autonomous learning means you take charge of what and how you learn — "
            "not wait to be told. It's tough in the first few months, yes. But by year 2, "
            "you know how to learn anything. That skill — knowing how to learn — is more valuable "
            "than any specific subject you study.'"
        ),
    },

    {
        "id": "mentor_quality",
        "label": "Who Are Kalvium Mentors? (FAQ #3)",
        "slides": "53",
        "priority": "should",
        "keywords": [
            r"\bmentor\b", r"\bfacult\b.*\bdifferent\b",
            r"\bkalvium.*train.*mentor\b", r"\bregular.*facult.*can.t\b",
            r"\bindustry.*mentor\b", r"\btrained.*kalvium\b",
        ],
        "description": "Counsellor explained the Kalvium mentor model — industry-trained, not regular faculty.",
        "how_to_explain": (
            "Be direct: 'Regular college faculty are great at teaching theory. "
            "But Kalvium's learning system is fundamentally different, "
            "so we train our own mentors extensively in this method. "
            "Every mentor you meet has been specifically trained by Kalvium — "
            "they guide, not lecture.'"
        ),
    },

    {
        "id": "university_vs_kalvium",
        "label": "University vs Kalvium Roles (FAQ #4)",
        "slides": "54",
        "priority": "should",
        "keywords": [
            r"\buniversity.*responsible\b", r"\bkalvium.*responsible\b",
            r"\binfrastructure.*university\b", r"\bdegree.*university\b",
            r"\bkalvium.*academic\b", r"\bkalvium.*training\b",
            r"\buniversity.*hostel\b", r"\buniversity.*rule\b",
        ],
        "description": "Counsellor clarified what Kalvium provides vs what the university provides.",
        "how_to_explain": (
            "Draw the line clearly: 'Kalvium is responsible for ALL academics, skilling, and training — "
            "everything you learn, we design and deliver. "
            "The university provides the infrastructure: hostel, campus, the degree, accreditation. "
            "Think of Kalvium as the brain, the university as the body.'"
        ),
    },

    {
        "id": "real_world_exposure",
        "label": "Simulated Work / FOSS / Internships",
        "slides": "46–49",
        "priority": "should",
        "keywords": [
            r"\bsimulated.*work\b", r"\bfoss\b", r"\bopen.source\b",
            r"\binternship.*curriculum\b", r"\bday.*1.*real\b",
            r"\bscrum\b", r"\bsprint\b", r"\bstand.up\b",
            r"\bsdlc\b", r"\b110.*student.*open.source\b",
            r"\baayush\b", r"\bnavaneeth\b", r"\breal.*project.*unicorn\b",
        ],
        "description": "Counsellor explained simulated work, FOSS contributions, and how internships are baked into curriculum.",
        "how_to_explain": (
            "Make it concrete: 'From semester 1, you do simulated work — real projects with daily stand-ups "
            "and weekly sprints, exactly like a software team. 110+ students are already contributing "
            "to open-source projects globally. By year 3, students like Aayush are building products "
            "at Indian unicorns. Internships aren't an add-on — they're in the curriculum.'"
        ),
    },

    {
        "id": "student_passion_projects",
        "label": "Student Passion Projects",
        "slides": "9–11",
        "priority": "should",
        "keywords": [
            r"\bpassion.*project\b", r"\bstudent.*built\b",
            r"\bstudent.*project\b", r"\bbuild.*real.*product\b",
            r"\bportfolio\b", r"\bgithub\b", r"\brepository\b",
        ],
        "description": "Counsellor showed student passion projects or portfolio as proof of quality.",
        "how_to_explain": (
            "Show, don't tell: 'Let me show you what a first-year Kalvium student built.' "
            "Pull up a GitHub repo or the passion project slide. "
            "'This was built by a student who joined with no prior coding experience. "
            "That's what Day 1 focus creates.'"
        ),
    },

    {
        "id": "discipline_culture",
        "label": "Discipline & Professional Culture",
        "slides": "22–31",
        "priority": "should",
        "keywords": [
            r"\bdiscipline\b.*\bexcellence\b", r"\bprofessional.*culture\b",
            r"\bconduct.*policy\b", r"\bmalpractice\b",
            r"\bdaily.*stand.up\b", r"\binternship.*selection.*round\b",
            r"\bintensity\b.*\bbuild\b", r"\breal.*tech.*team\b",
        ],
        "description": "Counsellor conveyed the discipline culture — daily stand-ups, professional conduct policy, consequences for malpractice.",
        "how_to_explain": (
            "Be real about expectations: 'Kalvium is not a relaxed college. "
            "We run daily stand-ups like real tech teams. We have a student conduct policy — "
            "2 students were banned from internship opportunities for malpractice. "
            "We prepare adults, not exam-takers. If your goal is 4 easy years, Kalvium is not for you. "
            "If your goal is career-ready from day 1, this is exactly the environment you need.'"
        ),
    },

    # ── CAN COVER (context-dependent) ────────────────────────────────────────

    {
        "id": "university_specific",
        "label": "Specific University Details",
        "slides": "76–159",
        "priority": "can",
        "keywords": [
            r"\balliance.*universit\b", r"\blpu\b", r"\bjecrc\b",
            r"\bsrm.*universit\b", r"\byenepoya\b", r"\brv.*universit\b",
            r"\bst.*joseph.*universit\b", r"\btakshashila\b",
            r"\bsgt.*universit\b", r"\bcape.*instit\b",
            r"\bkanyakumari\b", r"\bjaipur.*campus\b", r"\bpunjab.*campus\b",
            r"\bbengaluru.*campus\b", r"\bcampus.*acre\b",
            r"\bhostel.*fee\b", r"\bfee.*structure\b",
        ],
        "description": "Counsellor covered the specific university campus, infrastructure, hostel, fees, and rankings.",
        "how_to_explain": (
            "Personalise to the student's preferred location: "
            "'Let me show you [Alliance/LPU/JECRC/SRM/…]. It's a [X]-acre campus in [city]. "
            "Here's the fee structure and scholarship options based on your KNET score.' "
            "Show the campus slides only for the university the student is considering."
        ),
    },

    {
        "id": "scholarship_fee",
        "label": "Scholarships & Fee Structure",
        "slides": "94–95, per university",
        "priority": "can",
        "keywords": [
            r"\bscholarship\b", r"\b40[,\s]*000\b.*scholarship\b",
            r"\bknet.*score.*scholarship\b", r"\b12th.*board.*scholarship\b",
            r"\bjee.*scholarship\b", r"\bfee.*structure\b",
            r"\bemi\b", r"\bloan\b", r"\beducation.*loan\b",
        ],
        "description": "Counsellor explained ₹40,000 KNET scholarship, board/JEE scholarships, and fee + EMI options.",
        "how_to_explain": (
            "Frame as investment: 'There are scholarships available — up to ₹40,000 based on your KNET score, "
            "plus additional scholarships for your 12th board marks or JEE rank. "
            "The fee varies by university — let me walk you through the specific structure. "
            "Most students finance through education loans with EMIs that start after placement.'"
        ),
    },

    {
        "id": "entrepreneurship",
        "label": "Entrepreneurship Pathway",
        "slides": "46",
        "priority": "can",
        "keywords": [
            r"\bentrepreneur\b", r"\bstartup\b.*\bstudent\b",
            r"\bproduct.*market.*fit\b", r"\breal.*user\b",
            r"\bbuild.*test.*iterate\b", r"\bown.*product\b",
            r"\bmentor.*till.*product\b",
        ],
        "description": "Counsellor mentioned the entrepreneurship pathway — building real products with mentors.",
        "how_to_explain": (
            "For entrepreneurially minded students: 'If you want to build your own startup, "
            "Kalvium gives you the environment. You can build, test, and iterate a real product "
            "with real users — guided by mentors until you hit product-market fit. "
            "Several students are already building tech tools for global companies.'"
        ),
    },
]

# Quick lookup
PPT_SECTION_BY_ID = {s["id"]: s for s in PPT_SECTIONS}
MUST_COVER  = [s for s in PPT_SECTIONS if s["priority"] == "must"]
SHOULD_COVER = [s for s in PPT_SECTIONS if s["priority"] == "should"]
CAN_COVER   = [s for s in PPT_SECTIONS if s["priority"] == "can"]


# ─────────────────────────────────────────────────────────────────────────────
# Coverage checker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SectionCoverage:
    section_id: str
    label: str
    slides: str
    priority: str
    covered: bool
    matched_keywords: list[str] = field(default_factory=list)
    how_to_explain: str = ""
    description: str = ""


@dataclass
class PPTCoverageReport:
    must_covered: int
    must_total: int
    should_covered: int
    should_total: int
    can_covered: int
    can_total: int
    sections: list[SectionCoverage] = field(default_factory=list)
    coverage_score: float = 0.0   # 0–10: weighted by priority
    missed_must: list[str] = field(default_factory=list)
    missed_should: list[str] = field(default_factory=list)


def check_ppt_coverage(utterances: list[Utterance]) -> PPTCoverageReport:
    """
    Analyse transcript utterances to determine which PPT sections were covered.
    Only checks counsellor speech (not student/parent).
    """
    # Build counsellor transcript
    counsellor_text = " ".join(
        u.english_text for u in utterances
        if u.speaker == Speaker.COUNSELLOR
    ).lower()

    sections: list[SectionCoverage] = []

    for sec in PPT_SECTIONS:
        matched = []
        for kw in sec["keywords"]:
            if re.search(kw, counsellor_text, re.IGNORECASE):
                matched.append(kw)

        # A section is "covered" if at least 2 keywords match
        # (or 1 for short keyword lists)
        threshold = 1 if len(sec["keywords"]) <= 3 else 2
        covered = len(matched) >= threshold

        sections.append(SectionCoverage(
            section_id=sec["id"],
            label=sec["label"],
            slides=sec["slides"],
            priority=sec["priority"],
            covered=covered,
            matched_keywords=[kw[:40] for kw in matched[:3]],
            how_to_explain=sec["how_to_explain"],
            description=sec["description"],
        ))

    must_covered  = sum(1 for s in sections if s.priority == "must"   and s.covered)
    must_total    = len(MUST_COVER)
    should_covered = sum(1 for s in sections if s.priority == "should" and s.covered)
    should_total  = len(SHOULD_COVER)
    can_covered   = sum(1 for s in sections if s.priority == "can"    and s.covered)
    can_total     = len(CAN_COVER)

    # Weighted score: must=6pts each, should=2.5pts each, can=0.5pts each
    must_max    = must_total  * 6.0
    should_max  = should_total * 2.5
    can_max     = can_total   * 0.5
    total_max   = must_max + should_max + can_max

    raw_score = (
        must_covered   * 6.0 +
        should_covered * 2.5 +
        can_covered    * 0.5
    )
    coverage_score = round(min(10.0, (raw_score / total_max) * 10), 1) if total_max > 0 else 0.0

    missed_must   = [s.label for s in sections if s.priority == "must"   and not s.covered]
    missed_should = [s.label for s in sections if s.priority == "should" and not s.covered]

    return PPTCoverageReport(
        must_covered=must_covered,
        must_total=must_total,
        should_covered=should_covered,
        should_total=should_total,
        can_covered=can_covered,
        can_total=can_total,
        sections=sections,
        coverage_score=coverage_score,
        missed_must=missed_must,
        missed_should=missed_should,
    )


def ppt_coverage_to_dict(report: PPTCoverageReport) -> dict:
    """Serialise PPTCoverageReport to a plain dict for the report JSON."""
    return {
        "coverage_score": report.coverage_score,
        "must_covered": report.must_covered,
        "must_total": report.must_total,
        "should_covered": report.should_covered,
        "should_total": report.should_total,
        "can_covered": report.can_covered,
        "can_total": report.can_total,
        "missed_must": report.missed_must,
        "missed_should": report.missed_should,
        "sections": [
            {
                "section_id": s.section_id,
                "label": s.label,
                "slides": s.slides,
                "priority": s.priority,
                "covered": s.covered,
                "matched_keywords": s.matched_keywords,
                "how_to_explain": s.how_to_explain,
                "description": s.description,
            }
            for s in report.sections
        ],
    }
