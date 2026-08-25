"""
audit/evaluation_framework.py
10-category evaluation framework for Kalvium B.Tech CSE demo calls.

Design principles:
  - Calibrated for Indian education sales context
  - KNET (Kalvium Entrance Test) is the primary CTA — highest weight
  - Scoring is generous: 5 = average, not minimum. Only <4 for genuinely poor execution
  - English transcriptions: no language penalty, Indian phrases ("okay", "ji", "right") are normal
  - Formal + friendly feedback tone throughout
  - Weights sum to 100
"""
from __future__ import annotations
import re
from typing import Optional
from config.models import Utterance, StructuredEvent, Speaker


# ─────────────────────────────────────────────────────────────────────────────
# 10-Category Framework
# ─────────────────────────────────────────────────────────────────────────────

EVALUATION_CATEGORIES: list[dict] = [

    # ── 1. Opening & Rapport ─────────────────────────────────────────────────
    {
        "id": "rapport_building",
        "label": "Opening & Rapport",
        "weight": 9,
        "gpt_prompt": (
            "Evaluate how warmly and naturally the counsellor opened the call and built rapport. "
            "Context: Indian education sales calls. Warm greetings, use of 'sir/ma'am', asking "
            "about the student's background, school, city, or family are all positive signals. "
            "Small talk about boards, exams, or aspirations before the pitch is healthy. "
            "Agreement words like 'okay', 'right', 'sure', 'I see' are natural in Indian English "
            "and should NOT be penalised — they signal active listening. "
            "Score generously: a counsellor who greets warmly and shows genuine interest in the "
            "student deserves at least 6. Only score below 4 if the counsellor was cold, "
            "rushed straight into the pitch, or made the student/parent feel uncomfortable."
        ),
        "rule_signals": {
            "positive": [
                r"\bhow are you\b", r"\bnice to (meet|speak|talk)\b",
                r"\bwelcome\b", r"\bhappy to (help|speak|connect)\b",
                r"\bthank you for (joining|calling|your time)\b",
                r"\bwhere.*from\b", r"\bwhich.*school\b", r"\bwhich.*city\b",
                r"\bwhat.*stream\b", r"\bwhat.*interest\b",
                r"\bma'?am\b", r"\bsir\b",
            ],
            "negative": [
                r"\blet me.*start.*pitch\b", r"\bI'll.*brief.*quickly\b",
            ],
        },
        "evidence_keys": ["warm_greeting", "name_usage", "personal_questions", "sir_maam_usage"],
        "coaching_templates": {
            (0, 4): "The opening felt rushed or cold. Begin every call with a warm greeting and one personal question — ask which school or city they're from, or how their exam preparation is going. A relaxed student is a receptive student.",
            (4, 6): "The opening was decent. Strengthen it by spending just 2-3 minutes on a genuine personal question before any pitch. 'What made you curious about Kalvium?' is a great bridge from rapport into discovery.",
            (6, 8): "Good rapport was built. The student/parent felt comfortable early on. For even stronger results, reference something personal later in the call — it shows you were truly listening.",
            (8, 10): "Excellent opening — warm, personal, and professional. The student felt at ease from the very first minute.",
        },
        "expected_behaviors": [
            "Greets warmly using sir/ma'am or by name",
            "Asks one personal question (city, school, stream, aspirations) before pitching",
            "Acknowledgement words (okay, right, sure, I see) used naturally",
            "Student/parent sounds comfortable and engaged within first 2-3 minutes",
        ],
    },

    # ── 2. Need Discovery ─────────────────────────────────────────────────────
    {
        "id": "discovery_questions",
        "label": "Need Discovery",
        "weight": 12,
        "gpt_prompt": (
            "Evaluate how well the counsellor understood the student's situation before pitching. "
            "Look for open-ended questions about: career goals, favourite subjects, dream companies, "
            "current academic performance, concerns about the future, family expectations, "
            "or what prompted them to attend this demo. "
            "Indian context: questions about marks, stream (PCM/PCB), city, and family background "
            "are highly relevant and should be rewarded. "
            "A counsellor who asked even 3-4 good discovery questions and listened to the answers "
            "deserves at least a 6. Only score below 4 if they pitched without asking anything at all. "
            "IMPORTANT: give credit if the counsellor asked questions at any point in the call, "
            "not just at the beginning."
        ),
        "rule_signals": {
            "positive": [
                r"\bwhat.*goal\b", r"\bwhat.*dream\b", r"\bwhat.*interest\b",
                r"\bwhat.*concern\b", r"\bwhere.*see yourself\b",
                r"\btell me.*about\b", r"\bwhich.*stream\b",
                r"\bwhat.*marks\b", r"\bwhat.*percentile\b",
                r"\bwhat.*company\b", r"\bwhat.*career\b",
                r"\bhow.*feel\b", r"\bwhat.*expect\b",
                r"\bwhat.*worry\b", r"\bany.*question\b",
            ],
            "negative": [
                r"\blet me.*explain.*everything\b",
            ],
        },
        "evidence_keys": ["career_goal_asked", "concern_asked", "background_explored"],
        "coaching_templates": {
            (0, 4): "The call moved straight into a pitch without understanding the student's situation. Try starting with: 'Before I tell you about Kalvium, can you tell me a bit about yourself — which stream are you in, and what are you hoping to do after 12th?' This one question changes the entire conversation.",
            (4, 6): "Some discovery happened but it could go deeper. Follow up on the student's answers — if they mention 'I like coding', ask 'What kind of projects have you tried?' The answers will tell you exactly how to personalise the pitch.",
            (6, 8): "Good discovery. You understood the student's background and goals. For even better results, ask one question about their biggest uncertainty or fear — that concern becomes the most important thing to address in your pitch.",
            (8, 10): "Thorough and natural discovery. The student felt heard before the pitch even began — this is the foundation of a great demo call.",
        },
        "expected_behaviors": [
            "Asks 3+ open-ended questions about the student's background or goals",
            "Listens to answers and follows up with related questions",
            "Understands the student's current situation before pitching",
            "Discovery happens naturally — not as a rigid checklist",
        ],
    },

    # ── 3. Kalvium Program Pitch ──────────────────────────────────────────────
    {
        "id": "product_explanation",
        "label": "Kalvium Program Pitch",
        "weight": 17,
        "gpt_prompt": (
            "Evaluate how clearly the counsellor explained the Kalvium B.Tech CSE program. "
            "Key elements to check (give credit if at least 3 of these 5 are covered): "
            "(1) 4-year residential on-campus B.Tech CSE — AICTE/UGC approved degree; "
            "(2) 80% practical / 20% theory model — the flip from traditional colleges; "
            "(3) The 4 Pillars: Love for CS, Discipline & Ethics, Autonomous Learning, Real-World Readiness; "
            "(4) DOJO — the coding platform / learning system with industry-led content; "
            "(5) Real work experience — students work at real companies, not just internships. "
            "SCORING GUIDE: "
            "All 5 elements explained clearly = 9-10. "
            "3-4 elements explained with examples = 7-8. "
            "1-2 elements explained adequately = 5-6. "
            "Vague explanation without any specifics = 3-4. "
            "No product explanation at all = 0-2. "
            "Be generous if the counsellor explained the core concept (practical over theory) "
            "even without naming every pillar."
        ),
        "rule_signals": {
            "positive": [
                r"\b80.?%.*practical\b", r"\b20.?%.*theor\b", r"\bpractical.*first\b",
                r"\b4.*pillar\b", r"\bfour.*pillar\b", r"\blove.*computer.*science\b",
                r"\bdiscipline.*ethics\b", r"\bautonomous.*learn\b",
                r"\breal.world.*readiness\b", r"\bdojo\b",
                r"\bwork.*integrat\b", r"\breal.*compan\b",
                r"\bresidential\b", r"\bon.campus\b",
                r"\baicte\b", r"\bugc\b", r"\bapproved\b",
                r"\bhands.on\b", r"\bproject.based\b",
                r"\bnot.*spoon.fed\b", r"\bindustry.*mentor\b",
            ],
            "negative": [
                r"\byou can check.*website\b", r"\bjust.*good.*college\b",
            ],
        },
        "evidence_keys": ["pillars_explained", "80_20_model_explained", "dojo_mentioned", "degree_credentials"],
        "coaching_templates": {
            (0, 4): "The student left without understanding what Kalvium actually is. Lead with the core concept: 'Traditional college is 80% theory. Kalvium flips it — 80% practical. You build real products, work with real companies, and earn a UGC-recognised B.Tech.' That one sentence is the hook.",
            (4, 6): "The basics were covered but the key differentiators were missing. Walk through the 4 pillars — even briefly. Each pillar takes 30 seconds to explain and together they create a compelling picture of life at Kalvium.",
            (6, 8): "Good program explanation. To strengthen it, connect each element to what the student said they want. 'Since you want to work in product companies, let me show you how the Real-World Readiness pillar directly prepares you for that.'",
            (8, 10): "The program was explained clearly and compellingly. The student now has a vivid picture of what Kalvium is and why it is different.",
        },
        "expected_behaviors": [
            "Explains the 80% practical / 20% theory model clearly",
            "Describes at least 2 of the 4 pillars with examples",
            "Mentions the degree is UGC/AICTE approved",
            "Gives a concrete example of the kind of work students do",
        ],
    },

    # ── 4. KNET Pitch & CTA ───────────────────────────────────────────────────
    {
        "id": "closing_skills",
        "label": "KNET Pitch & CTA",
        "weight": 25,
        "gpt_prompt": (
            "KNET (Kalvium National Entrance Test) is the primary Call-To-Action of this demo. "
            "Evaluate how well the counsellor explained KNET and motivated the student to register. "
            "WHAT TO CHECK: "
            "(1) Did they explain what KNET is — an entrance assessment for Kalvium admission? "
            "(2) Did they clarify KNET difficulty — it is NOT like JEE; it is an aptitude + "
            "psychometric test that tests curiosity and problem-solving, not rote learning; "
            "(3) Did they walk through the admission process steps: "
            "Sign up → KNET test → In-person interview → University visit? "
            "(4) Did they create urgency — limited seats, current batch deadlines? "
            "(5) Did they make a clear ask — 'Shall I send you the KNET registration link right now?' "
            "SCORING GUIDE: "
            "All 5 done well = 9-10. "
            "KNET explained + process explained + ask made = 7-8. "
            "KNET mentioned + some steps explained = 5-6. "
            "KNET briefly mentioned but no clear CTA = 3-4. "
            "KNET not mentioned at all = 0-2 (this is critical — no CTA = no conversion). "
            "Give credit if the counsellor made any attempt to close with KNET registration, "
            "even if not perfectly executed."
        ),
        "rule_signals": {
            "positive": [
                r"\bknet\b", r"\bkalvium.*entrance\b", r"\bentrance.*test\b",
                r"\badmission.*process\b", r"\bstep.*1\b", r"\bfirst.*step\b",
                r"\bregister\b", r"\bregistration\b", r"\bapply\b", r"\bapplication\b",
                r"\blink.*send\b", r"\bsend.*link\b",
                r"\bpsychometric\b", r"\baptitude.*test\b",
                r"\bin.person.*interview\b", r"\binterview\b",
                r"\buniversity.*visit\b", r"\bvisit.*campus\b",
                r"\bnext.*step\b", r"\bmove.*forward\b",
                r"\blimited.*seat\b", r"\bdeadline\b", r"\bbatch.*clos\b",
                r"\bnot.*jee\b", r"\bno.*jee\b", r"\beasier.*jee\b",
                r"\bnot.*like.*board\b", r"\bnot.*about.*marks\b",
            ],
            "negative": [
                r"\bthink.*about.*it\b.*bye\b", r"\bwhenever.*ready\b",
                r"\bif.*interested.*call\b", r"\bno.*hurry\b",
            ],
        },
        "evidence_keys": ["knet_explained", "admission_steps_given", "knet_difficulty_clarified", "closing_ask_made", "urgency_created"],
        "coaching_templates": {
            (0, 4): "KNET was not introduced — this is a critical gap. Every demo must end with KNET. Try: 'The next step is the KNET test — it is not like JEE at all. It tests your curiosity and problem-solving ability, not your marks. I can send you the registration link right now. Would that be okay?' This one ask converts demos into admissions.",
            (4, 6): "KNET was mentioned but the explanation and the ask were not strong enough. Spend 2-3 minutes specifically on: what KNET tests (not marks, but thinking), the 4 steps to admission, and end with a direct offer to send the link. Make it feel easy, not like an exam.",
            (6, 8): "KNET was explained and a closing ask was made. Strengthen the urgency: 'Seats for this batch are limited and are filling up. If you register for KNET this week, I will personally help you prepare.' A personal offer makes the urgency feel real, not pressured.",
            (8, 10): "Excellent KNET pitch — clearly explained, not intimidating, steps were clear, and a confident closing ask was made. This is the kind of CTA that drives conversions.",
        },
        "expected_behaviors": [
            "Explains KNET clearly — what it is and what it tests",
            "Clarifies KNET is NOT JEE-level — it is about thinking, not marks",
            "Walks through the 4 admission steps: Sign up → KNET → Interview → University Visit",
            "Makes a direct ask: 'Shall I send you the registration link?'",
            "Creates appropriate urgency around seat availability or batch deadlines",
        ],
    },

    # ── 5. Placement & Career Credibility ─────────────────────────────────────
    {
        "id": "placement_credibility",
        "label": "Placement & Career Story",
        "weight": 10,
        "gpt_prompt": (
            "Evaluate how credibly the counsellor presented Kalvium's placement outcomes. "
            "Specific stats from the Kalvium deck (use these to check): "
            "82% of the first graduating batch are placed; salary range 10-34 LPA; "
            "13 students received PPOs (Pre-Placement Offers) in their 3rd year itself; "
            "real student examples: Aayush Arora (2nd year, Indian unicorn) and Navaneeth Arunkumar (3rd year). "
            "IMPORTANT: Did the counsellor honestly state that Kalvium does NOT guarantee 100% placement? "
            "This honesty actually builds trust, not reduces it. "
            "SCORING GUIDE: "
            "Specific stats + real names + honest no-guarantee = 8-10. "
            "Some stats given with context = 6-7. "
            "Vague positive statements ('good placements') without data = 4-5. "
            "False 100% guarantee claim = cap at 4 (misleading students). "
            "No placement discussion at all = 2-3. "
            "Give credit if the counsellor shared even one specific data point."
        ),
        "rule_signals": {
            "positive": [
                r"\b82.?%\b", r"\b10.*lpa\b", r"\b34.*lpa\b",
                r"\bppo\b", r"\bpre.?placement.*offer\b",
                r"\baayush\b", r"\bnavaneeth\b",
                r"\bno.*guarantee\b", r"\bno.*free.*lunch\b",
                r"\breal.*placed\b", r"\bactually.*placed\b",
                r"\bfortune.*500\b", r"\bunicorn\b",
            ],
            "negative": [
                r"\b100.?%.*guarantee\b", r"\bguaranteed.*job\b",
                r"\bguaranteed.*placement\b",
            ],
        },
        "evidence_keys": ["placement_stats_quoted", "salary_range_given", "real_student_named", "no_guarantee_stated"],
        "coaching_templates": {
            (0, 4): "No placement data was shared. Use the exact numbers from the deck: '82% of our first graduating batch is placed, with salaries ranging from 10 to 34 LPA. 13 students got Pre-Placement Offers in their 3rd year itself.' Then add: 'We do not promise 100% placement — that depends on the student's effort. But here is what happens when students put in the work.'",
            (4, 6): "Placement was mentioned but without enough specifics. Students and parents want numbers and names. Quote 82%, mention the salary range, and name Aayush or Navaneeth. Real names make the outcomes feel real, not theoretical.",
            (6, 8): "Good placement coverage. Add the 'no 100% guarantee' statement — it might feel risky but it actually builds enormous trust. Parents appreciate honesty. Then follow it with the real stats to show what students who do put in the effort achieve.",
            (8, 10): "Excellent placement discussion — specific stats, real student examples, and honest about what it takes. This builds exactly the right kind of trust.",
        },
        "expected_behaviors": [
            "Quotes the 82% placement stat for the first batch",
            "Mentions the salary range (10-34 LPA) or specific examples",
            "Names a real student — Aayush or Navaneeth — with their company context",
            "Honestly states Kalvium does NOT guarantee 100% placement",
        ],
    },

    # ── 6. Fee & ROI Discussion ───────────────────────────────────────────────
    {
        "id": "fee_discussion",
        "label": "Fee & ROI Discussion",
        "weight": 9,
        "gpt_prompt": (
            "Evaluate how the counsellor handled the fee discussion. "
            "Key things to look for: "
            "(1) Was the fee clearly stated — not hidden or vague? "
            "(2) Were EMI or payment plan options mentioned? "
            "(3) Was the fee framed as an investment with a return (salary outcome)? "
            "(4) Were scholarships or financial aid options mentioned? "
            "(5) Was the fee compared to alternative colleges to contextualise the value? "
            "Indian context: Fee is almost always the biggest concern in education sales. "
            "A counsellor who brings it up proactively and explains payment options deserves credit. "
            "SCORING: "
            "Fee explained + EMI mentioned + ROI framed = 8-10. "
            "Fee explained with at least one supporting context = 6-7. "
            "Fee mentioned but no EMI/ROI context = 4-5. "
            "Fee avoided or treated as uncomfortable = 2-3. "
            "If the student/parent did not ask about fees and the counsellor did not raise it, "
            "score at 5 (opportunity missed but not critical if call was short)."
        ),
        "rule_signals": {
            "positive": [
                r"\bfee.*structure\b", r"\btotal.*fee\b", r"\bfee.*is\b",
                r"\bemi\b", r"\binstalment\b", r"\bmonthly.*payment\b",
                r"\bscholarship\b", r"\bfinancial.*aid\b", r"\bloan.*option\b",
                r"\breturn.*investment\b", r"\broi\b", r"\bsalary.*lakh\b",
                r"\bpayback\b", r"\binvestment.*not.*expense\b",
            ],
            "negative": [
                r"\bdon't.*worry.*fee\b", r"\bfee.*not.*issue\b",
                r"\bwe.*figure.*out\b",
            ],
        },
        "evidence_keys": ["fee_stated_clearly", "emi_mentioned", "roi_explained", "scholarship_mentioned"],
        "coaching_templates": {
            (0, 4): "The fee was not discussed clearly. Bring it up proactively — parents respect transparency. State the total fee, then immediately show the EMI option and the ROI: 'The total fee is X. Via our EMI plan, that comes to Y per month. Our placed students earn an average of Z — the fee pays for itself in about 8-10 months of working.'",
            (4, 6): "You mentioned the fee but did not provide enough context. Always pair the fee with the EMI figure and the salary outcome. 'The monthly EMI is roughly the cost of a family mobile plan — and it pays back within the first year of working' is a powerful reframe.",
            (6, 8): "Good fee discussion. Add the comparison angle: 'A traditional private engineering college costs 8-12 lakhs with zero work experience. Kalvium costs X, and you graduate with 2 years of real work already on your resume.' Value comparison changes how the fee feels.",
            (8, 10): "The fee was handled confidently and transparently — stated clearly, contextualised with EMI and ROI, and framed as an investment. This is how trust is built around pricing.",
        },
        "expected_behaviors": [
            "States the fee clearly and without hesitation",
            "Mentions EMI or monthly payment plan options",
            "Frames the fee as an investment with a concrete return",
            "Addresses scholarship or financial support options if relevant",
        ],
    },

    # ── 8. Engagement & Listening ─────────────────────────────────────────────
    {
        "id": "two_way_communication",
        "label": "Engagement & Listening",
        "weight": 9,
        "gpt_prompt": (
            "Evaluate whether the call was a genuine two-way conversation or a one-sided pitch. "
            "Look for: the counsellor pausing to let the student/parent respond, "
            "asking 'does that make sense?' or 'any questions on that?', "
            "referencing something the student said earlier to show they were listening, "
            "adjusting the pitch based on what was shared during discovery. "
            "Indian context: counsellors often speak more than the student — "
            "this is normal and does not always indicate poor listening. "
            "Give credit for any signs of genuine listening and adaptation. "
            "SCORING: "
            "Natural dialogue, student/parent actively participating = 8-10. "
            "Counsellor-led but with regular check-ins = 6-7. "
            "Mostly one-way but some engagement = 4-5. "
            "Pure monologue with no space for the student = 2-3. "
            "Note: if the student was quiet or shy, do not penalise the counsellor harshly."
        ),
        "rule_signals": {
            "positive": [
                r"\bdoes.*make sense\b", r"\bany.*question\b", r"\bwhat.*think\b",
                r"\bhow.*sound\b", r"\byou.*mention\b", r"\bas.*you.*said\b",
                r"\bbased.*on.*what.*you\b", r"\bgo ahead\b", r"\btell me\b",
                r"\bI hear you\b", r"\bI understand\b",
            ],
            "negative": [
                r"\blet me.*continue\b", r"\bas I was saying\b",
            ],
        },
        "evidence_keys": ["check_ins_done", "student_references", "student_questions", "listening_signals"],
        "coaching_templates": {
            (0, 4): "The call was more of a presentation than a conversation. After every major point, pause and check in: 'Does that make sense?' or 'What do you think about this?' These simple questions invite the student into the conversation and signal that their response matters.",
            (4, 6): "The call had some back-and-forth but could flow more naturally. When the student shares something, acknowledge it before moving on — even a 'That is a great perspective, and here is how Kalvium connects to that' makes a big difference.",
            (6, 8): "Good conversational balance. To elevate further, try referencing something the student said earlier in the call when making a key point — 'Since you mentioned wanting to work in tech startups, let me show you something specific to that.' It shows you truly listened.",
            (8, 10): "Excellent conversational quality — natural, balanced, and responsive. The student felt like an active participant, not just an audience.",
        },
        "expected_behaviors": [
            "Checks in regularly — 'Does that make sense?' or 'Any questions on that?'",
            "References something the student said earlier in the call",
            "Adjusts explanation based on the student's level of understanding",
            "Student/parent participates voluntarily — asks questions or shares views",
        ],
    },

    # ── 9. Trust & Institutional Credibility ──────────────────────────────────
    {
        "id": "trust_building",
        "label": "Trust & Credibility",
        "weight": 5,
        "gpt_prompt": (
            "Evaluate how well the counsellor established Kalvium's credibility as an institution. "
            "Look for: AICTE/UGC recognition mentioned, real student success stories, "
            "specific numbers (2,228 students across India, 82% placement), "
            "any honest acknowledgement of what Kalvium does not guarantee (transparency builds trust), "
            "partner university mentioned, Kalvium's track record of a few years explained. "
            "In Indian education, parents in particular need institutional credibility signals. "
            "SCORING: "
            "Multiple credibility signals including honesty about limitations = 8-10. "
            "At least 2 credibility signals present = 6-7. "
            "At least 1 signal present = 4-5. "
            "No credibility signals at all = 2-3."
        ),
        "rule_signals": {
            "positive": [
                r"\baicte\b", r"\bugc\b", r"\bapproved\b", r"\brecognised\b",
                r"\baccredited\b", r"\b2[,\s]?228\b", r"\bstudents.*india\b",
                r"\bno.*guarantee\b", r"\bno.*free.*lunch\b", r"\bhonest\b",
                r"\bour.*track.*record\b", r"\bproof\b", r"\bdata\b",
                r"\breal.*number\b", r"\bpartner.*universit\b",
            ],
            "negative": [
                r"\btrust us\b", r"\bjust.*believe\b",
                r"\b100.?%.*guarantee\b",
            ],
        },
        "evidence_keys": ["aicte_ugc_cited", "student_count_cited", "partner_uni_mentioned", "honest_limitations"],
        "coaching_templates": {
            (0, 4): "The student has no reason to trust Kalvium as an institution. Establish credibility early: 'Kalvium is in collaboration with UGC and AICTE-approved universities. We currently have 2,228 students across India. And unlike many institutions, we do not claim 100% placement — we show you what happens when students put in the work.' Credentials plus honesty is the most powerful trust combination.",
            (4, 6): "Some credibility was established but it needs more substance. Add the student count (2,228), name the UGC/AICTE approval, and show one real outcome — a specific student's story. Numbers and names make institutions feel real.",
            (6, 8): "Good credibility building. The student/parent likely felt confident about Kalvium's legitimacy. To strengthen further, mention the university partnership and the specific academic credentials behind the degree.",
            (8, 10): "Excellent credibility established — institutional credentials, real student outcomes, honest about limitations. Parents would feel comfortable proceeding based on this call.",
        },
        "expected_behaviors": [
            "Mentions UGC/AICTE approval explicitly",
            "Cites real numbers — student count, placement rate, or salary range",
            "Honestly states that Kalvium does not guarantee 100% placement",
            "Names a real student or shares a specific outcome as social proof",
        ],
    },

    # ── 10. Follow-up & Next Steps ────────────────────────────────────────────
    {
        "id": "follow_up_clarity",
        "label": "Follow-up & Next Steps",
        "weight": 4,
        "gpt_prompt": (
            "Evaluate how clearly the counsellor defined next steps at the end of the call. "
            "The primary next step should be KNET registration. "
            "Look for: a specific action being defined ('I will send you the KNET registration link'), "
            "a timeline given ('I will send it by today evening'), "
            "the counsellor's contact shared for follow-up questions, "
            "a follow-up call or message planned ('I will check in with you on Thursday'). "
            "SCORING: "
            "Specific action + timeline + contact shared = 8-10. "
            "Clear next step with KNET link mentioned = 6-7. "
            "Vague follow-up without specifics = 4-5. "
            "No follow-up defined at all = 2-3. "
            "Be generous if the counsellor offered to send information — that is a positive signal."
        ),
        "rule_signals": {
            "positive": [
                r"\bsend.*link\b", r"\blink.*send\b", r"\bknet.*link\b",
                r"\bregistration.*link\b", r"\bapplication.*link\b",
                r"\bI.*follow.*up\b", r"\bwhatsapp.*you\b", r"\bI'll.*message\b",
                r"\bby.*today\b", r"\bby.*tonight\b", r"\bby.*tomorrow\b",
                r"\bnext.*call\b", r"\bschedule.*call\b", r"\bmy.*number\b",
                r"\bmy.*contact\b", r"\beach.*out\b", r"\bI.*reach\b",
            ],
            "negative": [
                r"\bwhenever.*ready.*call\b", r"\bif.*interested\b",
                r"\bwe.*see\b", r"\bno.*hurry\b.*bye\b",
            ],
        },
        "evidence_keys": ["knet_link_offered", "follow_up_timeline_given", "contact_shared"],
        "coaching_templates": {
            (0, 4): "The call ended without a clear next step. Always close with: 'I am going to send you the KNET registration link on WhatsApp right now. Take a look, and let us speak again on [specific day] so I can answer any questions before you register.' A clear, bilateral next step keeps the momentum alive.",
            (4, 6): "A follow-up was mentioned but it was not specific enough. Replace 'I will send you some information' with 'I am sending you the KNET link right now — it takes 5 minutes to register.' Specific offers get specific responses.",
            (6, 8): "Good follow-up intent. Make it even clearer by giving a specific time: 'I will WhatsApp you by 6 pm today' is far more likely to result in action than 'I will follow up soon'.",
            (8, 10): "Clear, specific, and bilateral next steps were defined. Both the counsellor and the student know exactly what happens next and when — this is what keeps conversions from going cold.",
        },
        "expected_behaviors": [
            "Offers to send the KNET registration link immediately",
            "Gives a specific time for follow-up: 'I will message you by this evening'",
            "Shares their contact number or WhatsApp for easy reach",
            "Student/parent agrees to a concrete next action before the call ends",
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Inject expected_behaviors and build lookup maps
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_BY_ID: dict[str, dict] = {cat["id"]: cat for cat in EVALUATION_CATEGORIES}
CATEGORY_WEIGHTS: dict[str, float] = {cat["id"]: cat["weight"] for cat in EVALUATION_CATEGORIES}

assert sum(CATEGORY_WEIGHTS.values()) == 100, (
    f"Weights sum to {sum(CATEGORY_WEIGHTS.values())}, must be 100"
)


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based pre-scorer
# ─────────────────────────────────────────────────────────────────────────────

def rule_score_category(category_id: str, utterances: list[Utterance]) -> tuple[float, list[str]]:
    """
    Returns a rule-based baseline score (0-10) and evidence list.
    Calibrated generously — used as a soft anchor for GPT, not a hard gate.
    """
    cat = CATEGORY_BY_ID.get(category_id)
    if not cat or "rule_signals" not in cat:
        return 5.0, []

    counsellor_text = " ".join(
        u.english_text for u in utterances if u.speaker == Speaker.COUNSELLOR
    ).lower()

    pos_hits = sum(
        1 for p in cat["rule_signals"].get("positive", [])
        if re.search(p, counsellor_text, re.IGNORECASE)
    )
    neg_hits = sum(
        1 for p in cat["rule_signals"].get("negative", [])
        if re.search(p, counsellor_text, re.IGNORECASE)
    )

    pos_max = len(cat["rule_signals"].get("positive", [])) or 1
    # Generous base: 0 hits = 4.0 (not 0), scales up from there
    base = 4.0 + min(6.0, (pos_hits / pos_max) * 8.0) - (neg_hits * 0.5)
    score = round(max(0.0, min(10.0, base)), 1)

    evidence = [
        f"Signal present: '{p}'" for p in cat["rule_signals"].get("positive", [])
        if re.search(p, counsellor_text, re.IGNORECASE)
    ][:3]

    return score, evidence


def get_coaching(category_id: str, score: float) -> str:
    """Return the appropriate coaching template for a given category and score."""
    cat = CATEGORY_BY_ID.get(category_id)
    if not cat:
        return ""
    for (lo, hi), text in cat.get("coaching_templates", {}).items():
        if lo <= score < hi:
            return text
    return ""
