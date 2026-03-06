"""BMAD-style agentic workflow engine for protocol assistant."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


TOKEN_RE = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ]+(?:\.[0-9a-zA-Zа-яА-ЯёЁ]+)?")
MKB_RE = re.compile(r"^[A-Za-z]\d{1,2}(?:\.\d{1,2})?$")

PEDIATRIC_MARKERS = {
    "ребенок",
    "ребёнок",
    "дети",
    "детский",
    "новорожденный",
    "новорождённый",
    "младенец",
}
ADULT_MARKERS = {"взрослый", "взрослых", "adult"}
EMERGENCY_MARKERS = {
    "потеря",
    "сознания",
    "анафилаксия",
    "анафилактический",
    "судороги",
    "удушье",
    "острая",
    "боль",
    "кровотечение",
}

SYMPTOM_RULES = [
    {
        "id": "cough",
        "pattern": re.compile(r"каш[её]л|кашл", re.I),
        "question": "Есть ли выраженный кашель (сухой или с мокротой)?",
    },
    {
        "id": "chronic_course",
        "pattern": re.compile(r"более\s+\d+\s*(дн|нед|месяц)|длительн|хронич", re.I),
        "question": "Симптомы длятся длительно (недели и более)?",
    },
    {
        "id": "sore_throat",
        "pattern": re.compile(r"(бол|першен).{0,20}горл|горл.{0,20}(бол|першен)", re.I),
        "question": "Есть ли боль или выраженное першение в горле?",
    },
    {
        "id": "fever",
        "pattern": re.compile(r"лихорад|температур|жар", re.I),
        "question": "Есть ли температура 38°C и выше?",
    },
    {
        "id": "chills",
        "pattern": re.compile(r"озноб", re.I),
        "question": "Есть ли выраженный озноб?",
    },
    {
        "id": "rhinitis",
        "pattern": re.compile(r"насморк|риноре|заложен.{0,10}нос", re.I),
        "question": "Есть ли насморк или заложенность носа?",
    },
    {
        "id": "dyspnea",
        "pattern": re.compile(r"одыш|затруднен.{0,20}дых", re.I),
        "question": "Есть ли одышка или ощущение нехватки воздуха?",
    },
    {
        "id": "wheeze",
        "pattern": re.compile(r"свист.{0,20}дых|хрип", re.I),
        "question": "Есть ли свистящее дыхание или хрипы?",
    },
    {
        "id": "sputum",
        "pattern": re.compile(r"мокрот", re.I),
        "question": "Есть ли мокрота (продуктивный кашель)?",
    },
    {
        "id": "chest_pain",
        "pattern": re.compile(r"бол.{0,20}груд|загрудин", re.I),
        "question": "Есть ли боль в грудной клетке?",
    },
    {
        "id": "abdominal_pain",
        "pattern": re.compile(r"бол.{0,20}живот|абдомин", re.I),
        "question": "Есть ли выраженная боль в животе?",
    },
    {
        "id": "ruq_pain",
        "pattern": re.compile(r"прав.{0,20}подребер", re.I),
        "question": "Боль локализуется в правом подреберье?",
    },
    {
        "id": "nausea",
        "pattern": re.compile(r"тошнот", re.I),
        "question": "Есть ли тошнота?",
    },
    {
        "id": "vomiting",
        "pattern": re.compile(r"рвот", re.I),
        "question": "Есть ли рвота?",
    },
    {
        "id": "diarrhea",
        "pattern": re.compile(r"диаре|жидк.{0,10}стул", re.I),
        "question": "Есть ли диарея (частый жидкий стул)?",
    },
    {
        "id": "jaundice",
        "pattern": re.compile(r"желтуш|иктер", re.I),
        "question": "Есть ли желтушность кожи или склер?",
    },
    {
        "id": "dysuria",
        "pattern": re.compile(r"дизур|мочеиспуск|цистит", re.I),
        "question": "Есть ли боль/жжение при мочеиспускании или учащенное мочеиспускание?",
    },
    {
        "id": "flank_pain",
        "pattern": re.compile(r"боль.{0,20}поясниц|поясниц.{0,20}бол|боль.{0,20}в\s+бок", re.I),
        "question": "Есть ли боль в пояснице или в боку (фланковая боль)?",
    },
    {
        "id": "rash",
        "pattern": re.compile(r"сып|высып", re.I),
        "question": "Есть ли кожная сыпь?",
    },
    {
        "id": "bleeding",
        "pattern": re.compile(r"кровотеч", re.I),
        "question": "Есть ли кровотечение или кровохарканье?",
    },
    {
        "id": "trauma",
        "pattern": re.compile(r"травм|ушиб|перелом", re.I),
        "question": "Была ли недавняя травма?",
    },
    {
        "id": "tonsillitis",
        "pattern": re.compile(r"тонзил|ангин|миндал", re.I),
        "question": "Есть ли налет/увеличение миндалин или признаки ангины?",
    },
    {
        "id": "tonsil_plaque",
        "pattern": re.compile(r"нал[её]т.{0,20}миндал|миндал.{0,20}нал[её]т|гнойн.{0,10}миндал", re.I),
        "question": "Есть ли налет на миндалинах?",
    },
    {
        "id": "lymph_nodes",
        "pattern": re.compile(r"лимфоузл|лимфаден", re.I),
        "question": "Есть ли увеличение лимфоузлов?",
    },
    {
        "id": "edema",
        "pattern": re.compile(r"отек|отеки|отёк|отёки", re.I),
        "question": "Есть ли выраженные отеки (ног, лица или генерализованные)?",
    },
    {
        "id": "neuro_focal",
        "pattern": re.compile(r"асимметр|нарушен.{0,15}реч|слабост.{0,15}(рук|ног)|онемен", re.I),
        "question": "Есть ли очаговые неврологические симптомы (слабость в конечности, нарушение речи, асимметрия лица)?",
    },
    {
        "id": "seizure",
        "pattern": re.compile(r"судорог|припад|конвульс", re.I),
        "question": "Были ли судороги или судорожный приступ?",
    },
    {
        "id": "syncope",
        "pattern": re.compile(r"потер.{0,10}созн|обморок", re.I),
        "question": "Была ли потеря сознания или обморок?",
    },
    {
        "id": "headache",
        "pattern": re.compile(r"головн.{0,10}бол", re.I),
        "question": "Есть ли интенсивная головная боль?",
    },
    {
        "id": "meningeal",
        "pattern": re.compile(r"ригидн.{0,12}шеи|светобоязн|менингеал", re.I),
        "question": "Есть ли признаки менингеального синдрома (ригидность шеи, светобоязнь)?",
    },
    {
        "id": "epigastric_pain",
        "pattern": re.compile(r"эпигастр|бол.{0,20}подложеч", re.I),
        "question": "Есть ли боль в эпигастрии (подложечной области)?",
    },
    {
        "id": "vaginal_discharge",
        "pattern": re.compile(r"выделен|полов.{0,10}акт|вагин|тазов", re.I),
        "question": "Есть ли патологические выделения или боли, связанные с половым актом?",
    },
    {
        "id": "menstrual_relation",
        "pattern": re.compile(r"менстру", re.I),
        "question": "Связаны ли симптомы с менструальным циклом?",
    },
    {
        "id": "tb_constitutional",
        "pattern": re.compile(r"похуд|ночн.{0,10}пот|кашл.{0,20}(месяц|недел)", re.I),
        "question": "Есть ли похудение, ночная потливость или длительный кашель более 3 недель?",
    },
    {
        "id": "palpitations",
        "pattern": re.compile(r"сердцеби|перебо.{0,8}серд|аритм", re.I),
        "question": "Есть ли сердцебиение или ощущение перебоев в работе сердца?",
    },
]

SYMPTOM_TO_DOMAINS: Dict[str, List[str]] = {
    "cough": ["respiratory"],
    "chronic_course": [],
    "sore_throat": ["respiratory"],
    "fever": [],
    "chills": ["infectious"],
    "rhinitis": ["respiratory"],
    "dyspnea": ["respiratory", "cardio"],
    "wheeze": ["respiratory"],
    "sputum": ["respiratory"],
    "chest_pain": ["respiratory", "cardio"],
    "abdominal_pain": ["gastro"],
    "ruq_pain": ["gastro"],
    "nausea": [],
    "vomiting": [],
    "diarrhea": ["gastro"],
    "jaundice": ["gastro"],
    "dysuria": ["urinary"],
    "flank_pain": ["urinary"],
    "rash": ["dermatology"],
    "bleeding": ["hematology"],
    "trauma": ["trauma"],
    "tonsillitis": ["respiratory"],
    "tonsil_plaque": ["respiratory", "infectious"],
    "lymph_nodes": ["infectious"],
    "edema": ["cardio"],
    "neuro_focal": ["neuro"],
    "seizure": ["neuro"],
    "syncope": ["neuro"],
    "headache": ["neuro"],
    "meningeal": ["neuro", "infectious"],
    "epigastric_pain": ["gastro"],
    "vaginal_discharge": ["obgyn"],
    "menstrual_relation": ["obgyn"],
    "tb_constitutional": ["respiratory"],
    "palpitations": ["cardio"],
}

SYMPTOM_DOMAIN_WEIGHTS: Dict[str, float] = {
    "vaginal_discharge": 2.2,
    "neuro_focal": 2.2,
    "meningeal": 2.2,
    "seizure": 2.0,
    "syncope": 1.8,
    "tonsil_plaque": 1.8,
    "tb_constitutional": 1.8,
    "palpitations": 1.8,
    "edema": 1.6,
    "flank_pain": 1.6,
    "lymph_nodes": 1.4,
    "sputum": 1.3,
}

SYMPTOM_QUERY_EXPANSIONS: Dict[str, List[str]] = {
    "cough": ["бронхит", "пневмония", "инфекция дыхательных путей"],
    "sore_throat": ["фарингит", "тонзиллит", "ангина", "назофарингит"],
    "chronic_course": ["хроническое течение", "длительные симптомы"],
    "chills": ["бактериальная инфекция", "сепсис"],
    "rhinitis": ["ринит", "орви"],
    "dyspnea": ["бронхиальная астма", "обструктивный синдром"],
    "wheeze": ["бронхиальная астма", "бронхообструкция"],
    "sputum": ["пневмония", "бронхит", "хобл"],
    "tonsillitis": ["тонзиллит", "ангина"],
    "tonsil_plaque": ["инфекционный мононуклеоз", "дифтерия", "скарлатина"],
    "lymph_nodes": ["инфекционный мононуклеоз", "дифтерия", "вирусная инфекция"],
    "abdominal_pain": ["гастроэнтерит", "абдоминальный синдром"],
    "ruq_pain": ["холецистит", "желчнокаменная болезнь"],
    "nausea": ["диспепсия", "гастрит"],
    "vomiting": ["гастроэнтерит", "пищевое отравление"],
    "diarrhea": ["кишечная инфекция", "энтерит", "колит"],
    "jaundice": ["гепатит", "холестаз"],
    "dysuria": ["цистит", "пиелонефрит"],
    "flank_pain": ["пиелонефрит", "инфекция мочевой системы"],
    "rash": ["экзантема", "дерматит"],
    "trauma": ["черепно-мозговая травма", "сотрясение мозга", "ушиб головы"],
    "edema": ["сердечная недостаточность", "декомпенсация"],
    "neuro_focal": ["инсульт", "онмк", "ишемический инсульт"],
    "seizure": ["эпилепсия", "фебрильные судороги"],
    "syncope": ["обморок", "нарушение сознания"],
    "headache": ["инсульт", "кровоизлияние"],
    "meningeal": ["менингит", "энцефалит", "нейроинфекция"],
    "epigastric_pain": ["панкреатит", "язвенная болезнь", "гастрит"],
    "vaginal_discharge": ["аднексит", "эндометрит", "воспалительные заболевания органов малого таза"],
    "menstrual_relation": ["аднексит", "эндометрит", "воспалительные заболевания органов малого таза"],
    "tb_constitutional": ["туберкулез органов дыхания", "туберкулез", "латентная туберкулезная инфекция"],
    "palpitations": ["аритмия", "фибрилляция предсердий", "нарушение ритма сердца"],
}

DOMAIN_QUERY_EXPANSIONS: Dict[str, List[str]] = {
    "respiratory": [
        "орви",
        "грипп",
        "ларингит",
        "трахеит",
        "бронхит",
        "пневмония",
        "астма",
        "хобл",
        "фарингит",
    ],
    "gastro": ["гастроэнтерит", "гастрит", "колит", "панкреатит", "холецистит"],
    "urinary": ["инфекция мочевыводящих путей", "цистит", "пиелонефрит"],
    "cardio": ["сердечная недостаточность", "кардиомиопатия", "ишемическая болезнь сердца"],
    "neuro": ["инсульт", "эпилепсия", "судорожный синдром", "черепно-мозговая травма"],
    "obgyn": ["аднексит", "эндометрит", "воспалительные заболевания органов малого таза"],
    "trauma": ["черепно-мозговая травма", "сотрясение мозга", "ушиб головы", "перелом"],
}

DOMAIN_TEXT_WEIGHTS: Dict[str, float] = {
    "obgyn": 3.0,
    "neuro": 2.6,
    "cardio": 2.4,
    "respiratory": 2.0,
    "gastro": 2.0,
    "urinary": 2.0,
    "trauma": 2.1,
    "infectious": 1.8,
}

DOMAIN_PATTERNS: Dict[str, re.Pattern[str]] = {
    "respiratory": re.compile(
        r"дых|легк|бронх|пневм|ларинг|фаринг|тонзилл|ангин|орви|грипп|трахе|ринит|коклюш|астм|верхн.{0,8}дых",
        re.I,
    ),
    "gastro": re.compile(
        r"живот|абдомин|гастр|киш|печен|желч|холецист|панкреат|гепат|колит|энтер|рвот|тошнот|подребер",
        re.I,
    ),
    "urinary": re.compile(r"почек|почеч|моче|цистит|пиелонеф|уретр|гидронеф", re.I),
    "cardio": re.compile(
        r"серд|карди|инфаркт|гипертенз|аритм|стенокард|одышк|отеки|отёк|ортопноэ|сердцебиен",
        re.I,
    ),
    "neuro": re.compile(
        r"невр|инсульт|эпилеп|менинг|энцефал|нерв|асимметр|реч|судорог|потер.{0,10}созн|головн.{0,8}бол",
        re.I,
    ),
    "trauma": re.compile(r"травм|перелом|ушиб|ожог|рана|ампутац", re.I),
    "oncology": re.compile(r"рак|карцин|сарком|лимфом|лейкоз|новообраз|опухол|онкол", re.I),
    "transplant": re.compile(r"трансплант|донор|аллогенн|гемопоэтическ|костного мозга", re.I),
    "obgyn": re.compile(
        r"беремен|роды|послерод|гинек|матк|плод|акушер|выделен|вагин|полов.{0,8}акт|тазов|менстру",
        re.I,
    ),
    "infectious": re.compile(
        r"инфекц|лихорад|озноб|сып|скарлат|корь|краснух|ветрян|дифтер|мононуклеоз|сепсис|энтеровир",
        re.I,
    ),
}

DOMAIN_QUESTION_PRIORITIES: Dict[str, List[str]] = {
    "respiratory": [
        "fever",
        "rhinitis",
        "dyspnea",
        "wheeze",
        "sputum",
        "tonsillitis",
        "tonsil_plaque",
        "chest_pain",
    ],
    "gastro": [
        "abdominal_pain",
        "ruq_pain",
        "nausea",
        "vomiting",
        "diarrhea",
        "jaundice",
        "fever",
    ],
    "urinary": ["dysuria", "flank_pain", "fever", "abdominal_pain"],
    "trauma": ["trauma", "bleeding", "chest_pain", "abdominal_pain"],
    "cardio": ["dyspnea", "chest_pain", "palpitations", "edema", "fever"],
    "neuro": ["neuro_focal", "seizure", "meningeal", "syncope", "headache", "fever"],
    "infectious": ["fever", "rash", "tonsillitis", "tonsil_plaque", "lymph_nodes", "meningeal"],
    "obgyn": ["vaginal_discharge", "menstrual_relation", "fever", "abdominal_pain", "bleeding"],
}

GENERIC_QUESTION_PRIORITY: List[str] = [
    "fever",
    "dyspnea",
    "sputum",
    "chest_pain",
    "abdominal_pain",
    "menstrual_relation",
    "flank_pain",
    "meningeal",
    "dysuria",
    "rash",
    "trauma",
]

NEGATION_PATTERNS: Dict[str, re.Pattern[str]] = {
    "fever": re.compile(r"без\s+температур|температур[аы]?\s+нет|нет\s+температур|афебрил", re.I),
    "dyspnea": re.compile(r"без\s+одышк|одышк[аи]?\s+нет|нет\s+одышк", re.I),
    "cough": re.compile(r"без\s+каш[её]л|без\s+кашл|каш[её]л[яе]?\s+нет|кашл[яе]?\s+нет|нет\s+каш[её]л|нет\s+кашл", re.I),
    "vomiting": re.compile(r"без\s+рвот|рвот[аы]?\s+нет|нет\s+рвот", re.I),
    "diarrhea": re.compile(r"без\s+диаре|диаре[яи]?\s+нет|нет\s+диаре", re.I),
}


def collect_symptoms(text: str) -> List[str]:
    found: List[str] = []
    lowered = (text or "").casefold()
    for rule in SYMPTOM_RULES:
        rule_id = str(rule["id"])
        if not rule["pattern"].search(lowered):
            continue
        neg_pattern = NEGATION_PATTERNS.get(rule_id)
        if neg_pattern and neg_pattern.search(lowered):
            continue
        found.append(rule_id)
    return found


def infer_domains_from_symptoms(symptom_ids: List[str]) -> List[str]:
    domains: set[str] = set()
    for symptom_id in symptom_ids:
        domains.update(SYMPTOM_TO_DOMAINS.get(symptom_id, []))
    return sorted(domains)


def collect_domains_from_text(text: str) -> List[str]:
    domains: List[str] = []
    for name, pattern in DOMAIN_PATTERNS.items():
        if pattern.search(text):
            domains.append(name)
    return sorted(set(domains))


def infer_domain_scores(query_text: str, symptom_ids: List[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for symptom_id in symptom_ids:
        weight = float(SYMPTOM_DOMAIN_WEIGHTS.get(symptom_id, 1.0))
        for domain in SYMPTOM_TO_DOMAINS.get(symptom_id, []):
            scores[domain] = scores.get(domain, 0.0) + weight
    for domain, pattern in DOMAIN_PATTERNS.items():
        if pattern.search(query_text):
            scores[domain] = scores.get(domain, 0.0) + float(DOMAIN_TEXT_WEIGHTS.get(domain, 2.0))
    return scores


def select_primary_domains(scores: Dict[str, float]) -> List[str]:
    if not scores:
        return []
    max_score = max(scores.values())
    if max_score <= 0:
        return []
    selected = [
        domain
        for domain, score in scores.items()
        if score >= (max_score - 0.25) and score >= 1.5
    ]
    return sorted(set(selected))


class SearchBackend(Protocol):
    """Expected protocol search interface."""

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        ...


@dataclass
class WorkflowContext:
    query: str
    top_k: int
    normalized_query: str = ""
    tokens: List[str] = field(default_factory=list)
    mkb_codes: List[str] = field(default_factory=list)
    audience_hint: str = "unknown"  # unknown|adult|pediatric
    query_symptoms: List[str] = field(default_factory=list)
    query_domains: List[str] = field(default_factory=list)
    query_domain_scores: Dict[str, float] = field(default_factory=dict)
    retrieval_query: str = ""
    retrieval_expansions: List[str] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    ranked_candidates: List[Dict[str, Any]] = field(default_factory=list)
    clarification_answers: Dict[str, str] = field(default_factory=dict)
    confidence: Dict[str, Any] = field(default_factory=dict)
    clarification: Dict[str, Any] = field(default_factory=dict)
    safety: Dict[str, Any] = field(default_factory=dict)
    assistant_answer: str = ""


class BaseAgent:
    name = "base"

    def run(self, context: WorkflowContext, step_cfg: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class IntakeAgent(BaseAgent):
    name = "intake"

    def run(self, context: WorkflowContext, step_cfg: Dict[str, Any]) -> Dict[str, Any]:
        normalized = " ".join(context.query.strip().split())
        tokens = [t.casefold() for t in TOKEN_RE.findall(normalized)]
        mkb_codes = sorted(
            {token.upper() for token in tokens if MKB_RE.fullmatch(token.upper())}
        )

        audience = "unknown"
        if any(token in PEDIATRIC_MARKERS for token in tokens):
            audience = "pediatric"
        elif any(token in ADULT_MARKERS for token in tokens):
            audience = "adult"

        context.normalized_query = normalized
        context.tokens = tokens
        context.mkb_codes = mkb_codes
        context.audience_hint = audience
        query_symptoms = collect_symptoms(normalized)
        domain_scores = infer_domain_scores(normalized, query_symptoms)
        query_domains = select_primary_domains(domain_scores)
        if not query_domains:
            query_domains = sorted(
                set(infer_domains_from_symptoms(query_symptoms))
                | set(collect_domains_from_text(normalized))
            )
        context.query_symptoms = query_symptoms
        context.query_domains = query_domains
        context.query_domain_scores = domain_scores

        return {
            "normalized_query": normalized,
            "token_count": len(tokens),
            "mkb_codes": mkb_codes,
            "audience_hint": audience,
            "query_symptoms": query_symptoms,
            "query_domains": query_domains,
            "query_domain_scores": domain_scores,
        }


class RetrievalAgent(BaseAgent):
    name = "retrieval"

    def __init__(self, search_backend: SearchBackend) -> None:
        self.search_backend = search_backend

    @staticmethod
    def _dedupe_terms(terms: List[str], max_terms: int) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for raw in terms:
            term = " ".join(str(raw).strip().split())
            key = term.casefold()
            if not term or key in seen:
                continue
            seen.add(key)
            out.append(term)
            if len(out) >= max_terms:
                break
        return out

    def _build_expansion_terms(
        self,
        context: WorkflowContext,
        max_terms: int,
    ) -> List[str]:
        terms: List[str] = []
        for domain in context.query_domains:
            terms.extend(DOMAIN_QUERY_EXPANSIONS.get(domain, []))
        query_domain_set = set(context.query_domains)
        for symptom_id in context.query_symptoms:
            symptom_domains = set(SYMPTOM_TO_DOMAINS.get(symptom_id, []))
            if query_domain_set and symptom_domains and not (query_domain_set & symptom_domains):
                continue
            terms.extend(SYMPTOM_QUERY_EXPANSIONS.get(symptom_id, []))
        return self._dedupe_terms(terms, max_terms=max_terms)

    @staticmethod
    def _merge_candidates(
        query_results: List[tuple[str, List[Dict[str, Any]]]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        source_weights = {
            "base": 1.0,
            "full_expanded": 0.9,
            "variant": 0.8,
            "term_only": 0.7,
        }
        merged: Dict[str, Dict[str, Any]] = {}

        def add_item(item: Dict[str, Any], source: str, rank: int) -> None:
            cid = str(item.get("id", "")).strip()
            if not cid:
                return
            score = float(item.get("relevance", 0.0))
            src_weight = float(source_weights.get(source, 0.6))
            rank_boost = src_weight * (1.0 / (1.0 + rank))
            current = merged.get(cid)
            if current is None:
                rec = dict(item)
                rec["_sources"] = {source}
                rec["_max_relevance"] = score
                rec["_source_rank_boost"] = rank_boost
                merged[cid] = rec
                return
            current["_sources"].add(source)
            current["_source_rank_boost"] = float(current.get("_source_rank_boost", 0.0)) + rank_boost
            if score > float(current.get("_max_relevance", 0.0)):
                for key, value in item.items():
                    current[key] = value
                current["_max_relevance"] = score

        for source, rows in query_results:
            for idx, item in enumerate(rows, start=1):
                add_item(item, source, idx)

        rows: List[Dict[str, Any]] = []
        for item in merged.values():
            src_count = len(item.get("_sources", set()))
            base = float(item.get("_max_relevance", item.get("relevance", 0.0)))
            support_bonus = 0.02 * max(0, src_count - 1)
            rank_bonus = min(0.12, float(item.get("_source_rank_boost", 0.0)))
            item["relevance"] = round(min(1.0, base + support_bonus + rank_bonus), 4)
            item.pop("_sources", None)
            item.pop("_max_relevance", None)
            item.pop("_source_rank_boost", None)
            rows.append(item)

        rows.sort(
            key=lambda row: (
                float(row.get("relevance", 0.0)),
                str(row.get("title", "")),
            ),
            reverse=True,
        )
        return rows[:limit]

    def run(self, context: WorkflowContext, step_cfg: Dict[str, Any]) -> Dict[str, Any]:
        params = step_cfg.get("params", {})
        multiplier = int(params.get("candidate_pool_multiplier", 4))
        max_expansion_terms = max(0, int(params.get("max_expansion_terms", 10)))
        max_search_variants = max(0, int(params.get("max_search_variants", 4)))
        term_only_queries = max(0, int(params.get("term_only_queries", 2)))
        pool_size = max(context.top_k * multiplier, int(params.get("min_pool_size", 50)))

        base_query = context.normalized_query or context.query
        expansion_terms = self._build_expansion_terms(
            context=context,
            max_terms=max_expansion_terms,
        )
        expanded_query = base_query
        if expansion_terms:
            expanded_query = f"{base_query} {' '.join(expansion_terms)}"

        search_batches: List[tuple[str, List[Dict[str, Any]]]] = []
        primary = self.search_backend.search(base_query, limit=pool_size)
        search_batches.append(("base", primary))

        expanded: List[Dict[str, Any]] = []
        if expansion_terms:
            expanded = self.search_backend.search(expanded_query, limit=pool_size)
            search_batches.append(("full_expanded", expanded))

            for term in expansion_terms[:max_search_variants]:
                variant_query = f"{base_query} {term}"
                variant_rows = self.search_backend.search(variant_query, limit=pool_size)
                search_batches.append(("variant", variant_rows))

            for term in expansion_terms[:term_only_queries]:
                term_rows = self.search_backend.search(term, limit=pool_size)
                search_batches.append(("term_only", term_rows))

        candidates = self._merge_candidates(search_batches, limit=pool_size)
        context.retrieval_query = expanded_query
        context.retrieval_expansions = expansion_terms
        context.candidates = candidates

        return {
            "pool_size": pool_size,
            "retrieved": len(candidates),
            "retrieved_primary": len(primary),
            "retrieved_expanded": len(expanded),
            "search_variants": len(search_batches),
            "expansions_used": expansion_terms,
            "top_candidate_id": candidates[0]["id"] if candidates else None,
        }


class RankingAgent(BaseAgent):
    name = "ranking"

    @staticmethod
    def _overlap_ratio(query_tokens: List[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        candidate_tokens = [token.casefold() for token in TOKEN_RE.findall(text)]
        if not candidate_tokens:
            return 0.0
        matched = 0
        for token in query_tokens:
            if token in candidate_tokens:
                matched += 1
        return matched / len(query_tokens)

    @staticmethod
    def _candidate_text(item: Dict[str, Any]) -> str:
        title = str(item.get("title", ""))
        summary = str(item.get("summary", ""))
        snippet = str(item.get("snippet", ""))
        sections = item.get("sections") or []
        section_text = " ".join(str(x) for x in sections if str(x).strip())
        return f"{title} {summary} {snippet} {section_text}"

    @staticmethod
    def _mkb_bonus(query_codes: List[str], candidate_codes: List[str]) -> float:
        if not query_codes:
            return 0.0
        up = [str(code).upper() for code in candidate_codes]
        hits = 0
        for code in query_codes:
            if any(code == c or c.startswith(code) for c in up):
                hits += 1
        return min(0.4, hits * 0.2)

    @staticmethod
    def _audience_bonus(audience_hint: str, title: str) -> float:
        title_low = title.casefold()
        if audience_hint == "pediatric" and "дет" in title_low:
            return 0.1
        if audience_hint == "adult" and "взросл" in title_low:
            return 0.1
        return 0.0

    @staticmethod
    def _symptom_signal(query_symptoms: List[str], candidate_text: str) -> float:
        if not query_symptoms:
            return 0.0
        query_set = set(query_symptoms)
        candidate_set = set(collect_symptoms(candidate_text))
        if not candidate_set:
            return -0.06

        hits = len(query_set & candidate_set)
        if hits <= 0:
            return -0.12

        coverage = hits / max(1, len(query_set))
        precision = hits / max(1, len(candidate_set))
        signal = (0.22 * coverage) + (0.08 * precision)
        if hits == len(query_set):
            signal += 0.05
        return min(0.32, signal)

    @staticmethod
    def _domain_signal(query_domains: List[str], candidate_text: str) -> float:
        if not query_domains:
            return 0.0
        q = set(query_domains)
        c = set(collect_domains_from_text(candidate_text))
        overlap = q & c
        signal = 0.0

        focused_domains = {
            "respiratory",
            "gastro",
            "urinary",
            "trauma",
            "cardio",
            "neuro",
            "obgyn",
            "infectious",
        }
        focused_query = q & focused_domains

        if overlap:
            signal += min(0.18, 0.10 + (0.04 * len(overlap)))
        elif c:
            signal -= 0.12

        if focused_query and not (c & focused_query) and c:
            signal -= 0.14
        if "respiratory" in focused_query and "respiratory" not in c and c:
            signal -= 0.10
        if "urinary" in focused_query and "urinary" not in c and c:
            signal -= 0.12
        if "gastro" in focused_query and "gastro" not in c and c:
            signal -= 0.08
        if "cardio" in focused_query and "cardio" not in c and c:
            signal -= 0.12
        if "neuro" in focused_query and "neuro" not in c and c:
            signal -= 0.12
        if "obgyn" in focused_query and "obgyn" not in c and c:
            signal -= 0.14
        if "infectious" in focused_query and "infectious" not in c and c:
            signal -= 0.10

        if "respiratory" in q and {"oncology", "transplant"} & c:
            signal -= 0.18
        if "gastro" in q and {"oncology", "transplant"} & c:
            signal -= 0.14
        if "urinary" in q and {"oncology", "transplant"} & c:
            signal -= 0.12
        if "trauma" in q and "trauma" not in c and c:
            signal -= 0.10
        if "obgyn" in q and {"gastro", "urinary"} & c and "obgyn" not in c:
            signal -= 0.12

        return max(-0.35, min(0.28, signal))

    @staticmethod
    def _title_expansion_boost(title: str, expansion_terms: List[str]) -> float:
        if not expansion_terms:
            return 0.0
        title_low = title.casefold()
        hits = 0
        for term in expansion_terms:
            t = str(term).strip().casefold()
            if not t:
                continue
            if t in title_low:
                hits += 1
                continue
            parts = [p for p in t.split() if len(p) >= 5]
            if parts and any(part in title_low for part in parts):
                hits += 1
        return min(0.18, 0.05 * hits)

    @staticmethod
    def _special_penalty(
        query_text: str,
        query_symptoms: List[str],
        candidate_text: str,
    ) -> float:
        q = (query_text or "").casefold()
        c = (candidate_text or "").casefold()
        symptoms = set(query_symptoms)
        penalty = 0.0

        if re.search(r"беремен|послерод|акушер|роды|плод", c) and not re.search(
            r"беремен|послерод|акушер|роды|плод|вагин|гинек|полов|выделен",
            q,
        ):
            penalty += 0.45

        if re.search(r"трансплант|донор|аллогенн|гемопоэтическ", c) and not re.search(
            r"трансплант|донор|гемобласт|реципиент",
            q,
        ):
            penalty += 0.40

        if re.search(r"эбол|геморрагическ.{0,20}лихорад|сибирск.{0,10}язв", c) and not re.search(
            r"эбол|геморраг|контакт|эпидем|поездк|тропик|кровотеч",
            q,
        ):
            penalty += 0.40

        if re.search(r"без\s+температур|температур[аы]?\s+нет|нет\s+температур|афебрил", q) and re.search(
            r"лихорад|грипп|орви|вирус|инфекци",
            c,
        ):
            penalty += 0.10

        if "vaginal_discharge" in symptoms and not re.search(r"гинек|аднекс|эндомет|тазов|матк|акуш", c):
            penalty += 0.24

        if {"neuro_focal", "seizure", "syncope"} & symptoms and re.search(r"хроническ.{0,20}ишеми", c):
            penalty += 0.24

        if {"abdominal_pain", "diarrhea", "vomiting", "epigastric_pain", "ruq_pain"} & symptoms:
            if re.search(r"ботулизм|холер", c) and not ({"neuro_focal", "seizure", "syncope"} & symptoms):
                penalty += 0.34

        if {"dyspnea", "chest_pain", "edema", "palpitations"} & symptoms:
            if re.search(r"туберкул|орви|грипп", c) and not ({"cough", "rhinitis", "tb_constitutional"} & symptoms):
                penalty += 0.24
        if "edema" in symptoms and not ({"cough", "rhinitis", "tb_constitutional"} & symptoms):
            if re.search(r"туберкул|орви|грипп|пневмон|бронх", c):
                penalty += 0.18
        if {"cough", "dyspnea", "wheeze"} <= symptoms and "tb_constitutional" not in symptoms and re.search(
            r"туберкул",
            c,
        ):
            penalty += 0.22
        if {"cough", "dyspnea", "wheeze"} <= symptoms and "fever" not in symptoms and re.search(
            r"гиперсенситив|интерстиц|фиброз",
            c,
        ):
            penalty += 0.22

        if {"cough", "fever"} <= symptoms and re.search(r"инородн.{0,15}пищевод", c):
            penalty += 0.34
        if {"cough", "chest_pain", "fever"} <= symptoms and re.search(r"герпес|трихинел|лейшман", c):
            penalty += 0.24
        if {"cough", "chest_pain", "fever"} <= symptoms and re.search(
            r"кардиомиопати|сердечн.{0,20}недостаточ|гипертрофическ",
            c,
        ):
            penalty += 0.18

        if {"seizure", "meningeal", "neuro_focal"} & symptoms and re.search(
            r"психическ|поведенческ|табак|психоактив",
            c,
        ):
            penalty += 0.34

        if "vaginal_discharge" in symptoms and re.search(
            r"аппендиц|кишечн|колит|гастроэнтер",
            c,
        ) and not re.search(r"гинек|аднекс|эндомет|тазов|матк|воспалительн", c):
            penalty += 0.42
        if "menstrual_relation" in symptoms and re.search(
            r"перитонит|колит|аппендиц|мегауретер",
            c,
        ) and not re.search(r"гинек|аднекс|эндомет|тазов|матк|менстру|дисменор", c):
            penalty += 0.22

        if {"ruq_pain", "jaundice"} <= symptoms and re.search(
            r"лямблиоз|кишечн|энтер|колит",
            c,
        ) and not re.search(r"гепат|холецист|желч|печен|панкреат", c):
            penalty += 0.26

        if {"trauma", "headache"} <= symptoms and re.search(
            r"ишемическ.{0,20}инсульт|онмк",
            c,
        ):
            penalty += 0.20

        if {"flank_pain", "dysuria", "fever"} <= symptoms and re.search(
            r"цистит|уретр",
            c,
        ) and not re.search(r"пиелонеф|мочев", c):
            penalty += 0.10

        if {"sore_throat", "tonsillitis", "tonsil_plaque", "lymph_nodes", "fever"} <= symptoms and re.search(
            r"тонзилл|фаринг",
            c,
        ) and not re.search(r"мононуклеоз|дифтер|скарлат|инфекц", c):
            penalty += 0.34
        return penalty

    @staticmethod
    def _combination_title_boost(query_symptoms: List[str], title: str) -> float:
        symptoms = set(query_symptoms)
        t = (title or "").casefold()
        boost = 0.0

        if {"cough", "dyspnea", "wheeze"} <= symptoms and re.search(r"астм|обструк|хобл|бронх", t):
            boost += 0.22
        if {"cough", "dyspnea", "wheeze"} <= symptoms and "fever" not in symptoms and re.search(
            r"астм|обструк|хобл",
            t,
        ):
            boost += 0.14
        if {"cough", "sputum", "fever"} <= symptoms and re.search(r"пневмон|бронхит|орви|грипп", t):
            boost += 0.30
        if {"cough", "sputum", "chest_pain", "fever"} <= symptoms and re.search(r"пневмон|бронхит", t):
            boost += 0.10
        if {"chronic_course", "cough", "sputum"} <= symptoms and {"dyspnea", "wheeze"} & symptoms and re.search(
            r"хронич|хобл|бронхит|обструк",
            t,
        ):
            boost += 0.34
        if {"dyspnea", "chest_pain", "edema"} <= symptoms and re.search(r"сердечн|карди|аритм|гипертенз|ишем|стенокард|фибрил", t):
            boost += 0.32
        elif {"dyspnea", "edema"} <= symptoms and re.search(r"сердечн|карди|аритм|гипертенз|ишем|стенокард|фибрил", t):
            boost += 0.24
        elif {"dyspnea", "chest_pain"} <= symptoms and re.search(r"сердечн|карди|аритм|гипертенз|ишем|стенокард|фибрил", t):
            boost += 0.24
        if {"palpitations", "dyspnea"} & symptoms and re.search(r"аритм|фибрил|сердечн|карди", t):
            boost += 0.20
        if {"sore_throat", "tonsillitis"} <= symptoms and re.search(r"тонзил|ангин|фаринг", t):
            boost += 0.24
        if {"cough", "sore_throat"} <= symptoms and re.search(r"фаринг|тонзил|ангин|орви|грипп|ларинг", t):
            boost += 0.18
        if {"abdominal_pain", "nausea", "vomiting"} & symptoms and re.search(r"гастр|панкреат|энтер|колит", t):
            boost += 0.16
        if {"epigastric_pain", "nausea", "vomiting"} <= symptoms and re.search(r"панкреат|гастр|язв|желуд", t):
            boost += 0.24
        if {"abdominal_pain", "diarrhea", "vomiting"} <= symptoms and re.search(r"кишечн|энтер|колит|гастроэнтер", t):
            boost += 0.20
        if {"abdominal_pain", "diarrhea", "fever"} <= symptoms and re.search(r"кишечн|энтер|колит|гастроэнтер|диаре", t):
            boost += 0.24
        if {"ruq_pain", "jaundice"} <= symptoms and re.search(r"холецист|гепат|желч|печен", t):
            boost += 0.30
        elif {"ruq_pain", "jaundice"} & symptoms and re.search(r"холецист|гепат|желч|печен", t):
            boost += 0.16
        if {"ruq_pain", "nausea", "vomiting"} & symptoms and re.search(r"холецист|желч|печен|гепат", t):
            boost += 0.22
        if {"rhinitis"} <= symptoms and re.search(r"ринит|орви|грипп|аллерг", t):
            boost += 0.20
        if {"dysuria", "fever"} <= symptoms and re.search(r"цистит|пиелонеф|моч", t):
            boost += 0.20
        if {"flank_pain", "dysuria", "fever"} <= symptoms and re.search(r"пиелонеф|мочев|инфекц.{0,15}моч", t):
            boost += 0.28
        if {"neuro_focal", "headache"} & symptoms and re.search(r"инсульт|кровоизлияни|онмк", t):
            boost += 0.20
        if {"neuro_focal", "syncope"} & symptoms and re.search(r"инсульт|кровоизлияни|ишемическ", t):
            boost += 0.12
        if {"seizure"} & symptoms and re.search(r"эпилеп|судорог", t):
            boost += 0.18
        if {"seizure", "meningeal"} & symptoms and re.search(r"невр|эпилеп|судорог|менинг|энцефал|кровоизлияни", t):
            boost += 0.18
        if {"headache", "meningeal", "fever"} <= symptoms and re.search(r"менинг|энцефал|нейроинфекц", t):
            boost += 0.30
        if {"tb_constitutional", "cough"} <= symptoms and re.search(r"туберкул", t):
            boost += 0.26
        if {"trauma", "headache"} <= symptoms and re.search(r"травм|череп|сотряс|ушиб|голов", t):
            boost += 0.28
        elif {"trauma", "headache", "syncope"} & symptoms and re.search(r"травм|череп|сотряс|голов", t):
            boost += 0.16
        if {"vaginal_discharge", "abdominal_pain"} <= symptoms and re.search(r"аднекс|эндометрит|гинек|тазов", t):
            boost += 0.30
        if {"vaginal_discharge", "abdominal_pain", "fever"} <= symptoms and re.search(
            r"аднекс|эндометрит|гинек|тазов|матк|воспалительн",
            t,
        ):
            boost += 0.12
        if {"vaginal_discharge", "dysuria", "fever"} <= symptoms and re.search(
            r"аднекс|эндометрит|гинек|тазов|матк|воспалительн",
            t,
        ):
            boost += 0.16
        if {"menstrual_relation", "abdominal_pain"} <= symptoms and re.search(
            r"аднекс|эндометрит|гинек|тазов|матк|воспалительн|дисменор",
            t,
        ):
            boost += 0.18
        if {"menstrual_relation", "abdominal_pain", "fever"} <= symptoms and re.search(
            r"аднекс|эндометрит|гинек|тазов|матк|воспалительн|дисменор",
            t,
        ):
            boost += 0.30
        if {"sore_throat", "tonsillitis", "tonsil_plaque", "lymph_nodes", "fever"} <= symptoms and re.search(
            r"мононуклеоз|дифтер|скарлат|корь|краснух|ветрян|инфекц",
            t,
        ):
            boost += 0.32
        if {"rash", "fever", "tonsillitis"} & symptoms and re.search(r"скарлат|краснух|корь|мононуклеоз|дифтер", t):
            boost += 0.20
        if {"chills", "fever"} <= symptoms and re.search(r"инфекц|сепсис|бактериальн", t):
            boost += 0.10
        if {"diarrhea", "vomiting", "fever"} <= symptoms and re.search(r"кишечн|энтер|гастроэнтер|инфекц", t):
            boost += 0.18

        return min(0.45, boost)

    def run(self, context: WorkflowContext, step_cfg: Dict[str, Any]) -> Dict[str, Any]:
        ranked: List[Dict[str, Any]] = []
        for item in context.candidates:
            title = str(item.get("title", ""))
            candidate_text = self._candidate_text(item)
            base_relevance = float(item.get("relevance", 0.0))
            query_overlap = self._overlap_ratio(context.tokens, candidate_text)
            mkb_bonus = self._mkb_bonus(context.mkb_codes, item.get("mkb_codes", []))
            audience_bonus = self._audience_bonus(context.audience_hint, title)
            symptom_signal = self._symptom_signal(context.query_symptoms, candidate_text)
            domain_signal = self._domain_signal(context.query_domains, candidate_text)
            expansion_boost = self._title_expansion_boost(
                title,
                context.retrieval_expansions,
            )
            special_penalty = self._special_penalty(
                context.normalized_query or context.query,
                context.query_symptoms,
                candidate_text,
            )
            combo_boost = self._combination_title_boost(context.query_symptoms, title)

            final_score = (
                (0.42 * base_relevance)
                + (0.20 * query_overlap)
                + mkb_bonus
                + audience_bonus
                + symptom_signal
                + domain_signal
                + expansion_boost
                + combo_boost
                - special_penalty
            )
            enriched = dict(item)
            enriched["agentic_score"] = round(max(0.0, final_score), 4)
            ranked.append(enriched)

        ranked.sort(key=lambda x: x.get("agentic_score", 0.0), reverse=True)
        context.ranked_candidates = ranked
        top = ranked[: context.top_k]

        return {
            "ranked": len(ranked),
            "returned": len(top),
            "top_candidate_id": top[0]["id"] if top else None,
        }


class ClarificationAgent(BaseAgent):
    name = "clarification"

    @staticmethod
    def _candidate_text(item: Dict[str, Any]) -> str:
        title = str(item.get("title", ""))
        summary = str(item.get("summary", ""))
        snippet = str(item.get("snippet", ""))
        return f"{title} {summary} {snippet}"

    @staticmethod
    def _normalize_answer(value: Any) -> str:
        val = str(value or "").strip().casefold()
        if val in {"yes", "y", "1", "true", "да"}:
            return "yes"
        if val in {"no", "n", "0", "false", "нет"}:
            return "no"
        return "unknown"

    @staticmethod
    def _collect_symptoms(text: str) -> List[str]:
        return collect_symptoms(text)

    @staticmethod
    def _domain_question_priority(query_domains: List[str]) -> List[str]:
        ordered: List[str] = []
        seen: set[str] = set()
        for domain in query_domains:
            for symptom_id in DOMAIN_QUESTION_PRIORITIES.get(domain, []):
                if symptom_id in seen:
                    continue
                seen.add(symptom_id)
                ordered.append(symptom_id)
        for symptom_id in GENERIC_QUESTION_PRIORITY:
            if symptom_id in seen:
                continue
            seen.add(symptom_id)
            ordered.append(symptom_id)
        return ordered

    def _build_questions(
        self,
        query_text: str,
        query_domains: List[str],
        candidates: List[Dict[str, Any]],
        max_questions: int,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        query_symptoms = set(self._collect_symptoms(query_text))
        candidate_ids = [str(item.get("id", "")) for item in candidates if item.get("id")]
        total_candidates = len(candidate_ids)
        if total_candidates <= 0:
            return []

        presence: Dict[str, set[str]] = {}
        for item in candidates:
            cid = str(item.get("id", "")).strip()
            if not cid:
                continue
            symptoms = set(self._collect_symptoms(self._candidate_text(item)))
            for symptom in symptoms:
                if symptom in query_symptoms:
                    continue
                presence.setdefault(symptom, set()).add(cid)

        questions_by_id = {str(rule["id"]): str(rule["question"]) for rule in SYMPTOM_RULES}
        domain_priority = self._domain_question_priority(query_domains)

        discriminative: List[str] = []
        for symptom_id, proto_ids in presence.items():
            if 0 < len(proto_ids) < total_candidates:
                discriminative.append(symptom_id)
        discriminative_set = set(discriminative)

        selected_ids: List[str] = []
        seen_ids: set[str] = set()

        for symptom_id in domain_priority:
            if symptom_id in discriminative_set and symptom_id not in seen_ids:
                selected_ids.append(symptom_id)
                seen_ids.add(symptom_id)

        remaining_discriminative = sorted(
            [sid for sid in discriminative if sid not in seen_ids],
            key=lambda sid: min(len(presence[sid]), total_candidates - len(presence[sid])),
            reverse=True,
        )
        for sid in remaining_discriminative:
            selected_ids.append(sid)
            seen_ids.add(sid)

        if not selected_ids:
            for symptom_id in domain_priority:
                if symptom_id in presence and symptom_id not in seen_ids:
                    selected_ids.append(symptom_id)
                    seen_ids.add(symptom_id)

        if not selected_ids:
            selected_ids = sorted(
                presence.keys(),
                key=lambda sid: len(presence[sid]),
                reverse=True,
            )

        questions: List[Dict[str, Any]] = []
        for symptom_id in selected_ids:
            q_text = questions_by_id.get(symptom_id)
            rel = sorted(presence.get(symptom_id, set()))
            if not q_text or not rel:
                continue
            support = len(rel) / total_candidates
            balance = min(support, 1.0 - support)
            weight = 0.6 + (1.2 * balance)
            questions.append(
                {
                    "id": symptom_id,
                    "question": q_text,
                    "related_protocol_ids": rel,
                    "options": ["yes", "no", "unknown"],
                    "weight": round(weight, 3),
                }
            )
            if len(questions) >= max_questions:
                break
        return questions

    @staticmethod
    def _apply_answers(
        ranked: List[Dict[str, Any]],
        questions: List[Dict[str, Any]],
        answers: Dict[str, str],
        allowed_ids: set[str],
        yes_bonus: float,
        no_penalty: float,
        inverse_bonus: float,
    ) -> None:
        if not ranked or not questions or not answers:
            return

        for item in ranked:
            cid = str(item.get("id", ""))
            if cid not in allowed_ids:
                continue
            score = float(item.get("agentic_score", 0.0))
            for question in questions:
                qid = str(question.get("id", ""))
                answer = answers.get(qid, "unknown")
                if answer == "unknown":
                    continue
                related_ids = set(str(x) for x in question.get("related_protocol_ids", []))
                weight = max(0.25, float(question.get("weight", 1.0)))
                in_related = cid in related_ids
                if answer == "yes":
                    score += (yes_bonus * weight) if in_related else -(inverse_bonus * weight)
                elif answer == "no":
                    score += (inverse_bonus * weight) if not in_related else -(no_penalty * weight)
            item["agentic_score"] = round(max(0.0, score), 4)

        ranked.sort(key=lambda x: x.get("agentic_score", 0.0), reverse=True)

    @staticmethod
    def _confidence_distribution(
        ranked: List[Dict[str, Any]],
        temperature: float,
    ) -> List[Dict[str, Any]]:
        if not ranked:
            return []
        temp = max(0.01, float(temperature))
        scores = [max(float(item.get("agentic_score", 0.0)), 0.001) for item in ranked]
        max_score = max(scores)
        exp_scores = [math.exp((score - max_score) / temp) for score in scores]
        total = sum(exp_scores) or 1.0

        out: List[Dict[str, Any]] = []
        for item, score, exp_score in zip(ranked, scores, exp_scores):
            confidence_pct = 100.0 * exp_score / total
            out.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "agentic_score": round(score, 4),
                    "confidence_pct": round(confidence_pct, 1),
                }
            )
        return out

    def run(self, context: WorkflowContext, step_cfg: Dict[str, Any]) -> Dict[str, Any]:
        params = step_cfg.get("params", {})
        target_confidence = float(params.get("target_confidence_pct", 90.0))
        max_candidates = max(1, int(params.get("max_candidates", 5)))
        max_questions = max(1, int(params.get("max_questions", 5)))
        temperature = float(params.get("confidence_temperature", 0.05))
        yes_bonus = float(params.get("yes_bonus", 0.22))
        no_penalty = float(params.get("no_penalty", 0.22))
        inverse_bonus = float(params.get("inverse_bonus", 0.08))

        working = context.ranked_candidates[:max_candidates]
        questions = self._build_questions(
            query_text=context.normalized_query or context.query,
            query_domains=context.query_domains,
            candidates=working,
            max_questions=max_questions,
        )

        answers = {
            str(key): self._normalize_answer(value)
            for key, value in context.clarification_answers.items()
        }
        answered_count = sum(1 for answer in answers.values() if answer != "unknown")

        if answered_count > 0:
            scope_ids = {
                str(item.get("id", ""))
                for item in working
                if str(item.get("id", "")).strip()
            }
            self._apply_answers(
                ranked=context.ranked_candidates,
                questions=questions,
                answers=answers,
                allowed_ids=scope_ids,
                yes_bonus=yes_bonus,
                no_penalty=no_penalty,
                inverse_bonus=inverse_bonus,
            )
            working = context.ranked_candidates[:max_candidates]
            questions = self._build_questions(
                query_text=context.normalized_query or context.query,
                query_domains=context.query_domains,
                candidates=working,
                max_questions=max_questions,
            )

        distribution = self._confidence_distribution(working, temperature=temperature)
        top_confidence = float(distribution[0]["confidence_pct"]) if distribution else 0.0
        is_confident = top_confidence >= target_confidence
        required = len(distribution) > 1 and not is_confident

        remaining_questions = [
            question
            for question in questions
            if answers.get(str(question.get("id", "")), "unknown") == "unknown"
        ]

        context.confidence = {
            "target_confidence_pct": target_confidence,
            "top_protocol_confidence_pct": round(top_confidence, 1),
            "is_confident": is_confident,
            "candidate_confidences": distribution,
        }
        context.clarification = {
            "required": required,
            "questions": remaining_questions if required else [],
            "all_questions": questions,
            "answered_count": answered_count,
            "answers": answers,
        }

        return {
            "required": required,
            "top_confidence_pct": round(top_confidence, 1),
            "target_confidence_pct": target_confidence,
            "is_confident": is_confident,
            "questions_count": len(remaining_questions if required else []),
            "answered_count": answered_count,
        }


class SafetyAgent(BaseAgent):
    name = "safety"

    def run(self, context: WorkflowContext, step_cfg: Dict[str, Any]) -> Dict[str, Any]:
        emergency_hits = [token for token in context.tokens if token in EMERGENCY_MARKERS]
        emergency = len(emergency_hits) >= 2

        disclaimer = (
            "Ассистент носит справочный характер и не заменяет очную консультацию врача."
        )
        if emergency:
            warning = (
                "Обнаружены потенциально опасные признаки. "
                "Рекомендуется срочно обратиться за экстренной медицинской помощью."
            )
        else:
            warning = ""

        context.safety = {
            "disclaimer": disclaimer,
            "emergency_detected": emergency,
            "emergency_markers": sorted(set(emergency_hits)),
            "warning": warning,
        }
        return context.safety


class ResponseAgent(BaseAgent):
    name = "response"

    def __init__(self, search_backend: SearchBackend) -> None:
        self.search_backend = search_backend

    @staticmethod
    def _compact(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    @classmethod
    def _split_points(cls, text: str) -> List[str]:
        if not text:
            return []
        normalized = str(text).replace("\r", "\n")
        parts = re.split(r"\n+|[•▪●]\s*|\s*;\s*", normalized)
        out: List[str] = []
        seen: set[str] = set()
        for raw in parts:
            line = cls._compact(re.sub(r"^[\-\–\—\*\d\.\)\(]+\s*", "", raw))
            if len(line) < 28:
                continue
            key = line.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(line)
        return out

    @staticmethod
    def _build_clarification_only_answer(questions: List[Dict[str, Any]]) -> str:
        lines = ["Чтобы подготовить точные рекомендации, сначала уточните:"]
        for idx, question in enumerate(questions[:6], start=1):
            q = str(question.get("question", "")).strip()
            if q:
                lines.append(f"{idx}. {q}")
        if len(lines) == 1:
            lines.append("Опишите симптомы подробнее, включая длительность и сопутствующие признаки.")
        return "\n".join(lines)

    @classmethod
    def _join_sections(
        cls,
        content: Dict[str, str],
        section_markers: List[str],
    ) -> str:
        if not isinstance(content, dict):
            return ""
        chunks: List[str] = []
        for section, text in content.items():
            sec = str(section or "").casefold()
            if not sec:
                continue
            if not any(marker in sec for marker in section_markers):
                continue
            cleaned = cls._compact(str(text or ""))
            if cleaned:
                chunks.append(cleaned)
        return "\n".join(chunks)

    @classmethod
    def _pick_points(
        cls,
        text: str,
        pattern: str,
        limit: int,
    ) -> List[str]:
        out: List[str] = []
        for line in cls._split_points(text):
            if not re.search(pattern, line, re.I):
                continue
            out.append(line)
            if len(out) >= max(1, limit):
                break
        return out

    @staticmethod
    def _render_list(title: str, items: List[str]) -> str:
        if not items:
            return ""
        lines = [title]
        for item in items:
            lines.append(f"- {item}")
        return "\n".join(lines)

    def _fetch_protocol_payload(self, protocol_id: str) -> Dict[str, Any]:
        getter = getattr(self.search_backend, "get_protocol", None)
        if not callable(getter):
            return {}
        try:
            payload = getter(str(protocol_id), include_full_text=False)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}
        return {}

    def _build_clinical_answer(
        self,
        top_match: Dict[str, Any],
    ) -> str:
        title = str(top_match.get("title", "")).strip()
        protocol_id = str(top_match.get("id", "")).strip()
        payload = self._fetch_protocol_payload(protocol_id)
        content = payload.get("content") if isinstance(payload, dict) else {}
        if not isinstance(content, dict):
            content = {}

        treatment_text = self._join_sections(
            content,
            ["treatment", "лечен", "терап", "organization", "реабилит"],
        )
        hospitalization_text = self._join_sections(
            content,
            ["hospitalization", "госпитал", "стационар"],
        )

        treatment_points = self._pick_points(
            treatment_text,
            r"лечен|терап|рекоменд|режим|наблюден|контрол|симптом|ингаля|инфуз|питьев",
            limit=6,
        )
        if not treatment_points:
            treatment_points = self._split_points(treatment_text)[:6]

        medication_points = self._pick_points(
            treatment_text,
            r"препарат|антибиот|антибак|антивирус|доза|\bмг\b|таблет|раствор|ингаля|амокси|цеф|азит|ибупроф|парацет|сальбут|будесон",
            limit=6,
        )

        hospitalization_points = self._pick_points(
            hospitalization_text or treatment_text,
            r"госпитал|стационар|показан|экстр|неотлож|тяжел|сатурац|дыхательн|осложнен|критер",
            limit=6,
        )

        if not (treatment_points or medication_points or hospitalization_points):
            fallback = self._compact(str(top_match.get("snippet") or top_match.get("summary") or ""))
            if fallback:
                treatment_points = [fallback]

        sections: List[str] = []
        if title:
            sections.append(f"По вашему описанию наиболее подходит: {title}.")

        treat_block = self._render_list("Как лечить:", treatment_points)
        if treat_block:
            sections.append(treat_block)

        meds_block = self._render_list("Какие лекарства могут применяться:", medication_points)
        if meds_block:
            sections.append(meds_block)

        hosp_block = self._render_list("Когда нужна госпитализация:", hospitalization_points)
        if hosp_block:
            sections.append(hosp_block)

        if not sections:
            return (
                "По вашему описанию не удалось выделить рекомендации по лечению. "
                "Опишите симптомы подробнее (минимум 10 слов)."
            )
        return "\n\n".join(sections)

    def run(self, context: WorkflowContext, step_cfg: Dict[str, Any]) -> Dict[str, Any]:
        top_results = context.ranked_candidates[: context.top_k]
        top_match = top_results[0] if top_results else None
        alternatives = top_results[1:] if len(top_results) > 1 else []
        top_confidence = float(
            context.confidence.get("top_protocol_confidence_pct", 0.0)
        )
        target_confidence = float(
            context.confidence.get("target_confidence_pct", 90.0)
        )
        clarification_required = bool(context.clarification.get("required"))
        clarification_questions = context.clarification.get("questions") or []
        answered_count = int(context.clarification.get("answered_count") or 0)

        if not top_match:
            answer = (
                "По вашему описанию не удалось подобрать рекомендации по протоколу. "
                "Опишите жалобы подробнее: симптомы, длительность, температуру, возраст."
            )
        elif clarification_required and clarification_questions:
            answer = self._build_clarification_only_answer(clarification_questions)
        else:
            answer = self._build_clinical_answer(
                top_match=top_match,
            )

        if context.safety.get("warning"):
            answer = f"{answer}\n{context.safety['warning']}"

        context.assistant_answer = answer
        return {
            "top_match_id": top_match["id"] if top_match else None,
            "alternatives": len(alternatives),
            "top_confidence_pct": round(top_confidence, 1),
            "clarification_required": clarification_required,
            "clarification_answered_count": answered_count,
            "answer_preview": answer[:220],
        }


class AgentWorkflowEngine:
    """Executes configured agent steps in sequence."""

    def __init__(
        self,
        search_backend: SearchBackend,
        workflow_path: Optional[Path] = None,
    ) -> None:
        self.search_backend = search_backend
        self.workflow = self._load_workflow(workflow_path)
        self.agents = {
            "intake": IntakeAgent(),
            "retrieval": RetrievalAgent(search_backend),
            "ranking": RankingAgent(),
            "clarification": ClarificationAgent(),
            "safety": SafetyAgent(),
            "response": ResponseAgent(search_backend),
        }

    @staticmethod
    def _default_workflow() -> Dict[str, Any]:
        return {
            "name": "medstandartkz-protocol-agentic-workflow",
            "version": "1.0.0",
            "steps": [
                {"id": "intake", "agent": "intake", "enabled": True},
                {
                    "id": "retrieval",
                    "agent": "retrieval",
                    "enabled": True,
                    "params": {
                        "candidate_pool_multiplier": 8,
                        "max_expansion_terms": 12,
                        "max_search_variants": 4,
                        "term_only_queries": 2,
                        "min_pool_size": 50,
                    },
                },
                {"id": "ranking", "agent": "ranking", "enabled": True},
                {
                    "id": "clarification",
                    "agent": "clarification",
                    "enabled": True,
                    "params": {
                        "target_confidence_pct": 90,
                        "max_candidates": 6,
                        "max_questions": 5,
                        "confidence_temperature": 0.08,
                        "yes_bonus": 0.16,
                        "no_penalty": 0.16,
                        "inverse_bonus": 0.05,
                    },
                },
                {"id": "safety", "agent": "safety", "enabled": True},
                {"id": "response", "agent": "response", "enabled": True},
            ],
        }

    def _load_workflow(self, workflow_path: Optional[Path]) -> Dict[str, Any]:
        if not workflow_path:
            return self._default_workflow()
        if not workflow_path.exists():
            return self._default_workflow()
        with workflow_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict) or "steps" not in payload:
            return self._default_workflow()
        return payload

    def run(
        self,
        query: str,
        top_k: int = 3,
        include_trace: bool = True,
        clarification_answers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        top_k = max(1, min(int(top_k), 10))
        context = WorkflowContext(
            query=query,
            top_k=top_k,
            clarification_answers=dict(clarification_answers or {}),
        )
        trace: List[Dict[str, Any]] = []
        started = time.time()

        for step in self.workflow.get("steps", []):
            if not step.get("enabled", True):
                continue
            step_id = str(step.get("id", "unknown"))
            agent_name = str(step.get("agent", step_id))
            agent = self.agents.get(agent_name)
            if not agent:
                trace.append(
                    {
                        "step": step_id,
                        "agent": agent_name,
                        "status": "error",
                        "details": {"error": f"Agent not found: {agent_name}"},
                    }
                )
                break

            try:
                details = agent.run(context, step)
                trace.append(
                    {
                        "step": step_id,
                        "agent": agent_name,
                        "status": "ok",
                        "details": details,
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive guardrail
                trace.append(
                    {
                        "step": step_id,
                        "agent": agent_name,
                        "status": "error",
                        "details": {"error": str(exc)},
                    }
                )
                break

        top_results = context.ranked_candidates[: context.top_k]
        elapsed_ms = int((time.time() - started) * 1000)
        response = {
            "workflow": {
                "name": self.workflow.get("name", "agentic-workflow"),
                "version": self.workflow.get("version", "1.0.0"),
                "elapsed_ms": elapsed_ms,
            },
            "query": context.query,
            "normalized_query": context.normalized_query or context.query.strip(),
            "top_match": top_results[0] if top_results else None,
            "alternatives": top_results[1:] if len(top_results) > 1 else [],
            "results": top_results,
            "total_results": len(top_results),
            "assistant_answer": context.assistant_answer,
            "confidence": context.confidence,
            "clarification": context.clarification,
            "safety": context.safety,
        }
        if include_trace:
            response["workflow_trace"] = trace
        return response
