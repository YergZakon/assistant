#!/usr/bin/env python3
"""Generate an extended evaluation query set for protocol assistant accuracy tests."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List


WORD_RE = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ]+")

AGES = [3, 5, 8, 12, 17, 24, 31, 39, 48, 56, 67]
ADULT_AGES = [18, 24, 31, 39, 48, 56, 67]
PEDIATRIC_AGES = [3, 5, 8, 12, 17]
DURATIONS = ["2 дня", "3 дня", "5 дней", "1 неделя", "2 недели", "1 месяц"]
FEVERS = ["температура 37.8", "температура 38.2", "температура 38.8", "высокая температура 39"]


DOMAIN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "respiratory": {
        "count": 16,
        "keywords": ["орви", "грипп", "фаринг", "тонзил", "бронх", "астма", "пневмон", "туберкул", "ларинг"],
        "question_ids": ["rhinitis", "fever", "dyspnea", "wheeze", "tonsillitis", "cough", "chest_pain", "tb_constitutional"],
        "templates": [
            "Пациент {age} лет, {dur} кашель, боль в горле, {fever}, насморк, слабость, без выраженной одышки, МКБ J06.9.",
            "Пациент {age} лет, ночные приступы удушья, свистящее дыхание, сухой кашель, аллергия в анамнезе, температура не повышена.",
            "Пациент {age} лет, {dur} кашель с мокротой, {fever}, боль в грудной клетке при вдохе, озноб, слабость.",
            "Пациент {age} лет, длительный кашель более 3 недель, ночная потливость, снижение массы тела, субфебрилитет, утомляемость.",
        ],
    },
    "gastro": {
        "count": 16,
        "keywords": ["гастр", "панкреат", "холецист", "гепат", "энтер", "колит", "гастроэнтер", "желч"],
        "question_ids": ["abdominal_pain", "ruq_pain", "vomiting", "diarrhea", "jaundice", "epigastric_pain", "fever", "nausea"],
        "templates": [
            "Пациент {age} лет, схваткообразная боль в животе, диарея до 6 раз в сутки, {fever}, тошнота, сухость во рту, слабость.",
            "Пациент {age} лет, боль в правом подреберье после жирной пищи, тошнота, горечь во рту, эпизоды рвоты, субфебрилитет.",
            "Пациент {age} лет, сильная боль в эпигастрии с иррадиацией в спину, многократная рвота, тошнота, ухудшение после алкоголя.",
            "Пациент {age} лет, желтушность кожи и склер, темная моча, тяжесть в правом подреберье, слабость, снижение аппетита.",
        ],
    },
    "urinary": {
        "count": 16,
        "keywords": ["цистит", "пиелонеф", "моч", "уретр", "почек"],
        "question_ids": ["dysuria", "fever", "abdominal_pain", "vomiting", "diarrhea"],
        "templates": [
            "Пациент {age} лет, жжение при мочеиспускании, частые позывы, боль внизу живота, {fever}, мутная моча.",
            "Пациент {age} лет, боль в пояснице справа, {fever}, озноб, частое болезненное мочеиспускание, слабость.",
            "Пациент {age} лет, учащенное мочеиспускание, рези, дискомфорт над лоном, субфебрилитет, слабость и снижение аппетита.",
            "Пациент {age} лет, тянущая боль в пояснице, дизурия, озноб, {fever}, ощущение разбитости.",
        ],
    },
    "cardio": {
        "count": 14,
        "ages": ADULT_AGES,
        "keywords": ["сердечн", "карди", "ишем", "аритм", "гипертенз", "недостаточ"],
        "question_ids": ["dyspnea", "chest_pain", "edema", "fever", "syncope"],
        "templates": [
            "Пациент {age} лет, одышка в покое, отеки голеней к вечеру, слабость, тяжесть в груди, утомляемость.",
            "Пациент {age} лет, давящая боль за грудиной при нагрузке, одышка, холодный пот, выраженная слабость.",
            "Пациент {age} лет, перебои в сердце, эпизоды сердцебиения, одышка при ходьбе, головокружение, слабость.",
            "Пациент {age} лет, нарастающая одышка ночью, ортопноэ, отеки ног, снижение толерантности к нагрузке.",
        ],
    },
    "neuro": {
        "count": 16,
        "keywords": ["инсульт", "невр", "эпилеп", "менинг", "энцефал", "кровоизлияни"],
        "question_ids": ["neuro_focal", "headache", "seizure", "syncope", "fever", "trauma", "bleeding"],
        "templates": [
            "Пациент {age} лет, внезапная слабость в правой руке, нарушение речи, асимметрия лица, сильная головная боль, тошнота.",
            "Пациент {age} лет, эпизод судорог, сонливость после приступа, слабость, однократная рвота, субфебрилитет.",
            "Пациент {age} лет, резкая головная боль, рвота, светобоязнь, ригидность мышц шеи, повышение температуры.",
            "Пациент {age} лет, кратковременная потеря сознания, дезориентация, головная боль, неврологический дефицит.",
        ],
    },
    "trauma": {
        "count": 14,
        "keywords": ["травм", "перелом", "ушиб", "череп", "сотряс", "рана"],
        "question_ids": ["trauma", "bleeding", "headache", "syncope", "vomiting", "chest_pain"],
        "templates": [
            "После падения у пациента {age} лет сильная боль в голени, отек, деформация, невозможно наступить на ногу, была травма.",
            "После ДТП у пациента {age} лет ушиб головы, головная боль, тошнота, однократная рвота, кратковременная потеря сознания.",
            "После удара в грудную клетку у пациента {age} лет боль при дыхании, отек, усиливается при движении, подозрение на перелом ребра.",
            "Резаная рана предплечья, кровотечение, боль, отек тканей, ограничение движений в конечности, травма бытовая.",
        ],
    },
    "obgyn": {
        "count": 14,
        "ages": ADULT_AGES,
        "keywords": ["гинек", "эндометрит", "аднекс", "воспалительн", "матк", "тазов"],
        "question_ids": ["vaginal_discharge", "abdominal_pain", "fever", "bleeding", "nausea"],
        "templates": [
            "Пациентка {age} лет, боль внизу живота, патологические выделения, {fever}, дискомфорт при половом акте.",
            "Пациентка {age} лет, тянущие боли в малом тазу, лихорадка, гнойные выделения, слабость, снижение аппетита.",
            "Пациентка {age} лет, боли внизу живота после менструации, повышение температуры, болезненность при осмотре.",
            "Пациентка {age} лет, острые боли внизу живота, патологические выделения, озноб, субфебрилитет, дискомфорт при мочеиспускании.",
        ],
    },
    "infectious": {
        "count": 14,
        "ages": PEDIATRIC_AGES,
        "keywords": ["корь", "скарлат", "мононуклеоз", "краснух", "инфекц", "дифтер", "ветрян"],
        "question_ids": ["fever", "rash", "tonsillitis", "cough", "diarrhea", "vomiting"],
        "templates": [
            "Пациент {age} лет, высокая температура, боль в горле, выраженная слабость, увеличенные лимфоузлы, налет на миндалинах.",
            "Пациент {age} лет, сыпь на коже, лихорадка, боль в горле, снижение аппетита, общая интоксикация.",
            "Ребенок {age} лет, температура 39, сыпь по туловищу, кашель, насморк, конъюнктивит, вялость.",
            "Пациент {age} лет, лихорадка, диарея, рвота, боли в животе, выраженная слабость, признаки обезвоживания.",
        ],
    },
}


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def _load_base_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict) and str(row.get("query", "")).strip():
                rows.append(row)
    return rows


def build_extended_rows(
    base_rows: List[Dict[str, Any]],
    seed: int,
    min_words: int,
) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)
    rows = list(base_rows)
    existing_queries = {str(row.get("query", "")).strip() for row in rows}

    for domain, cfg in DOMAIN_CONFIGS.items():
        target = int(cfg["count"])
        templates = list(cfg["templates"])
        keywords = [str(x) for x in cfg["keywords"]]
        question_ids = [str(x) for x in cfg["question_ids"]]
        age_pool = [int(x) for x in cfg.get("ages", AGES)]

        created = 0
        idx = 0
        while created < target:
            template = templates[idx % len(templates)]
            age = age_pool[(idx + rnd.randint(0, 3)) % len(age_pool)]
            dur = DURATIONS[(idx + rnd.randint(0, 5)) % len(DURATIONS)]
            fever = FEVERS[(idx + rnd.randint(0, 3)) % len(FEVERS)]
            query = template.format(age=age, dur=dur, fever=fever)

            query = " ".join(query.split())
            idx += 1
            if query in existing_queries:
                continue
            if count_words(query) < min_words:
                continue

            row = {
                "query": query,
                "expected_title_keywords": keywords,
                "expected_domain": domain,
                "expected_question_ids": question_ids,
            }
            rows.append(row)
            existing_queries.add(query)
            created += 1
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate extended eval query set")
    parser.add_argument(
        "--base",
        default="docs/agentic_eval_queries.jsonl",
        help="Base JSONL with seed cases",
    )
    parser.add_argument(
        "--output",
        default="docs/agentic_eval_queries_extended.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-words", type=int, default=10, help="Minimum words per query")
    args = parser.parse_args()

    base_path = Path(args.base).resolve()
    out_path = Path(args.output).resolve()
    base_rows = _load_base_rows(base_path)
    rows = build_extended_rows(
        base_rows=base_rows,
        seed=int(args.seed),
        min_words=max(1, int(args.min_words)),
    )
    write_jsonl(out_path, rows)

    domain_counts: Dict[str, int] = {}
    for row in rows:
        domain = str(row.get("expected_domain", "unknown")).strip().lower() or "unknown"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    print(
        json.dumps(
            {
                "output": str(out_path),
                "total_cases": len(rows),
                "domain_counts": domain_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
