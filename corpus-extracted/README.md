# Corpus Index Specification: `corpus-extracted/`

[`index.json`](index.json) is the corpus inventory: one record per collection. This document defines the schema, the value rules, and the directory conventions those records follow.

---

## 1. Overview & Directory Architecture

Each ingested document produces one extraction directory and one record. The identifier (`id`) names all of them:

| Layer | Path | Contents |
|---|---|---|
| Source | `corpus-raw/<id>.pdf` | The ingested PDF. |
| Extracted text | `corpus-extracted/<id>/hybrid_txt/<id>.md` | Markdown produced by the pipeline. |
| Extraction artifacts | `corpus-extracted/<id>/hybrid_txt/<id>_*` | MinerU layout and model output; [`MINERU.md`](../MINERU.md) §3 lists the files. |
| Figures | `corpus-extracted/<id>/hybrid_txt/images/` | Figure and table crops. |
| Metadata | `corpus-extracted/index.json` | One record per collection. |

### 1.1 Naming

Form: `<surname-or-acronym>-<year>-<topic>`, lowercase, hyphen-separated. Institutional authors use the agency acronym, as in `psa-2024-flemms-media-exposure`. `<topic>` names the work's subject, dataset, or method, and it is what keeps the key unique when a surname and year repeat: `lin-2023-libmtl` and `lin-2023-dual-balancing-multi` are two papers by the same author in the same year.

### 1.2 Renaming

Changing an `id` means changing three things: the PDF in `corpus-raw/`, the extraction directory, and the record. The extraction loop in [`MINERU.md`](../MINERU.md) §7.1 derives its output path from the PDF stem, so an `id` that disagrees with its PDF stem makes the loop treat the document as unextracted and produce a second directory.

---

## 2. Record Schema

`index.json` is a JSON array of objects. Each object has this shape:

```json
{
  "id": "alam-2021-humaid-human-annotated",
  "title": "HumAID: Human-Annotated Disaster Incidents Data from Twitter with Deep Learning Benchmarks",
  "authors": [
    "Firoj Alam",
    "Umair Qazi",
    "Muhammad Imran",
    "Ferda Ofli"
  ],
  "year": 2021,
  "philippine_coverage": "none",
  "doi": "10.48550/arXiv.2104.03090",
  "alternate_id": null,
  "summary": "Social networks are widely used for information consumption and dissemination, especially during time-critical events such as natural disasters. Despite its significantly large volume, social media content is often too noisy for direct use in any application."
}
```

---

## 3. Data Dictionary

| Field | Type | Nullable | Constraints & Format | Description |
|---|---|---|---|---|
| `id` | `string` | No | Matches `^[a-z0-9]+-[0-9]{4}(-[a-z0-9]+)+$`. | Corpus key; see §1.1. Must equal the PDF stem and the directory name (§5.4). |
| `title` | `string` | No | As printed, including stylization such as `calamanCy`; no trailing period. | Document title. |
| `authors` | `string[]` | No | At least one element; never a bare string. Given names then surname, as printed (`Lester James V. Miranda`). | Authors in printed order. Institutional authors carry the acronym, e.g. `Philippine Statistics Authority (PSA)`. |
| `year` | `integer` | No | Four digits; must equal the year inside `id` (§5.3). | Publication year. |
| `philippine_coverage` | `string` | No | One of `full`, `partial`, `none`. | Subject classification; see §4.1. |
| `doi` | `string` | Yes | Bare DOI starting with `10.`, e.g. `10.48550/arXiv.2104.03090`; no URI prefix. | DOI, where the publisher registers one. See §4.2. |
| `alternate_id` | `string` | Yes | `<namespace>:<identifier>`, with the namespace from §4.2. | Registry identifier, used where no DOI exists or as a second pointer. |
| `summary` | `string` | No | Non-empty; machine-generated. | Search text for triage. See the warning below. |

> [!WARNING]
> A `summary` is machine-generated and has not been checked against the source. Read the collection's `.md` or the PDF in `corpus-raw/` before resting a citation or a number on it.

---

## 4. Value Rules

### 4.1 `philippine_coverage`

Assigned by two questions, asked in order.

```
[document ingested]
        │
        ▼
Q1  Does it mention Philippine entities, languages, regions, or agencies?
        │
        ├── no  ──► none
        └── yes
             │
             ▼
        Q2  Is the Philippines the main focus of the work?
             │
             ├── yes ──► full
             └── no  ──► partial
```

- **Search terms (illustrative, matched case-insensitively).** National and regional: `Philippines`, `Philippine`, `Filipino`, `Luzon`, `Visayas`, `Mindanao`, `Manila`. Languages: `Tagalog`, `Taglish`, `Cebuano`, `Hiligaynon`, `Ilocano`, `Bisaya`, `Waray`, `Bikol`. Agencies: `PAGASA`, `DSWD`, `DROMIC`, `NDRRMC`, `PSA`, `COA`, `NPC`. Typhoon names count as well.
- **Do not count.** A venue line in a reference entry (`Conference held in Manila, Philippines`) or a bibliography citation matches the search without making the document about the Philippines. Position is not evidence either: MinerU extracts tables and appendices after the reference heading, so a match late in the `.md` can still be body content.

### 4.2 Identifiers

- **`doi` (primary).** Record the bare DOI, without an `https://doi.org/` prefix. Check that a new DOI resolves before the record is added: `https://doi.org/api/handles/<doi>` returns `responseCode: 1` for a handle that exists.
- **`alternate_id` (fallback).** Used where the venue registers no DOI, or as a second pointer where one exists: `loshchilov-2019-decoupled-weight-decay` carries both, because the paper has an arXiv DOI and an OpenReview entry. The prefix names the registry that resolves the rest of the string.

| Namespace | Platform | Resolves as |
|---|---|---|
| `acl:` | ACL Anthology | `https://aclanthology.org/<identifier>/` |
| `jmlr:` | Journal of Machine Learning Research | `https://jmlr.org/papers/<identifier>.html` |
| `openreview:` | OpenReview | `https://openreview.net/forum?id=<identifier>` |
| `hdl:` | Handle System | `https://hdl.handle.net/<identifier>` |

- **No identifier.** Agency reports, gray literature, and NGO documents with no registry entry set both fields to `null`; cite them by publishing body and title.

---

## 5. Ingestion & Validation Invariants

A document is ingested by placing `corpus-raw/<id>.pdf` and running the extraction loop in [`MINERU.md`](../MINERU.md) §7.1, which skips anything already extracted. The record added to `index.json` must satisfy:

1. **Unique keys.** No `id` appears twice.
2. **Unique DOIs.** No non-null `doi` appears twice.
3. **Year agreement.** The year inside `id` equals the `year` field.
4. **Filesystem parity.** Both `corpus-raw/<id>.pdf` and `corpus-extracted/<id>/hybrid_txt/<id>.md` exist.
5. **Schema conformance.** Every field matches the type declared in §3, and `authors` holds at least one string.

Papers known to be missing from the index are tracked in [`missing-references.md`](missing-references.md).
