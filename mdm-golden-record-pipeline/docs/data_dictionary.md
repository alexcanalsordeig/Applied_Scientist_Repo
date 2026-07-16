# Data Dictionary & MDM Design Notes

This document describes the three source systems, the deliberate data-quality
problems embedded in them, and what a correct golden record should look like.
It exists so every design decision in the pipeline is traceable and defensible.

---

## The three source systems

Each source represents a real internal system that holds its own version of the
same client company. Crucially, **they share no common key** — matching must be
derived from the data itself. They also use *different schemas* for the same
concepts, which is what makes this a realistic MDM problem.

### 1. `source_crm.csv` — Sales CRM (Salesforce-like)
The sales team's view. Owns the sales relationship and contact.

| Column | Meaning | Notes |
|---|---|---|
| crm_id | Source system key | Only unique *within* CRM |
| account_name | Company name as sales typed it | Free-text, inconsistent |
| industry | Industry label | CRM's own taxonomy |
| country | Country | Mixed formats: `USA`, `US`, `United States` |
| city | City | Sometimes localised (`München`) |
| employees | Employee count | Sales estimate |
| annual_revenue_usd | Revenue in **USD** | Currency A |
| primary_contact_email | Main contact | — |
| last_modified | Last edit date | ISO `YYYY-MM-DD` |

### 2. `source_marketing.csv` — Marketing / Events
Conference and webinar registrations. High volume, low data discipline.

| Column | Meaning | Notes |
|---|---|---|
| lead_id | Source system key | Only unique within Marketing |
| company | Company name as attendee typed it | Often abbreviated (`Globex`, `Hooli`) |
| sector | Industry-ish label | *Different taxonomy* to CRM (`Tech` vs `Technology`) |
| country | Country | Mixed formats again |
| email_domain | Corporate domain | Useful secondary match signal |
| num_attendees | People at the event | **NOT** employee count — do not confuse |
| signup_date | Registration date | **European `DD/MM/YYYY`** — different format to CRM |

### 3. `source_billing.csv` — Billing / Finance
The system of record for money. Highest data quality; uses legal entity names.

| Column | Meaning | Notes |
|---|---|---|
| customer_no | Source system key | Only unique within Billing |
| legal_name | Registered legal name | `Acme Corp.`, `Globex Industries GmbH` |
| billing_country | Country | — |
| billing_city | City | — |
| headcount | Employee count | Finance's number, often differs from CRM |
| arr_eur | Annual recurring revenue in **EUR** | Currency B — conflicts with CRM's USD |
| vat_id | Tax ID | **Only Billing has this** → authoritative field |
| invoice_date | Last invoice date | Most recent activity signal |

---

## The deliberate MDM challenges (and how each should resolve)

Each of these maps to a named MDM concept. In interviews, point to the concept.

1. **Name variants → normalisation + fuzzy matching.**
   `Acme Corporation` (CRM) / `ACME Corp` (Marketing) / `Acme Corp.` (Billing)
   are one company. Must be standardised then matched despite spelling.

2. **Intra-source duplicate → deduplication.**
   CRM has `CRM002 Globex Industries` and `CRM003 globex industries ` (trailing
   space, lowercase, localised city `München`, missing revenue). Same company,
   entered twice. Must collapse to one before cross-source matching.

3. **Similar but DIFFERENT → match precision.**
   `Stark Manufacturing` (CRM006, Chicago, Manufacturing) and `Stark Industries`
   (CRM007, Los Angeles, Aerospace) are **different companies**. A naive match on
   "Stark" would wrongly merge them. The pipeline must NOT merge these.

4. **Parent / subsidiary → hierarchy, not merge.**
   `Acme Corporation` (parent) and `Acme Cloud Services` (subsidiary, separate
   city Seattle, own VAT ID, own domain) must be kept as **two golden records,
   linked** by a parent-child relationship — never merged into one.

5. **Conflicting values → survivorship rules.**
   Employees vs headcount disagree (Acme: CRM 5000 vs Billing 5200; Globex 3200
   vs 3350). A survivorship rule decides the winner (e.g. *Billing is
   authoritative for headcount*, or *most-recently-updated source wins*).

6. **Currency conflict → survivorship + normalisation.**
   Revenue exists as USD (CRM) and EUR ARR (Billing) — different currencies AND
   different meanings. Can't blindly pick; needs an explicit rule and a note.

7. **Missing values → coalesce / fill.**
   CRM Umbrella revenue is blank; Billing has it. Golden record fills the gap
   from the source that has the value.

8. **Country format drift → standardisation.**
   `USA` / `US` / `United States` → one canonical value.

9. **Date format drift → standardisation.**
   CRM `YYYY-MM-DD` vs Marketing `DD/MM/YYYY` → parse both to one format.

10. **Authoritative single-source field.**
    `vat_id` exists only in Billing → Billing is the source of truth for it.

11. **Single-source entity.**
    `Pied Piper` appears only in Marketing → golden record built from one source.

---

## Expected golden records (the "truth" to validate against)

After the pipeline runs, these distinct companies should survive (12 total):

| Golden entity | Sourced from | Key resolution point |
|---|---|---|
| Acme Corporation | CRM+MKT+Billing | 3 name variants merged; headcount 5200 (Billing) |
| Acme Cloud Services | MKT+Billing | Kept separate; linked as child of Acme Corporation |
| Globex Industries | CRM(x2)+MKT+Billing | Intra-CRM dup collapsed; legal name from Billing |
| Initech LLC | CRM+MKT+Billing | `initech` / `Initech LLC` merged |
| Umbrella Health Group | CRM+MKT+Billing | Missing CRM revenue filled from Billing |
| Stark Manufacturing | CRM+Billing | NOT merged with Stark Industries |
| Stark Industries | CRM only | Distinct company, single source |
| Wayne Enterprises | CRM+MKT+Billing | headcount 46000 (Billing, most recent) |
| Soylent Foods | MKT+Billing | No CRM record |
| Hooli | CRM+MKT | `Hooli Inc` / `Hooli` merged |
| Pied Piper | MKT only | Single-source golden record |
| Vehement Capital Partners | CRM+Billing | `Vehement Capital` / `...Partners` merged |

If the pipeline produces exactly these 12 (with Acme Cloud linked to Acme),
matching precision and recall are both correct.
