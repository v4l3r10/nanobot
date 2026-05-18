# Design — Memoria "Wiki-Tree" stile Obsidian per nanobot

- **Data:** 2026-05-18
- **Stato:** design validato (brainstorming concluso, 6/6 sezioni approvate)
- **Autore driver:** valerio.cavagni@gmail.com
- **Documenti correlati:**
  - `REPORT_memoria_nanobot_vs_openhuman.md` (confronto e raccomandazioni P1–P8, H1–H3)
  - `MECCANICHE_MEMORIA_openhuman.md` (riferimento esaustivo openhuman)
  - Karpathy "LLM Wiki" gist: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

---

## 0. Contesto e motivazione

Dal confronto nanobot (v0.2.0) vs openhuman emerge che il gap di nanobot **non è lo storage** ma il *retrieval* (pura recency, nessuna semantica), la *latenza* (conoscenza long-term solo via Dream batch ~2h) e l'**assenza di isolamento multi-utente**.

Difetti di openhuman da **non** replicare: crescita monotona illimitata senza oblio (nessun TTL/prune), scan vettoriale lineare, `learning.enabled=false` di default.

**Decisione architetturale:** non adottare embeddings né SQLite. Si mantiene Markdown + git + Dream come *verità curata*, e sopra si costruisce un **wiki navigabile stile Obsidian** (wikilink `[[ ]]` + MOC) con **oblio esplicito hot/cold**. La struttura segue i principi del gist "LLM Wiki" di Karpathy: 3 layer (raw sources / wiki mantenuto dall'LLM / schema), 3 operazioni (Ingest, Query, Lint), pagine essenzialmente piatte con `index.md` + `log.md`, Lint che cura contraddizioni / stale / orphan / missing-link, e l'LLM che fa il bookkeeping.

---

## 1. Architettura e layout su disco

Vault **per-utente**, derivato dalla session key (`channel:chat_id` → `user_slug` fs-safe). Questo risolve la raccomandazione **P3** (namespacing per-utente).

```
workspace/
  SOUL.md                       # GLOBAL — identità del bot
  AGENTS.md / TOOLS.md          # globali
  memory/
    wiki/SCHEMA.md              # schema master globale (curato a mano, versionato)
    users/<user_slug>/          # user_slug fs-safe da channel:chat_id
      MEMORY.md                 # root MOC → [[people/_index]] [[projects/_index]] …  (SEMPRE nel prompt)
      USER.md                   # profilo per-utente (fix P3)
      history.jsonl             # = log.md di Karpathy, per-utente
      wiki/
        SCHEMA.md               # copia dello schema master
        people/    _index.md  alice.md …
        projects/  _index.md  payment-svc.md …
        concepts/  _index.md  auth-model.md …
        decisions/ _index.md  YYYY-<slug>.md …
        .cold/<type>/…          # pagine decadute, fuori dal MOC, leggibili da tool
        .lint.log               # audit append-only delle run di Dream-Lint
  .git/                         # dulwich versiona tutto memory/
```

- `unified_session=true` → unico vault `users/unified/` (retro-compatibilità con il comportamento attuale).
- **Mapping Karpathy:** raw sources = session JSONL; `log.md` = `history.jsonl`; `index.md` = `MEMORY.md` (MOC radice); wiki = `wiki/*.md`; schema = `SCHEMA.md`.
- **Filing per tipo** (stabile), *non* per Area/topic (gerarchia di cartelle scoraggiata da Karpathy). La gerarchia di conoscenza vive nel **link graph**, non nelle cartelle.
- **Frontmatter per pagina:** `type, title, status (hot|cold), created, updated, last_touched, tags[], links_out[], pinned (opzionale)`.

---

## 2. SCHEMA.md — layer dichiarativo

Schema master globale curato a mano e versionato in git; copia per-vault al bootstrap. Definisce:

- I **tipi** ammessi (`people`, `projects`, `concepts`, `decisions`, estendibili) e la cartella di filing per ciascuno.
- `cold_after_days` per tipo (default: people 180, projects 90, concepts 365, decisions = mai).
- Campi frontmatter **obbligatori** e regole di naming degli slug.
- Cap dimensione del MOC (numero massimo di righe/voci nel digest radice).
- Regole di filing (a quale tipo assegnare un nuovo contenuto).

Dream-Lint legge SCHEMA per validare e curare. Cambiare politica di oblio o aggiungere un tipo = editare SCHEMA, non il codice.

---

## 3. Percorso di scrittura (dual pen)

Due penne scrivono sul wiki:

**a) Tool agente `wiki_note`** — operazioni `read(path)`, `create(type, slug, …)`, `append(path, text)`, `search(query)`. L'agente può **creare/append pagine foglia** e aggiungere lo **stub-link nel `_index.md`** del tipo. Non può: spostare in `.cold/`, fare merge di duplicati, riscrivere `_index.md`/MOC radice (operazioni **solo-Dream**). Ogni write è **validato contro SCHEMA**: frontmatter malformato o tipo errato → rifiuto con errore chiaro, così il modello si auto-corregge.

**b) Dream-Lint (batch, cadenza = `dream.interval_h`)** — fase **Ingest**: da `history.jsonl` estrae fatti, crea/aggiorna pagine, annota contraddizioni (senza sovrascrivere), aggiorna `last_touched`.

Hardening: **scritture atomiche** (`tmp` + `os.replace`) → fix **H1**; **file-lock per-vault** → fix **H2**. Dual-write significa che la conoscenza è disponibile **subito** via tool dell'agente, mentre Dream consolida dopo: questo mitiga la latenza **P2** senza eliminare Dream.

---

## 4. Percorso di lettura

- Il **MOC** (`MEMORY.md` per-utente) è **sempre nel prompt**, ma piccolo: link ai `_index` per tipo + un digest dei recent-hot. Sostituisce l'iniezione del `MEMORY.md` monolitico attuale e accorcia il replay delle ultime-50 di `history.jsonl` a una coda recente minima.
- Navigazione **model-driven** via tool `wiki_note.read` seguendo i wikilink (niente embeddings).
- Operazione `search` (keyword/tag/recency su **hot + cold**) per la discoverability di pagine non linkate dal MOC.
- **Reheat-on-read:** una lettura aggiorna `last_touched`; se la pagina era `cold` torna `status: hot`; il prossimo Lint la rimuove fisicamente da `.cold/`.
  - **Hand-off reheat / `.cold/` (contratto pinnato per il Lint, milestone 4.3):**
    - Leggere una pagina il cui path è sotto `.cold/` la reidrata **in place** (`status: cold`→`hot`, `last_touched`=oggi) lasciandola **fisicamente** in `.cold/`.
    - È compito del **Lint** (fase del Dream, milestone 4.3) **ricollocare fisicamente fuori** da `.cold/` ogni pagina con `status: hot` che si trova ancora sotto un componente `.cold` (chiave di rilocazione: `status == "hot" and ".cold" in path.parts`).
    - `append` invece **rifiuta** i path sotto `.cold/` (le pagine fredde si riattivano solo via `read`); questa asimmetria *read-reheats / append-refuses* è voluta.

Questo rimpiazza la raccomandazione **P1** (indice semantico) con navigazione esplicita wikilink + MOC + search keyword — scelta deliberata: niente vector store.

---

## 5. Oblio hot/cold + Lint

- `cold_after_days` per tipo da SCHEMA; superata la soglia di inattività (`last_touched`), Lint sposta la pagina in `.cold/<type>/`, fuori dal MOC ma ancora leggibile dal tool.
- `pinned: true` → immune al decay.
- **Nessun hard-delete automatico.** `.cold/` è il pavimento; git conserva la storia; solo un **merge** (dedup) o una cancellazione manuale rimuovono davvero una pagina. Cap su `.cold/` = **YAGNI, rimandato**.
- **Lint** (i 4 check di Karpathy + dedup): contraddizioni (registrate, non sovrascritte), stale → cold, orphan (pagina non raggiungibile dal MOC), missing/broken link (via `links_out`), dedup/merge di pagine duplicate. Poi **rigenera** `_index.md` di ogni tipo e il `MEMORY.md` radice **dal frontmatter** (deterministico).
- `.lint.log` = audit append-only di ogni run (cosa è stato raffreddato, unito, riparato).

Questo realizza l'**admission gate** (**P7**: segnali cheap + soglia, niente embeddings) e il **Dream gerarchico/per-topic** (**P8**: Lint opera per tipo).

---

## 6. Aggancio al codice nanobot, edge case, testing

**Punti d'innesto (riferimenti indicativi, nanobot v0.2.0):**

- `agent/context.py:37-76` + `:63-71` — iniettare il MOC per-utente invece del `MEMORY.md` monolitico; ridurre il replay ultime-50 a una coda recente piccola.
- `agent/memory.py:205-226` + `MemoryStore` — path per-utente; `MEMORY.md` diventa artefatto generato; fix scritture atomiche (H1) qui e nel tool.
- `agent/memory.py:785-1087` (`Dream`) + `templates/agent/dream_phase{1,2}.md` — Dream → Ingest+Lint guidati da SCHEMA.
- `session/manager.py` — derivare `user_slug` dalla session key; rispettare `unified_session`.
- **Nuovo tool `wiki_note`** in `agent/tools/`, sandboxed al workspace come `ReadFileTool`/`EditFileTool`, registrato nella tool registry.
- `utils/gitstore.py:45-391` — estendere i path tracciati a `memory/users/**` (così `/dream-log` e `/dream-restore` coprono il wiki).
- `config/schema.py` — knob: enable wiki, `cold_after_days` default (override da SCHEMA), toggle vault per-utente (legato a `unified_session`), cadenza Lint = `dream.interval_h`.

**Edge case:** migrazione one-time di `MEMORY.md`/`USER.md`/`history.jsonl` legacy → vault per-utente (primo Dream bootstrappa il wiki dal MEMORY.md esistente); wiki disabilitato → comportamento identico a oggi (retro-compat); pagina malformata → tool valida contro SCHEMA e rifiuta, Lint ripara comunque; collisione slug su reheat → Lint risolve via merge/suffix; lock per-vault, Dream salta un vault occupato e riprova; backlog Dream (P2) → le tool-write tengono la conoscenza disponibile, trigger opportunistico opzionale.

**Testing:**
- *Unit:* parser/validatore SCHEMA; round-trip frontmatter; predicato decay; detector orphan/broken-link; dedup/merge; rigenerazione deterministica MOC/`_index`; atomic write; derivazione `user_slug`.
- *Integration:* turno → `wiki_note.create` → turno dopo lo vede via navigazione; Dream Ingest da `history.jsonl` sintetico integra+linka; decay → `.cold`; reheat via search+read; contraddizione registrata non sovrascritta; commit git post-Dream contiene l'albero atteso.
- *Isolamento multi-utente:* due slug non si contaminano; `unified_session` collassa a un vault.
- *Golden regression:* "wiki off = nanobot attuale".
- *Migrazione:* workspace legacy → vault migrato senza perdita dati, storia git preservata.

---

## 7. Mappatura raccomandazioni del report

| Raccomandazione | Come è indirizzata in questo design |
|---|---|
| **P1** indice semantico | Sostituito da navigazione wikilink + MOC + `search` keyword (scelta esplicita: niente embeddings) |
| **P2** latenza Dream | Mitigata dal dual-write: tool agente rende la conoscenza disponibile subito |
| **P3** namespacing per-utente | Vault `users/<user_slug>/` + `USER.md` per-utente |
| **P7** admission gate | Validazione SCHEMA al write + admission in Lint (segnali cheap) |
| **P8** Dream gerarchico/per-topic | Lint opera per tipo, rigenera `_index` per tipo |
| **H1** scritture atomiche | `tmp` + `os.replace` nel tool e in `memory.py` |
| **H2** file-lock | Lock per-vault; Dream salta vault occupati |

---

## 8. Cosa NON facciamo (YAGNI)

- Niente embeddings / vector store / SQLite.
- Niente cap o eviction fisica su `.cold/` (rimandato).
- Niente hard-delete automatico.
- Niente gerarchia di cartelle per Aree/topic (la gerarchia vive nel link graph; il filing fisico resta per tipo, stabile).

---

## 9. Questioni aperte

- **Posizione/versionamento di questo documento:** `F:\dev\AI\NanoBot_openHuman` non è un repo git; il clone `nanobot/` è un repo di terze parti (HKUDS/nanobot) — non opportuno committarvi. Da decidere: lasciare il design al root del workspace insieme agli altri artefatti, o inizializzare un repo git dedicato per il workspace.
- Dettaglio del formato `search` (ranking keyword/tag/recency) da fissare in fase di plan implementativo.
