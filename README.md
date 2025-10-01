# TESTA_BUGS — Ambiente de Testes de Sistemas (Dados, Estatísticas, Clustering e Plotly)

> **Objetivo**: prover um _template_ sólido para **testes de sistemas** que envolvam leitura de bases (`.csv`, `.txt`, `.xlsx`), geração de **estatísticas descritivas**, **gráficos interativos (Plotly)**, análise de **correlação**, **curva do cotovelo (K-Means)** e **visualização de clusters** (PCA 2D).  
> Este README é **inicial** e será atualizado conforme novas funcionalidades forem sendo adicionadas.

---

## ✅ Requisitos
- **Python 3.13+**
- `pip` e `venv`/`virtualenv` disponíveis
- Dependências listadas em [`requirements.txt`](requirements.txt)

Sugerido (ambiente isolado):
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

> Dica: para evitar problemas de import ao rodar diretamente do repositório, exporte o `PYTHONPATH`:
> ```bash
> export PYTHONPATH="$PWD/src:$PYTHONPATH"
> ```

---

## 📦 Estrutura de Pastas

```
TESTA_BUGS/
├─ config/
│  └─ config.yaml              # caminhos e ajustes padrão (plotting & clustering)
├─ datasets/                   # arquivos de entrada (.csv/.txt/.xlsx) → NÃO versionados
├─ outputs/
│  ├─ figures/                 # gráficos (HTML/PNG) gerados
│  └─ data/                    # tabelas/CSVs gerados
├─ scripts/
│  └─ dev/
│     ├─ generate_synthetic.py # gera dataset sintético p/ testes
│     └─ run_pipeline.py       # pipeline completo p/ analisar uma base
├─ src/
│  ├─ analysis/                # estatísticas, clustering, métricas
│  ├─ io/                      # detecção de tipo + loaders (.csv/.txt/.xlsx)
│  └─ util/                    # utilidades (logger, paths, etc.)
├─ tests/                      # testes automatizados
├─ .gitignore
├─ README.md
└─ requirements.txt
```

> **Observação**: Se desejar tratar `src` como pacote, inclua `__init__.py` em `src/`, `src/analysis`, `src/io` e `src/util`. Alternativamente, mantenha o `PYTHONPATH` configurado conforme acima.

---

## ⚙️ Configuração (config/config.yaml)

```yaml
paths:
  datasets_dir: "datasets"
  outputs_data_dir: "outputs/data"
  outputs_figures_dir: "outputs/figures"

plotting:
  save_png: true        # requer kaleido para PNG; HTML sempre gerado
  html_interactive: true
  bins: 50

clustering:
  k_min: 1
  k_max: 10
  random_state: 42
  n_init: 10
```

---

## 🚀 Como Usar

### 1) Rodar o pipeline numa base existente
Coloque o arquivo em `datasets/` (ex.: `datasets/meus_dados.csv`) e execute:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"  # se ainda não fez
python scripts/dev/run_pipeline.py --input datasets/meus_dados.csv
# TXT com separador desconhecido:   --sep auto
# Excel com planilha específica:     --sheet Plan1
# Config alternativa:                 --config config/config.yaml
```

**Saídas**:
- **Tabelas**: `outputs/data/descriptive_stats.csv` + `dataset_head.csv`
- **Figuras**: `outputs/figures/`
  - Histogramas por feature (`hist_*.html`/`.png`)
  - Boxplot (`boxplot_all_features.*`)
  - Heatmap de correlação (`correlation_heatmap.*`)
  - Curva do cotovelo (`elbow_curve.*`)
  - Dispersão PCA(2) com clusters no **k recomendado** (`clusters_k{K}.*`)

> PNG depende de `kaleido`; HTML sempre é gerado.

### 2) Gerar uma base sintética para testes
```bash
python scripts/dev/generate_synthetic.py
# → grava datasets/synthetic.csv
```

---

## 🧠 Leitura de Arquivos & Detecção
- **CSV/TXT**: separador **inferido** automaticamente (`,`, `;`, `\t`, `|`) com `csv.Sniffer` e _fallback_ por frequência.  
  - Você pode forçar: `--sep ','`, `--sep ';'`, etc.
- **XLSX**: leitura com `pandas.read_excel`; indique planilha com `--sheet` (nome ou índice).
- **Encoding**: tenta `utf-8` e `latin1` para `.csv/.txt` (ajuste no loader se necessário).

---

## 🧪 Testes
```bash
pytest -q
```
- Cobrir leituras mínimas de `.csv`, `.txt` e `.xlsx` (ampliaremos conforme evoluir o pipeline).

---

## 🧰 Qualidade (opcional)
- **Ruff** (lint): `ruff check src`
- **Black** (format): `black src scripts tests`
- **Mypy** (types): `mypy src`

---

## 📌 Convenções
- **Dados brutos**: **não versionar** (`datasets/` já está no `.gitignore` — preservamos `datasets/.keep`).
- **Artefatos**: **não versionar** (`outputs/` ignorado; preservamos `outputs/*/.keep`).
- **Nomenclatura**: `snake_case` para arquivos, módulos e colunas.
- **Documentação**: manter este `README.md` e adotar `CHANGELOG.md` ao iniciar versionamento semântico.

---

## 🗺️ Roadmap (rápido)
- [ ] Pré-processamento (NA, normalização, one-hot) configurável.
- [ ] Métricas de cluster (silhouette, calinski-harabasz, davies-bouldin).
- [ ] Exportação Parquet/Feather (`pyarrow`).
- [ ] Orquestração (Makefile/Prefect) e CI (GitHub Actions).
- [ ] Validação de schema com `pandera`.
- [ ] Perfis de ambiente (dev/stage/prod) no `config/`.

---

## 📄 Licença
Definir --> ACC MIT.
