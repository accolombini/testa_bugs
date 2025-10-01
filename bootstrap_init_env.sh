#!/usr/bin/env bash
# Script para inicializar estrutura de pacotes Python

echo "Inicializando estrutura do projeto..."

# Criar arquivos __init__.py nas pastas src se não existirem
touch src/__init__.py
touch src/analysis/__init__.py
touch src/io/__init__.py
touch src/util/__init__.py

echo "Arquivos __init__.py criados."

# Criar .env com PYTHONPATH
echo "PYTHONPATH=./src" > .env
echo "Arquivo .env criado."

# Criar .env.zsh.sample com export
echo "export PYTHONPATH=./src" > .env.zsh.sample
echo "Arquivo .env.zsh.sample criado."

# Criar README_PATCH.md com instruções
cat > README_PATCH.md << 'EOF'
# Configuração do Ambiente

## Arquivos criados:
- `src/__init__.py`, `src/analysis/__init__.py`, `src/io/__init__.py`, `src/util/__init__.py`
- `.env` com `PYTHONPATH=./src`
- `.env.zsh.sample` com `export PYTHONPATH=./src`

## Como usar:

### Para usar o .env:
```bash
source .env
```

### Para usar com zsh:
```bash
source .env.zsh.sample
```

### Ou adicionar ao seu .zshrc:
```bash
echo "export PYTHONPATH=./src" >> ~/.zshrc
```
EOF

echo "README_PATCH.md criado com instruções."
echo "Estrutura do projeto inicializada com sucesso!"
