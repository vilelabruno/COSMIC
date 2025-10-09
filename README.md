# COSMIC

Um pacote Python para análise de turbulência magnética inspirado nos dados da missão **Cluster** da ESA.  
O projeto nasceu como suporte a um TCC e agora foi organizado como uma biblioteca reutilizável, com testes automatizados e uma API clara para ser compartilhada em portfólios profissionais.

## Principais recursos
- Leitura e pré-processamento de arquivos `.dat` e CEF comprimindo a lógica usada nos notebooks originais.
- Cálculo de gradientes e densidades de corrente via método do curlômetro.
- Métricas auxiliares para estudos de turbulência (PVI, volatilidade magnética, suavização com kernel gaussiano).
- Ferramentas de análise de valores extremos (POT, declustering e visualizações associadas).
- Testes automatizados em `pytest`, garantindo a reprodutibilidade dos notebooks.

## Instalação
```bash
git clone https://github.com/<usuario>/COSMIC.git
cd COSMIC
python3 -m pip install -e .
```

Dependências principais:
| Biblioteca | Uso |  
|------------|-----|  
| `pandas`, `numpy` | Estruturas de dados e álgebra vetorial |  
| `scipy` | Ajustes estatísticos (POT) e filtros |  
| `matplotlib` | Visualização dos resultados |  

Para o ambiente de desenvolvimento, instale também:
```bash
python3 -m pip install -r requirements-dev.txt
```

## Guia rápido
```python
from cosmic import CosmicAnalyzer

analyzer = CosmicAnalyzer()
df1 = analyzer.ler_arquivo_dat("data/C1.dat")
df2 = analyzer.ler_arquivo_dat("data/C2.dat")
df3 = analyzer.ler_arquivo_dat("data/C3.dat")
df4 = analyzer.ler_arquivo_dat("data/C4.dat")

# Diferenças de campo magnético e posição
B_diff = analyzer.calculate_B_diff(df1, df2)
r_diff = analyzer.calculate_r_diff(df1, df2)

# Densidade de corrente e curlômetro
J123 = analyzer.calculate_current_density(df1, df2, df3)
curl = analyzer.curlometer(df1, df2, df3, df4)

# PVI e detecção de current sheets
mod_B = analyzer.calculate_mod_B(df1, *analyzer.config.magnetic_columns)
pvi = analyzer.calculate_PVI(mod_B, tau=66)
cs_events = analyzer.limethod(df1[analyzer.config.magnetic_columns], theta_c=45.0, tau_sec=5.0)
```

As rotinas de visualização aceitam um objeto `Axes` opcional e evitam chamar `plt.show()` automaticamente em ambientes de teste.

## Estrutura do repositório
```
cosmic/
├── __init__.py
├── cosmic.py          # API principal (CosmicAnalyzer)
├── utils.py           # Espaço para utilidades adicionais
tests/
├── test_cosmic.py     # Testes automatizados para toda a API pública
setup.py
pyproject.toml
requirements-dev.txt
```

## Executando os testes
```bash
python3 -m pytest
```
Os testes criam dados sintéticos e exercitam os fluxos completos de leitura, análise e visualização.  
Caso esteja em um ambiente sem `pytest`, instale as dependências apontadas em `requirements-dev.txt`.

## Boas práticas adotadas
- Código tipado, orientado a objetos e com docstrings.
- Alias de compatibilidade (`cosmic`) para não quebrar notebooks existentes.
- Testes automatizados que replicam os passos dos notebooks.
- Documentação clara e pronta para ser exibida em portfólios / LinkedIn.

## Próximos passos sugeridos
- Adicionar um pipeline de CI (GitHub Actions) para rodar os testes automaticamente.
- Publicar o pacote no TestPyPI/PyPI.
- Incluir scripts para baixar e preparar datasets da missão Cluster.

Sinta-se à vontade para abrir *issues* ou enviar *pull requests*. Este repositório foi preparado com carinho para demonstrar boas práticas em ciência de dados, tecnologia e programação.
