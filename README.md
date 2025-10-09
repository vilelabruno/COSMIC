# COSMIC

Um pacote Python para análise de turbulência magnética inspirado nos dados da missão **Cluster** da ESA.  
O projeto nasceu como suporte a um TCC e agora foi organizado como uma biblioteca reutilizável, com testes automatizados e uma API clara para ser compartilhada em portfólios profissionais.

## Principais recursos
- Leitura e pré-processamento de arquivos `.dat` e CEF comprimindo a lógica usada nos notebooks originais.
- Cálculo de gradientes e densidades de corrente via método do curlômetro.
- Métricas auxiliares para estudos de turbulência (PVI, volatilidade magnética, suavização com kernel gaussiano).
- Limpeza e normalização robusta de séries temporais com detecção de outliers, energia magnética e `resampling` em uma linha.
- Diagnósticos espectrais (PSD total/paralelo/perpendicular), correlações temporal e espacial e funções de estrutura de alta ordem.
- MVA, análise de timing multi-ponto, estimativa de curvatura das linhas de campo, helicidade e fatores de qualidade do tetraedro.
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

# Limpeza, energia magnética e reamostragem
clean_df = analyzer.remove_outliers(df1, threshold=3.0)
energy_density = analyzer.calculate_magnetic_energy_density(df1)
normalized = analyzer.normalize_magnetic_field(df1)
coarse = analyzer.resample_dataframe(df1, "1min")

# Espectro, correlações e estatísticas de ordem superior
psd = analyzer.power_spectral_density(mod_B, sample_frequency_hz=22.0, slope_range=(0.1, 1.0))
spectra = analyzer.component_power_spectra(df1[analyzer.config.magnetic_columns], sample_frequency_hz=22.0)
lags, autocorr, decor = analyzer.autocorrelation(mod_B, max_lag=200)
structure = analyzer.structure_functions(mod_B, orders=(2, 4), lags=(5, 20, 50))

# Gradientes e geometria multi-ponto
divB = analyzer.calculate_divergence(df1, df2, df3, df4)
curvature = analyzer.magnetic_curvature_and_radius(df1, df2, df3, df4)
helicity = analyzer.current_helicity_components(df1, df2, df3, df4)
quality = analyzer.tetrahedron_quality_metrics(df1, df2, df3, df4)
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

## Próximos passos sugeridos
- Adicionar um pipeline de CI (GitHub Actions) para rodar os testes automaticamente.
- Publicar o pacote no TestPyPI/PyPI.
- Incluir scripts para baixar e preparar datasets da missão Cluster.
