# cosmic

Um pacote Python para análise de dados da missão cluster ESA.

## Instalação
'''
python3 setup.py sdist

pip3 install dist/cosmic
'''
## Funcionalidades


 Função     def ler_arquivo_dat(nome_arquivo):

## Uso


 Função     def ler_arquivo_dat(nome_arquivo):

Para utilizar a função `ler_arquivo_dat`, é necessário passar o nome do arquivo que deseja ler como argumento. A função irá retornar um DataFrame do pandas que contém o conteúdo do arquivo lido.

Exemplo de uso:

```python
df = ler_arquivo_dat('meu_arquivo.dat')
```
Onde 'meu_arquivo.dat' é o arquivo que você deseja ler.
 Função     def calculate_B_diff(df1, df2):

### Guia de utilização da função `calculate_B_diff`

Para executar a função `calculate_B_diff`, você precisará passar dois argumentos:
* `df1` - sera o primeiro data frame que representa o conjunto de dados do primeiro satélite.
* `df2` - será o segundo data frame que representa o conjunto de dados do segundo satélite.

A função irá retornar a diferença de campo magnético entre os dois satélites.

Exemplo de uso:

```python
import pandas as pd

# Carregue seu dado em df1 e df2
df1 = pd.read_csv('Caminho para seu arquivo /satellite_1.csv')
df2 = pd.read_csv('Caminho para seu arquivo /satellite_2.csv')

# Calcule a diferença de campo magnético
B_diff = calculate_B_diff(df1, df2)

print(B_diff)
```

Certifique-se que seus data frames contêm as colunas necessárias ('B_vec_xyz_gse__C1_CP_FGM_FULL1', 'B_vec_xyz_gse__C1_CP_FGM_FULL2', 'B_vec_xyz_gse__C1_CP_FGM_FULL3') para que a função funcione corretamente.
 Função     def calculate_r_diff(df1, df2):

Para usar a função `calculate_r_diff(df1, df2)`, é necessário ter dois dataframes `df1` e `df2`, que correspondem aos dados dos dois satélites que você deseja medir a distância entre eles. Esses dataframes devem conter as seguintes colunas: 'sc_pos_xyz_gse__C1_CP_FGM_FULL1', 'sc_pos_xyz_gse__C1_CP_FGM_FULL2' e 'sc_pos_xyz_gse__C1_CP_FGM_FULL3', que representam as coordenadas tridimensionais de cada satélite. Simplesmente passe seus dataframes como argumentos de entrada ao chamar a função: `calculate_r_diff(df1, df2)`, e ela retornará um valor que representa a distância entre os dois satélites.
 Função     def calculate_current_density(df1, df2, df3):

Para utilizar a função `calculate_current_density(df1, df2, df3)`, você precisa fornecer três DataFrames como argumentos (df1, df2, df3). Esses DataFrames devem conter dados que serão usados para calcular a densidade de corrente. A função retorna uma série de pandas com a densidade de corrente calculada para cada linha dos DataFrames fornecidos.

Por exemplo: 

```python
Jijk = calculate_current_density(df1, df2, df3)
``` 

Onde `Jijk` é a variável onde o resultado final será armazenado e `df1`, `df2` e `df3` são os DataFrames que você forneceu como argumentos.
 Função     def curlometer(spacecraft1, spacecraft2, spacecraft3, spacecraft4):

Para utilizar a função `curlometer`, forneça as medidas dos quatro veículos espaciais. A função vai retornar a densidade atual calculada pelo curlômetro. Siga o exemplo abaixo para usá-la:

```python
density_result = curlometer(spacecraft1, spacecraft2, spacecraft3, spacecraft4)
```

Note que `spacecraft1`, `spacecraft2`, `spacecraft3` e `spacecraft4` devem ser substituídos pelas medidas reais dos veículos espaciais. O resultado, `density_result`, é a densidade de corrente calculada.
 Função     def calculate_mod_B(df, Bx_column, By_column, Bz_column):

Para usar a função `calculate_mod_B`, você precisará fornecer quatro argumentos. O primeiro argumento (`df`) é um dataframe do pandas de onde os dados serão recebidos. Os próximos três argumentos (`Bx_column`, `By_column`, `Bz_column`) são as strings que correspondem aos nomes das colunas no dataframe que contém seus dados Bx, By e Bz, respectivamente. 

A função retornará uma serie do pandas que representa o módulo de B (mod_B). 

Exemplo de uso:

```python
df = pd.read_csv('dados.csv')
Bx_column = 'Bx'
By_column = 'By'
Bz_column = 'Bz'
mod_B = calculate_mod_B(df, Bx_column, By_column, Bz_column)
``` 

Isso irá retornar a série `mod_B` calculada a partir dos dados nas colunas Bx, By, Bz do dataframe do pandas `df`.
 Função     def plot_mod_B(yy):

## Como usar a função plot_mod_B(yy)

A função `plot_mod_B(yy)` serve para traçar um gráfico simples de uma sequência de dados. Para usá-la, siga estes passos:

1. Certifique-se de que a biblioteca `matplotlib.pyplot` esteja importada em seu código, pois é necessário para a criação do gráfico.
2. Prepare a sequência de dados (`yy`) que você gostaria de visualizar em um gráfico. Esta sequência pode ser uma lista simples ou um array do Numpy, por exemplo.
3. Chame a função `plot_mod_B(yy)`, passando sua sequência de dados como argumento.

Aqui está um exemplo de uso:

```python
import matplotlib.pyplot as plt

# sua sequência de dados
dados = [1, 2, 3, 4, 5]

# chame a função
plot_mod_B(dados)
```

Após chamar a função, um gráfico simples será exibido, traçando seus dados na coordenada `Y` em função da sua posição na sequência em `X`.
 Função     def calculate_PVI(x, tau=66):

Para utilizar a função `calculate_PVI`, forneça uma série pandas como o primeiro argumento. Opcionalmente, você pode fornecer um segundo argumento, `tau`, que é um inteiro, default é 66.

Aqui está um exemplo de como usar essa função:

```python
import pandas as pd

# Criando uma série pandas
x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Chamando a função calculate_PVI
PVI = calculate_PVI(x)

# Imprimindo a resposta
print(PVI)
```

A função retorna outra série pandas que representa o PVI calculado. Você pode então utilizar este resultado para a análise desejada.

Lembre-se também que esta função depende do `pandas` e do `numpy`, então certifique-se de ter esses pacotes instalados e importados em seu script.
 Função     def plot_data(PVI):

# Como usar a função `plot_data(PVI)`

Para usar a função `plot_data(PVI)`, basta tomar as seguintes etapas:

1. Importe o módulo necessário executando: `import matplotlib.pyplot as plt`.
2. Defina ou importe seus dados para uma variável que represente `PVI`.
3. Chame a função `plot_data(PVI)`.
4. A função irá gerar um gráfico de linha dos seus dados.

Exemplo de uso:

```python
import matplotlib.pyplot as plt

# Suponha que seus dados são a seguinte lista:
PVI = [10, 15, 7, 10, 13]

# Para plotar esses dados, simplesmente chame a função:
plot_data(PVI)
```

Isso irá exibir um gráfico em linha dos seus dados.
 Função     def norm(vector):

# Guia de Utilização da função "norm"

Esta função, chamada `norm`, é usado para calcular a norma de um vetor. Abaixo estão os passos sobre como usar esta função.

## Como usar

### Passo 1: Importe a biblioteca Numpy
Para usar a função `norm`, você precisa primeiramente importar a biblioteca Numpy. Caso não a tenha, você pode instalar esta biblioteca via pip, da seguinte maneira:

```
pip install numpy
```

### Passo 2: Crie/Obtenha o vetor
Você precisa criar ou obter um vetor ao qual deseja aplicar a função. Este vetor deve ser um array ou uma lista de números.

### Passo 3: Use a função norm
Chame a função norm e passe o vetor como argumento. Por exemplo:

```python
import numpy as np

def norm(vector):
    return np.sqrt(np.sum(np.square(vector)))

vetor = np.array([1, 2, 3, 4, 5])
normvetor = norm(vetor)

print(normvetor)
```

A função `norm` retornará a norma do vetor, que é um número único.

## Importante
Assegure-se que o tipo de dado passado como argumento é compatível, a função `norm` espera receber um vetor de elementos que possam ser manipulados matematicamente.
 Função     def dot(v1, v2):

# Utilização

A função `dot(v1, v2)` é usada para calcular o produto escalar (dot product) de dois vetores. 

Você só precisa passar dois vetores como argumentos na função. Os vetores podem ser listas, tuplas ou arrays do numpy. 

O produto escalar será retornado.

Exemplo de uso:

```python
import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

resultado = dot(v1, v2)
print(resultado)  # Saída: 32
```
 Função     def angle(v1, v2):

Para usar a função `angle(v1, v2)`, passe dois vetores `v1` e `v2` como argumentos. A função vai retornar o ângulo, em graus, entre esses dois vetores. 

Os vetores (`v1`, `v2`) devem ser passados na forma de listas, tuplas ou arrays do Numpy. 

Exemplo de uso:

```python
import numpy as np

# defina os dois vetores
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])

# chame a função passando os vetores como argumentos
angle = angle(v1, v2)

print(f'O ângulo entre v1 e v2 é {angle} graus.')
```
 Função     def cs_detection(df, tau, theta_c):

# Uso de `cs_detection(df, tau, theta_c)`

Essa função é usada para detectar mudanças em uma série de dados, especificamente analisando a diferença de ângulos entre intervalos consecutivos de observações. Os parâmetros necessários são:

- `df`: um DataFrame do pandas que contém a série de dados que você deseja analisar. 
- `tau`: o tamanho do intervalo que a função deve considerar ao analisar a série. Por exemplo, se `tau` é 10, a função irá comparar os ângulos entre os primeiros 10 pontos e os 10 próximos. 
- `theta_c`: é o valor mínimo do ângulo que será considerado como mudança. 

A função retorna 1 se a frequência de ângulos maiores ou iguais a `theta_c` for maior que 15%, ou 0 caso contrário. Em outras palavras, retorna 1 se detectar mudanças significativas na série de dados e 0 se não. 

Exemplo de uso:

```python
import pandas as pd

dados = pd.DataFrame({ 'valores': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] })
mudanca = cs_detection(dados, 5, 0.8)
```
Neste exemplo, a função irá retornar se houve uma mudança significativa nos ângulos dos primeiros 5 pontos em comparação com os 5 pontos seguintes, considerando como mudança ângulos maiores ou iguais a 0.8.
 Função     def limethod(df, theta_c = 35.0, tau_sec = 10):

Para utilizar a função `limethod`, você precisa passar um DataFrame (pandas.DataFrame) para o parâmetro `df`, o qual contém seus dados, além de opcionais: `theta_c` (um float que define um limite, padrão é 35.0) e `tau_sec` (um inteiro que especifica o tempo em segundos, padrão é 10). 

Exemplo de uso:

```python
import pandas as pd

# seu DataFrame
df = pd.read_csv("seu_arquivo.csv")

# chamando a função limethod 
resultado = limethod(df, theta_c = 50.0, tau_sec = 5)
```

Depois disso, `resultado` será um novo DataFrame contendo índices de tempo e as saídas do detector CS.
 Função     def convert_to_float(s):

Para usar a função `convert_to_float(s)`, você precisa passar uma string como argumento. Essa função tentará convertê-lo em um número de tipo float. Se a string passada contiver uma 'D', ela será substituída por 'E' para facilitar a conversão para float. Por exemplo:

```python
print(convert_to_float("1234D56"))
```

O código acima imprimirá `1234E56` como um número float. 

Por favor, esteja ciente de que se a string fornecida não puder ser convertida em float, será lançada uma ValueError.
 Função     def calculate_magnetic_volatility(df, B, tau=50, w=50):

Para usar a função `calculate_magnetic_volatility`, você precisa passar os seguintes parâmetros: 

- `df`: um DataFrame do pandas com seus dados 
- `B`: uma string com o nome da coluna que contém o campo magnético 
- `tau`: um número inteiro opcional para o valor de τ (padrão é 50) 
- `w`: um número inteiro opcional para o tamanho da janela para calcular a volatilidade magnética (padrão é 50) 

A função retornará uma série do pandas com a volatilidade magnética calculada. 

Aqui está um exemplo de como você pode usar essa função: 

```python
df_vol_mag = calculate_magnetic_volatility(df, 'campo_magnetico', tau=100, w=30)
```

Neste exemplo, estamos calculando a volatilidade magnética de um DataFrame chamado `df` que tem uma coluna `'campo_magnetico'`, com um valor τ de 100 e um tamanho de janela de 30.
 Função     def apply_gaussian_kernel(x_coords, sigma):

## Como Utilizar a Função apply_gaussian_kernel

Esta função é usada para aplicar um filtro gaussiano em uma lista de coordenadas.

Aqui está um exemplo simples de como usar:

```python
x_coords = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sigma = 1.0

smoothed_coords = apply_gaussian_kernel(x_coords, sigma)
```

Neste exemplo, `x_coords` é a lista de coordenadas que você deseja suavizar e `sigma` é o parâmetro de desvio padrão para o filtro gaussiano. O resultado será armazenado na variável `smoothed_coords`.

Modifique os valores de acordo com as necessidades do seu projeto.
 Função     def declustering_function(data, u=30000, run=10):

# Como usar a função `declustering_function`

A função `declustering_function` é usada para aplicar a técnica de declustering em um conjunto de dados.

### Sintaxe

A função `declustering_function` toma três argumentos:

```python 
declustering_function(data, u=30000, run=10)
```

### Parâmetros

1. `data`: A entrada de dados na qual a técnica de declustering será aplicada. Isso deve ser fornecido como uma estrutura de dados do Python, como um DataFrame ou uma lista.

2. `u`(opcional): Este é um parâmetro de ajuste usado na técnica de declustering. O valor padrão é suficiente na maioria dos casos, mas pode ser ajustado para experimentação ou refinamento.

3. `run`(opcional): Outro parâmetro de ajuste usado na técnica de declustering. Podem ser necessárias várias execuções (run) para completar o processo. O valor padrão é 10, mas isso pode ser ajustado conforme necessário.

## Exemplo de uso

Suponha que você tenha dados hospedados em um DataFrame do Pandas chamado `my_data`. Para aplicar a função de declustering a esses dados, você chamaria a função como abaixo:

```python
declustering_function(my_data, u=30000, run=10)
```

A função então retornará os dados resultantes após aplicar o processo de declustering.
 Função         # Threshold and run are parameters for the function with default values

Para usar esta função, forneça os parâmetros de *threshold* e *run*, que possuem valores padrão, assim como seus dados na variável `data`. A função divide seus dados em "Peaks Over Threshold" (POT) e "Values Below Threshold" (VBT). 

Após essa divisão, a função calcula os POT desagrupados - aqueles com um intervalo maior que o valor de *run*. Em seguida, um gráfico é criado plotando points abaixo e acima do threshold, além dos pontos desagrupados. 

Finalmente, a função retorna os POT desagrupados e o número total de clusters é impresso.

Por exemplo:

```python
import pandas as pd

# Substitute your data here
data = pd.read_csv('your_file.csv')

# Call the function with default threshold and run parameters
result = your_function(data)

# The output will be the declustered POT values and the total number of clusters is printed.
``` 

Note: certifique-se de que a variável 'value', que a função usa para cálculos e gráficos, esteja presente em seus dados.
 Função     def stats_excess(data, threshold):

# Guia de utilização para a função stats_excess()

A função `stats_excess()` é usado para calcular a média e o desvio padrão de valores que excedem um determinado limite em um conjunto de dados. 

## Como usar?

1. **Parâmetros de entrada**

    A função `stats_excess()` requer dois parâmetros:

    * `data` - É a lista, array ou série de dados numéricos nos quais você deseja calcular a média e o desvio padrão de excesso.

    * `threshold` - É o valor de limite que você deseja usar para determinar os valores excessivos.

2. **Chamando a função**

    Para usar a função, é simplesmente necessário chamar a função stats_excess com os parâmetros necessários. Por exemplo:

        mean_excess, std_excess = stats_excess(data, threshold)
     
3. **Resultado da função**

    A função retornará a média (`mean_excess`) e o desvio padrão (`std_excess`) dos valores que excedem o limite fornecido.

Lembre-se de importar a biblioteca `numpy` como `np` antes de usar esta função, uma vez que ela utiliza funções `np.mean()` e `np.std()`. 

Exemplo de uso da função:

```python
    import numpy as np

    data = np.array([1,2, 3, 4, 5, 6, 7, 8, 9, 10])
    threshold = 5
  
    # Invocando a função stats_excess
    mean_excess, std_excess = stats_excess(data, threshold)
  
    print("Média do excesso: ", mean_excess)
    print("Desvio Padrão do excesso: ", std_excess)
```
Estes passos devem ajudá-lo a usar `stats_excess()` na sua análise de dados.
 Função     def plot_mean_excess(data, min_thresh, max_thresh, num_threshs=100):

Para utilizar a função `plot_mean_excess` você precisa fornecer pelo menos três parâmetros: `data`, `min_thresh` e `max_thresh`. 

`data`: esse é o seu conjunto de dados, no qual a função irá operar. 

`min_thresh` e `max_thresh`: esses são seus limiares mínimo e máximo. A função irá gerar automaticamente 100 limiares entre esses valores para calcular o excesso médio e seu desvio padrão. 

Se desejar criar mais ou menos do que 100 limiares, você pode fazer isso fornecendo um quarto parâmetro, `num_threshs`.

A função irá gerar um gráfico com o excesso médio para cada limiar, junto com barras de erro representando o desvio padrão. O eixo x representará os limiares e o eixo y o excesso médio. O gráfico será exibido automaticamente.
 Função     def fit_pot_model(data, min_threshold, max_threshold, num_thresholds):

# Como usar a função fit_pot_model

A função `fit_pot_model` é usada para ajustar um modelo Peaks Over Threshold (POT) a uma série de dados, dentro de um intervalo de limiares.

Segue um exemplo de como você pode usar essa função:

1. Primeiro, importe as bibliotecas necessárias:

   ```python
   import pandas as pd
   import numpy as np
   ```

2. Prepare seus dados e certifique-se de que eles são um modelo pandas Series ou DataFrame.

3. Usando a função `fit_pot_model`, forneça os seguintes parâmetros:

    - `data`: Esta é a sua série ou DataFrame.
    - `min_threshold`: Este é o valor mínimo do limiar.
    - `max_threshold`: Este é o valor máximo do limiar.
    - `num_thresholds`: Este é o número de limiares.

    Por exemplo:

   ```python
   result = fit_pot_model(data, min_threshold=10, max_threshold=100, num_thresholds=10)
   ```

4. A função retornará um DataFrame com os parâmetros ajustados do modelo POT para cada limiar. Você pode visualizar o resultado utilizando:

   ```python
   print(result)
   ```

Isso é tudo o que você precisa para usar a função `fit_pot_model` em seus projetos!

 Função     def plot_shape_parameter(results):

# Utilização da função plot_shape_parameter

Para utilizar a função `plot_shape_parameter`, você precisa de um DataFrame que foi retornado pela função `fit_pot_model`. Este DataFrame deve conter os parâmetros ajustados do modelo 'Peaks Over Threshold' (POT) para cada limite.

Aqui está um exemplo de como a função `plot_shape_parameter` pode ser chamada:

```python
# Primeiro execute a função 'fit_pot_model' para obter o DataFrame 'results'
results = fit_pot_model(data)

# Agora você pode plotar o parâmetro 'Shape' com a função 'plot_shape_parameter'
plot_shape_parameter(results)
```

Isso produzirá um gráfico mostrando a evolução do parâmetro 'Shape' em relação aos limites. Seu gráfico será exibido imediatamente por causa da linha `plt.show()`. 

Note que não há um valor de retorno para esta função, pois seu principal objetivo é plotar e visualizar os dados.
 Função     def plot_mean_residual_life(data, thresholds):

Para utilizar essa função, basta seguir as etapas:

1. Prepare seus dados. Essa função espera receber um conjunto de dados numéricos (em formato de lista ou array NumPy, por exemplo) através do parâmetro `data`.

2. Defina seus thresholds. Isso significa decidir sobre quais valores (limites) você deseja analisar nos dados. Novamente, essa informação deve ser fornecida em formato de lista ou array NumPy ao parâmetro `thresholds`.

3. Chame a função com seus dados e thresholds. Por exemplo: `plot_mean_residual_life(meus_dados, meus_thresholds)`. A função irá então gerar o gráfico Mean Residual Life (Média de Vida Residual).

4. Analise os resultados. A função irá exibir um gráfico de pontos onde o eixo X representa os limites (thresholds) e o eixo Y o valor médio do excesso em relação ao limite. Uma linha vertical representando o ponto onde a taxa de variação dos valores médios se aproxima de uma constante também será exibida.

## Documentação da função ler_arquivo_dat

# Entrada

A função ler_arquivo_dat recebe um único parâmetro:

`nome_arquivo`: É uma string que representa o caminho e o nome do arquivo .dat que será lido.

## Processamento

A função primeiro abre e lê todas as linhas do arquivo. Em seguida, o script busca a primeira linha contendo a string "START_VARIABLE" e armazena o valor encontrado após o símbolo de igual como uma variável temporária (`var`).

Após isso, o script analisa o arquivo novamente procurando por linhas que começam com "SIZES". Sempre que encontra uma dessas linhas, o script adiciona um número equivalente ao valor logo após o símbolo de igual em "SIZES" de variáveis à lista `variaveis`, sendo que cada nome de variável é formado pela variável temporária `var` mais o número que representa a ordem dessa variável.

O script também atualiza a variável temporária `var` sempre que encontra uma nova linha que começa com "START_VARIABLE" e interrompe o loop se encontrar uma linha que contém "EOF".

Finalmente, o script lê as linhas restantes do arquivo que contenham exatamente 11 elementos, separados por vírgulas, criando uma lista de listas. Em seguida, ele usa esta lista para criar e retornar um DataFrame do pandas, onde os nomes das colunas são os nomes das variáveis encontradas anteriormente.

## Saída

A saída da função é um DataFrame do pandas. Cada linha do DataFrame corresponde a uma linha do arquivo que continha exatamente 11 elementos separados por vírgulas. As colunas do DataFrame representam as variáveis descritas no arquivo, no formato determinado pelas linhas que começam com "START_VARIABLE" e "SIZES".
 Função     def calculate_B_diff(df1, df2):

Função: calculate_B_diff(df1, df2)

Para usar essa função, é preciso duas entradas. Ambas são quadros de dados (ou dataframes) do pandas. Esses dataframes devem conter três colunas específicas, que contêm valores numéricos que representam o campo magnético medido por dois satélites ("B_vec_xyz_gse__C1_CP_FGM_FULL1", "B_vec_xyz_gse__C1_CP_FGM_FULL2" e "B_vec_xyz_gse__C1_CP_FGM_FULL3").

Entradas:
- df1: pandas DataFrame 
-- Exige que a DataFrame tenha as três colunas seguintes: 'B_vec_xyz_gse__C1_CP_FGM_FULL1', 'B_vec_xyz_gse__C1_CP_FGM_FULL2' e 'B_vec_xyz_gse__C1_CP_FGM_FULL3'
- df2: pandas DataFrame
-- Exige que a DataFrame tenha as três colunas seguintes: 'B_vec_xyz_gse__C1_CP_FGM_FULL1', 'B_vec_xyz_gse__C1_CP_FGM_FULL2' e 'B_vec_xyz_gse__C1_CP_FGM_FULL3'

Processamento:
A função calcula a diferença dos campos magnéticos medidos pelos satélites, que é obtida pela subtração elemento a elemento dos dados contidos nas mesmas colunas correspondentes dos dois DFs.

Saída:
- B_diff: numpy array
-- Um array no formato numpy contendo a diferença dos campos magnéticos dos dois satélites, e é calculada como df1 - df2. O resultado tem o mesmo formato dos dataframes de entrada, sem os rótulos das colunas.
 Função     def calculate_r_diff(df1, df2):

## Documentação da função calculate_r_diff

### Descrição

Esta função é usada para calcular a diferença de distância entre dois satélites.

### Entrada

A função `calculate_r_diff` requer 2 argumentos de entrada:

- df1: Um dataframe que contém as seguintes colunas: 'sc_pos_xyz_gse__C1_CP_FGM_FULL1', 'sc_pos_xyz_gse__C1_CP_FGM_FULL2' e 'sc_pos_xyz_gse__C1_CP_FGM_FULL3'. Estas colunas representam as coordenadas XYZ de um satélite no sistema de referência GSE (Geocentric Solar Ecliptic).

- df2: Outro dataframe que também contém as mesmas colunas representando a posição de um segundo satélite.

### Processamento
A função subtrai as correspondentes coordenadas XYZ do segundo dataframe das do primeiro dataframe. Esta subtração é efetuada para as três componentes do vetor posição (X, Y e Z).

### Saída
A função retorna um array Numpy contendo a diferença de distância entre os dois satélites nas três coordenadas XYZ. Cada linha do array é um vetor tridimensional representando a diferença de posição entre os dois satélites em um determinado ponto no tempo.
 Função     def calculate_current_density(df1, df2, df3):

# Documentação da Função calculate_current_density

A função `calculate_current_density` é usada para calcular a densidade de corrente total usando a equação (2). Ela toma três dataframes como entrada, realiza vários cálculos, e retorna uma série de densidade de corrente.

## Entrada
A função aceita três argumentos de entrada:

nome | descrição
---|---
df1 | Um objeto dataframe do pandas. Cada linha do dataframe representa um vetor.
df2 | Similar ao df1, é um dataframe onde cada linha representa um vetor.
df3 | Similar aos dois anteriores, representa vários vetores.

## Processamento
A função primeiramente calcula as diferenças entre os vetores df1, df2 e df3 respecivamente. Ela armazena essas diferenças em `r12`, `r13` e `r23`.

A função então calcula as diferenças `B` entre os mesmos vetores, armazenando os resultados em `B12`, `B13` e `B23`.

Os resultados armazenados são então usados para calcular a densidade de corrente com a fórmula dada.

## Saída
A função retorna uma Serie do pandas `Jijk` representando a densidade de corrente.

 Função     def curlometer(spacecraft1, spacecraft2, spacecraft3, spacecraft4):

# Documentação da Função Curlometer

## Descrição Geral  

A função curlometer calcula a densidade de corrente usando a equação (2). Isto é realizado através da calculação da densidade da corrente para cada combinação de três entre as quatro naves espaciais fornecidas e depois média destes quatro valores. A raiz quadrada do quadrado desta média é então retornada.

## Entradas

1. spacecraft1: Array de dados para a primeira nave espacial.
2. spacecraft2: Array de dados para a segunda nave espacial.
3. spacecraft3: Array de dados para a terceira nave espacial.
4. spacecraft4: Array de dados para a quarta nave espacial.

Cada um desses parâmetros deve representar vetores tridimensionais de posição no espaço (i.e., [x, y, z]) para uma nave espacial em um dado momento.

## Processamento

1. A densidade de corrente é calculada para cada uma das combinações possíveis de três naves espaciais: J123, J124, J134 e J234.

2. A média destas quatro densidades de corrente é então calculada. 

3. O valor médio é então elevado ao quadrado e finalmente a raiz quadrada deste resultado é tomada. 

## Saídas

Retorna a densidade de corrente calculada usando a equação (2), que é baseada nos vetores de posição das quatro naves espaciais fornecidas.

É importante notar que esta função depende de uma função adicional chamada `calculate_current_density` que necessita ser definida e fornecida com os vetores de posição das naves espaciais. A 'equação (2)' referida na descrição original precisa estar definida ou documentada em algum lugar onde essa função está sendo usada.
 Função     def calculate_mod_B(df, Bx_column, By_column, Bz_column):

Função: calculate_mod_B(df, Bx_column, By_column, Bz_column)

- Descrição: A função calculate_mod_B foi projetada para calcular o módulo B (mod_B), ou a magnitude de B, de um conjunto de dados fornecido em um dataframe pandas. O cálculo é realizado usando as raízes quadradas dos componentes Bx, By e Bz do dataframe.

- Entrada: 
  - df : Um DataFrame do pandas representando o conjunto de dados
  - Bx_column, By_column, Bz_column : Strings (str) que representam os nomes das colunas no dataframe onde estão localizados os dados Bx, By e Bz respectivamente.

- Processamento: 
  - A função first obtém os valores das colunas especificadas Bx, By e Bz do dataframe.
  - Em seguida, calcula o módulo B (mod_B), com base nessas colunas usando a fórmula do vetor magnitude, que é a raiz quadrada da soma dos quadrados dos seus componentes (neste caso, Bx, By, Bz).

- Saída:
  - Retorna uma Série Pandas com os valores calculados do módulo B (mod_B).
  
- Exemplo de uso: 
  ```
  data = pd.DataFrame({'Bx': [1,2,3,4], 'By': [2,3,4,5], 'Bz': [3,4,5,6]})
  mod_B = calculate_mod_B(data, 'Bx', 'By', 'Bz')
  print(mod_B)
  ```
Obs: esta função também pode ser usada em conjunto com uma função de plotagem para visualizar os resultados calculados do módulo B.
 Função     def plot_mod_B(yy):

## Documentação

### Nome da Função
`plot_mod_B`

### Descrição
Esta função é utilizada para gerar um gráfico simples de linhas de um conjunto único de dados.

### Parâmetros
`yy`: Este parâmetro é o conjunto de dados a ser representado no gráfico. A entrada de dados deve ser uma lista, uma série do pandas, um array de numpy ou qualquer outro objeto iterável de Python.

### Processamento
A função `plot_mod_B` cria uma nova figura, traça uma linha baseada nos valores da entrada `yy` e exibe o gráfico utilizando as funções `figure`, `plot`, e `show` respectivamente da biblioteca matplotlib.pyplot.

### Saída
A função não retorna nenhum valor, sua saída é a exibição do gráfico em uma nova janela.

### Exemplo
```python
data = [1, 2, 3, 4, 5]
plot_mod_B(data)
```
Este exemplo traçará uma linha de um ponto a outro da lista de dados. A janela do gráfico será exibida após a chamada da função.

### Requisitos
Esta função depende da biblioteca matplotlib.pyplot. Certifique-se de ter instalado esta biblioteca (`pip install matplotlib`) e de tê-la importado corretamente (`import matplotlib.pyplot as plt`) em seu código antes de usar a função `plot_mod_B`.
 Função     def calculate_PVI(x, tau=66):

**Função: calculate_PVI(x, tau=66)**

**Descrição:** Esta função calcula o PVI (Price and Volume Trend Indicator) a partir de uma série fornecida.

**Entrada**: 
- x: uma série pandas (pandas.Series), que estão os dados para os quais o PVI será calculado. 
  
- tau: um número inteiro (int), padrão é 66. Este é o deslocamento temporal para a diferença.

**Processamento**:

Primeiro, a função assegura que 'x' é uma série do pandas. Em seguida, o tamanho de 'x' é determinado. Um loop é usado para computar 'delta', a diferença entre elementos 'tau' à parte em 'x'.

'delta' então é convertido a uma série de pandas e os seguintes são calculados:

- abs_delta: o valor absoluto de 'delta',
  
- square_delta: cada elemento de 'abs_delta' é elevado ao quadrado,
  
- mean_delta: a média de 'square_delta',
  
- sqrt_delta: a raiz quadrada de 'mean_delta'.

PVI é então calculado como a divisão de 'abs_delta' por 'sqrt_delta'. Por fim, PVI é transposto.

**Saída**:

A função retorna 'PVI', que é uma série pandas (pandas.Series).

**Impressão**:

Esta função não produz qualquer saída impressa ou de plotagem diretamente. A utilidade da biblioteca matplotlib mencionada no final da descrição é fora do escopo desta função e seria usada para plotar a saída se necessário.
 Função     def plot_data(PVI):

Função: plot_data(PVI)

Objetivo: Esta função cria um gráfico visual simples dos dados de entrada fornecidos utilizando a biblioteca matplotlib.pyplot.

Entrada: 
---
A função aceita um parâmetro de entrada:

PVI - Uma lista ou matriz contendo números. PVI poderia ser um conjunto uni-dimensional de dados.

Processamento:
---
1. A função chama `plt.figure()` para criar uma nova figura de gráficos.
2. Posteriormente, `plt.plot(PVI)` é usado para gerar a linha gráfica dos dados de entrada.
3. Por último, `plt.show()` é utilizado para exibir o gráfico plotado na tela.

Saída: 
---
Esta função não retorna um valor. O resultado é a apresentação do gráfico em um novo quadro de figura criado por `plt.figure()`. O gráfico gerado mostra a representação visual dos dados inseridos.

O gráfico pode ser usado para ver tendências ou padrões em seu conjunto de dados.

Exceções: 

- Se o PVI for uma lista vazia ou None, a função não exibirá nenhum gráfico.
- Se o PVI contiver tipos de dados que não sejam números (como strings ou booleanos), a função pode lançar um erro.
 Função     def norm(vector):

---
## Documentação para a função norm()

### Descrição:

A função `norm()` calcula e retorna a norma euclidiana (ou "comprimento") de um vetor.

### Entrada:

A função aceita como entrada um vetor no formato de lista ou array de números (`vector`). Os números podem ser inteiros ou de ponto flutuante (float).

Exemplo:

```python
vector = [1, 2, 3, 4, 5]
```

### Processamento:

A função usa a biblioteca NumPy para realizar as seguintes operações:
1. Elevar ao quadrado cada elemento do vetor (`np.square(vector)`)
2. Somar todos os elementos quadrados do vetor (`np.sum(...)`)
3. Retornar a raiz quadrada da soma anterior (`np.sqrt(...)`)

Essas operações juntas calculam a norma euclidiana do vetor.

### Saída:

A função retorna a norma euclidiana (um número real) do vetor de entrada.

Exemplo:

```python
print(norm([1, 2, 3, 4, 5]))
```

Resultado:

```python
7.416198487095663
```

### Considerações finais:

A função presume que a entrada seja sempre um vetor (lista ou array de números), e não verifica se essa condição é atendida. Assegure-se de fornecer a entrada correta para evitar erros durante a execução.
 Função     def dot(v1, v2):

Nome da Função: dot

Entrada:
- A função aceita dois parâmetros: v1 e v2. Ambos os parâmetros podem ser n-dimensionais e devem ser de tipos numéricos (como inteiros e flutuantes).

Processamento:
- Internamente, a função utiliza a operação 'dot' da biblioteca numpy. Esta operação determina o produto escalar (dot product) dos vetores fornecidos.

Saída:
- A função retorna o resultado do produto escalar dos dois vetores de entrada.

Exceções:
- Se os dois vetores de entrada não forem compatíveis para a operação do produto escalar (por exemplo, os dois vetores têm diferentes dimensões), a função irá gerar um erro.
- A função também irá gerar um erro se algum dos parâmetros do vetor de entrada não for numérico.

Exemplo de uso:
```
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(dot(v1, v2)) # Deve imprimir: 32
```

Esta função é relevante em muitas áreas que utilizam matemática vetorial, como física, engenharia e ciência da computação. Ela é frequentemente usada em operações que envolvem a manipulação de matrizes e vetores, como a multiplicação de matrizes.

 Função     def angle(v1, v2):

# Documentação para a função `angle(v1, v2)`

## Entrada: 

A função `angle` recebe dois parâmetros:

- `v1`: Um vetor (array-like) de números reais.
- `v2`: Um vetor (array-like) de números reais. `v1` e `v2` devem ter a mesma dimensão.

## Processamento:

1. A função calcula a norma (ou magnitude) de cada vetor usando a função `norm` que deve estar definida previamente no escopo da execução. 
   
2. Em seguida, normaliza ambos os vetores (v1 e v2) dividindo cada vetor pela sua respectiva norma. A normalização é feita para obter vetores unitários (com comprimento igual a 1) na mesma direção de v1 e v2.

3. Calcula o produto escalar (ou dot product) dos vetores normalizados usando a função `dot` que deve estar definida previamente no escopo da execução.

4. Finalmente, calcula o ângulo entre os vetores usando a função `arccos` da biblioteca numpy aplicada ao produto escalar. O resultado obtido estará em radianos, então a função converte para graus multiplicando por 180 e dividindo pelo número pi.

## Saída:

A função retorna um número real representando o ângulo (em graus) entre os vetores `v1` e `v2`, cujo valor varia entre 0 e 180. Se os vetores são ortogonais, o ângulo será 90 graus. Se os vetores apontam na mesma direção, o ângulo será 0 grau. Se a direção dos vetores é oposta, o ângulo será 180 graus.
 Função     def cs_detection(df, tau, theta_c):

# Documentação para a função cs_detection

## Descrição:
Esta função realiza um tipo de detecção baseada na comparação de ângulos entre vetores de dados em partes separadas de um DataFrame.

## Entradas:
- df: DataFrame do pandas de entrada. Este deve ser um DataFrame numérico unidimensional.
- tau: inteiro que representa o intervalo de tempo base ou deslocamento que divide os dados em dois blocos.
- theta_c: limiar para o ângulo entre dois vetores adequadamente normalizados.

## Processamento:
A função divide o DataFrame em dois blocos (b1 e b2) com base no valor de tau. Em seguida, para cada elemento nos blocos, calcula o ângulo entre os elementos correspondentes dos dois blocos.

Um contador (cont) é incrementado se o ângulo calculado for maior ou igual ao ângulo limiar (theta_c). Isso é feito para todos os tau elementos em b1 e b2.

Após isso, a frequência de ângulos maiores ou iguais a theta_c é calculada como a razão entre cont e tau. Se essa frequência é maior ou igual a um limiar pré-determinado (ff_c), a função retorna 1, caso contrário, retorna 0. 

## Saídas:
- out: Inteiro que é 1 se a frequência de ângulos acima do limiar é maior ou igual ao limite ff_c. Caso contrário, o valor de saída é 0. 

## Nota:
A função assume que o valor de tau é menor ou igual ao comprimento do DataFrame df/2. Caso contrário, um erro pode ocorrer. Deve-se observar também que a função angle não é definida na documentação, então ela deve ser implementada ou importada de uma biblioteca anteriormente.
 Função     def limethod(df, theta_c = 35.0, tau_sec = 10):

# Documentação da Função 'limethod'

## Entrada

- `df`: dataframe do pandas. Requer uma sequência ordenada de dados numéricos.
- `theta_c`: um real, valor padrão é 35.0. Parâmetro que é utilizado no método de detecção `cs_detection`. Geralmente, serve para definir alguma espécie de limite ou "corte" no processamento e/ou análise dos dados.
- `tau_sec`: um real, valor padrão é 10. Isso define a largura da janela de tempo em segundos para processar os dados.

## Processamento

A função `limethod` primeiro realiza a conversão da largura da janela de tempo em timesteps (conversão realizada pela divisão de 1 por 22 para obter `dt`, e a multiplicação de 22 por `tau_sec` para obter `tau`) e prepara uma lista vazia para os resultados.

A seguir, um loop é iniciado, que percorrerá todos os pontos de dados no dataframe de entrada. Para cada ponto de dados, uma "janela" de tempo é definida. 

Esta 'janela' é uma porção do dataframe original, baseado na posição do atual ponto de dados e na largura da janela (determinada por `tau`) e é passada para a função 'cs_detection' com 'tau' e 'theta_c' como argumentos.

O resultado da função 'cs_detection' juntamente com o índice do ponto de dados atual é então anexado à lista de outputs.

## Saída

Finalmente, a função retorna um dataframe onde a primeira coluna, intitulada 'Time', contém os índices dos pontos de dados e a segunda coluna, intitulada 'cs_out', contém os resultados correspondentes da função 'cs_detection'.
 Função     def convert_to_float(s):

**Nome da Função**: convert_to_float

**Descrição**: A função converte uma string em um valor do tipo float. Caso a string contém um 'D', ele será automaticamente convertido para 'E' antes da conversão para um valor do tipo float. 

**Entradas**: 
1. A função aceita um único argumento que é uma string (s). Esta é a string que será convertida em um número de ponto flutuante. 

**Processamento**:
1. A função tenta converter a string em um valor do tipo float usando a função incorporada float() do Python.
2. Se a conversão falhar e lançar um ValueError, a função substituirá 'D' por 'E' na string e tentará novamente a conversão.
3. Nota adicional: 'D' é frequentemente usado em notações científicas para indicar um valor de ponto flutuante. Isso é comummente encontrado em outputs gerados por Fortran e algumas outras linguagens de programação. A função garante a correta conversão expectada nestes casos.

**Saída**:
1. A função retorna um valor do tipo float. Este é o valor de ponto flutuante que é obtido a partir da conversão da string fornecida.
2. Caso a string não possa ser convertida para um número de ponto flutuante mesmo após a substituição de 'D' por 'E', a função lançará um ValueError.

**Exceções**:
1. ValueError: Isso será lançado se a string fornecida não puder ser convertida para um número de ponto flutuante, mesmo após a substituição de 'D' por 'E'.
 Função     def calculate_magnetic_volatility(df, B, tau=50, w=50):

# Documentação resumida

## Entrada
A função `calculate_magnetic_volatility` aceita quatro parâmetros:

1. `df`: Um DataFrame do pandas que contém os dados necessários para o cálculo. A necessidade dos dados ou das colunas específicos dentro do dataframe não é especificada na função original.

2. `B`: Uma string que representa o nome da coluna dentro do DataFrame `df` que contém o campo magnético a ser utilizado no cálculo.

3. `tau`: Um número inteiro opcional a ser usado para τ. Por padrão, é definido como 50.

4. `w`: Um número inteiro opcional que define o tamanho da janela usada para calcular a volatilidade magnética. Por padrão, é definido como 50.


## Processamento
A função primeiro calcula a diferença ("Delta_r_mag") do logaritmo natural do campo magnético (`B`) para um período determinado por `tau`. 

Depois disso, são removidas quaisquer linhas do DataFrame que contenham NaN, que pode ser produzido pelo cálculo anterior. 

Em seguida, a função calcula a volatilidade magnética ("vol_mag") como desvio padrão da diferença do logaritmo natural calculada anteriormente, considerando uma janela de tamanho `w`.

## Saída
A função retorna uma série pandas representando a volatilidade magnética ("vol_mag"). O índice da série corresponderá ao índice do DataFrame de entrada após a remoção das linhas NaN.
 Função     def apply_gaussian_kernel(x_coords, sigma):

**Nome da Função:** apply_gaussian_kernel

---

**Entrada:**

Esta função aceita dois parâmetros:

1. **x_coords**: uma sequência (list ou array) de coordenadas numéricas para as quais a função gaussiana será aplicada.

2. **sigma**: um número real que representa o desvio padrão da distribuição gaussiana usada como núcleo de suavização.

---

**Processamento:** 

A função `apply_gaussian_kernel` suaviza as coordenadas fornecidas aplicando um filtro gaussiano unidimensional sobre elas. [`gaussian_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html) é uma função da biblioteca SciPy, que suaviza uma entrada com uma média rotacional gaussiana com um dado desvio padrão (sigma).

---

**Saída:**

A função retorna `smoothed_x` que é um numpy array da mesma dimensão que o input `x_coords` mas com suas coordenadas suavizadas pela aplicação do filtro gaussiano.

---

**Exemplo de uso:**

```python
import numpy as np

x_coords = np.array([1, 2, 3, 4, 5])
sigma = 1.0

smoothed_coords = apply_gaussian_kernel(x_coords, sigma)
print(smoothed_coords)
```

**Nota:**

Em termos de aplicações práticas, a função `apply_gaussian_kernel` pode ser usada para suavizar dados ruidosos, desempenhar um papel na detecção de bordas em processamento de imagens, etc.
 Função     def declustering_function(data, u=30000, run=10):

Função: declustering_function

Descrição:
A função "declustering_function" realiza um procedimento conhecido como "declustering". Este é um procedimento estatístico frequentemente utilizado em estudos geoespaciais que buscam reduzir a dependência entre observações em tempo e/ou espaço. 

Entrada: 
- data: uma estrutura de dados bidimensional que contém as observações que serão processadas. Espera-se que a estrutura de dados seja um Dataframe de pandas, onde cada linha representa uma observação e cada coluna um atributo da observação.
- u: O valor limite para considerar um cluster. O padrão é 30000. Ele é usado como um parâmetro de controle de densidade para definir qual observação fica ou sai do cluster.
- run: O número de execuções que a função realizará. O padrão é 10.

Processamento:
A função percorre os dados para identificar clusters com base no limite "u" especificado. Em seguida, percorre cada cluster identificado e remove observações redundantes ou excessivamente próximas, reduzindo a dependência entre as observações. Esse processo é repetido para o número especificado de execuções ("run").

Saída:
A função retorna um Dataframe de pandas que contém o conjunto declusterizado de observações. As observações que foram removidas durante o processo de declustering são excluídas desse Dataframe final.

Notas:
- A função presume que os dados estão em uma estrutura espacial ou temporal onde a proximidade entre observações é importante.
- "u" e "run" podem ser ajustados para alterar a severidade e a extensão do declustering.
- A função não modifica o Dataframe original "data", retornando um novo Dataframe com as observações após o declustering.
 Função         # Threshold and run are parameters for the function with default values

## Documentação da função decluster

### Entrada:
A função recebe um DataFrame 'data' com pelo menos uma coluna chamada 'value', e dois parâmetros opcionais: 'u' que define um valor de limiar, e 'run' que especifica uma distância cumulativa na qual um novo cluster será iniciado se a diferença entre duas posições consecutivas for maior. 

### Processamento:
1. A função primeiro computa os Picos Acima do Limiar (POT - Peaks Over Threshold), criando um novo DataFrame 'pot_df' que contém as observações do DataFrame 'data' onde o valor é maior que o limite 'u'.

2. Em seguida, os Valores Abaixo do Limiar (VBT - Values Below Threshold) são computados para completude e armazenados em 'vbt_df'.

3. Introduz-se uma nova coluna 'cluster' em 'pot_df', onde um novo cluster começa se o gap entre as posições for maior que 'run'.

4. A função então calcula os picos acima do limiar desaglomerados (declustered POT), que são os valores máximos em cada cluster.

5. A função realiza um plot dos pontos abaixo do limiar (em preto), acima do limiar (em cinza transparente), pontos declustered (em vermelho), e a linha do limiar (em vermelho tracejado). 

### Saída:
1. A saída principal da função é 'declustered_pot', que é o DataFrame dos valores picos acima do limiar declustered.

2. A outra saída é um gráfico que é exibido com a função plt.show().

3. Além disso, a função imprime o número de clusters encontrados com a forma de 'declustered_pot'.
 Função     def stats_excess(data, threshold):

Função: stats_excess(data, threshold)

**Entrada:**
- `data` (array-like): Um conjunto de dados numéricos aos quais a função será aplicada.
- `threshold` (float): Um valor limite para o qual a função vai extrair valores acima deste. 

**Processamento:**
1. Extrai valores do conjunto de dados que são maiores que o valor limite ('threshold').
2. Calcula a média destes valores excedentes, subtraindo o valor limite de cada um.
3. Calcula o desvio padrão dos valores excedentes, também subtraindo o valor limite de cada um.

**Saída:**
- `mean_excess` (float): A média dos valores excedentes.
- `std_excess` (float): O desvio padrão dos valores excedentes.

**Resumo:**
Esta função serve para calcular a média e o desvio padrão de valores que excedem um determinado limite em um conjunto de dados numéricos. Os cálculos são realizados após a subtração do valor limite. A função retorna a média e o desvio padrão calculado.
 Função     def plot_mean_excess(data, min_thresh, max_thresh, num_threshs=100):

# Documentação da Função `plot_mean_excess`

## Descrição

A função `plot_mean_excess` plota a média do excesso e seu desvio padrão para uma série de limiares, com barras de erro.

## Entrada

A função tem quatro parâmetros: 'data', 'min_thresh', 'max_thresh', 'num_threshs'.

- **data** (list/array): Dados numéricos para os quais a média do excesso e o desvio padrão devem ser calculados. 

- **min_thresh** (float): O valor mínimo do limiar para o qual a média do excesso deve ser calculada.

- **max_thresh** (float): O valor máximo do limiar para o qual a média do excesso deve ser calculada.

- **num_threshs** (int, opcional): O número de limiares entre o mínimo e máximo para os quais a média de excesso deve ser calculada, padrão é 100.

## Processamento

A função realiza as seguintes etapas:

- Cria um array numpy de limiares de 'min_thresh' para 'max_thresh'.

- Calcula a média do excesso e o desvio padrão para cada limiar chamando a função `stats_excess` para cada limiar.

- Cria uma figura e eixos usando o matplotlib.

- Desenha o gráfico usando `errorbar` do matplotlib ao longo dos limiares com a média do excesso e desvio padrão como barras de erro.

## Saída

A função não retorna nenhum valor. Mas gera um gráfico usando os valores calculados. O gráfico mostra a média do excesso ao longo dos limiares, com barras de erro representando o desvio padrão. Os eixos x e y estão rotulados como "Threshold" e "Mean Excess", respectivamente. Uma legenda é adicionada no melhor local. O gráfico é exibido na saída.
 Função     def fit_pot_model(data, min_threshold, max_threshold, num_thresholds):

### Função: fit_pot_model

**Entrada:**

1. *data*: Uma Série ou DataFrame pandas. Os dados aos quais o modelo potenciais superiores ao limite (POT - Peaks Over Threshold) irá se ajustar.

2. *min_threshold*: O valor mínimo do limite.

3. *max_threshold*: O valor máximo do limite.

4. *num_thresholds*: O número de limites.

**Processamento:**

A função começa a criar uma lista vazia chamada "results". Em seguida, cria uma série de limites usando a função np.linspace com os parâmetros min_threshold, max_threshold e num_thresholds.

Para cada limite na série de limites, a função seleciona os dados da entrada que estão acima desse limite e lhes subtrai o valor do limite, resultando nas "excedências".

Em seguida, a função ajusta a Distribuição Generalizada de Pareto (GPD) às "excedências" usando a função genpareto.fit. Os parâmetros da distribuição GPD ajustada (forma, localização e escala) são salvos em um dicionário junto com o valor do limite correspondente.

Esse dicionário é adicionado à lista de "results".

**Saída:**

A função retorna um DataFrame pandas que contém os parâmetros de forma, localização e escala do modelo GPD ajustado para cada um dos limites na série de limites. Isso permite uma análise detalhada dos diferentes modelos potenciais superiores ao limite (POT) que podem ser ajustados aos dados, dependendo do limite escolhido.
 Função     def plot_shape_parameter(results):

# Documentação da Função `plot_shape_parameter`

## Descrição

A função `plot_shape_parameter` é usada para representar visualmente a evolução do parâmetro 'Forma' (Shape) de um modelo Peaks Over Threshold (POT) em relação aos diferentes limiares.

## Entrada

- `results`: um DataFrame do Pandas contendo os parâmetros ajustados do modelo POT para cada limiar. O DataFrame deve ter colunas nomeadas 'Threshold' e 'Shape'. Assumimos que 'results' é o DataFrame retornado pela função `fit_pot_model`.

## Processamento

O `plot_shape_parameter` função realiza as seguintes etapas de processamento:

1. Cria uma nova figura com tamanho de 10x6.
2. Plota o parâmetro 'Forma' (Shape) em função do 'Limiar' (Threshold), marcando cada ponto com um 'o'.
3. Define o título do gráfico como 'Evolution of Shape Parameter over Thresholds'.
4. Define 'Threshold' como a legenda do eixo x e 'Shape Parameter' como a legenda do eixo y.
5. Ativa a grade no gráfico.
6. Exibe o gráfico.

## Saída

A saída é um gráfico de linhas mostrando a evolução do parâmetro 'Shape' dos resultados em função do 'Threshold'. O gráfico é exibido na tela. A função não retorna nada.
 Função     def plot_mean_residual_life(data, thresholds):

## Documentação Resumida

**Função:** plot_mean_residual_life(data, thresholds)

**Descrição:** Esta função é destinada a visualizar a vida residual média do conjunto de dados fixado com limites.

### Entrada:

- **data:** Um array unidimensional do numpy que representa os dados numéricos.
- **thresholds:** Uma lista de limites numéricos.

### Processamento:

1. Inicializa um gráfico com dimensões especificadas.
2. Para cada limite em "thresholds".
    - Calcula a média dos dados que são maiores que o limite e diminui o próprio limite.
    - Adiciona a média calculada à lista "means".
3. Calcula a derivação da "means".
4. Calcula a taxa de mudança da derivação.
5. Encontra o índice onde a taxa de mudança é menor que 0.01. Este índice corresponde ao início aproximado onde a derivação mantém quase constanta.
6. Encontra o limite correspondente a este índice.
7. Desenha a vida residual média, uma linha vertical que representa o início da aproximação constante, e o texto que indica o valor do início da aproximação.
8. Mostra o gráfico criado.

### Saída:

Um gráfico com:

- Eixo X representando os limites.
- Eixo Y representando a vida residual média.
- Uma linha que indica a vida residual média para cada limite.
- Uma linha vertical que indica o início da aproximação constante.
- Um texto que indica o valor do início da aproximação constante ou o limite onde a curva da vida residual média começa a se estabilizar.

Arremessendo um gráfico, esta função não possui um retorno de valor numérico. No entanto, os gráficos gerados podem fornecer insights significativos para os usuários.

## Licença

Este projeto está licenciado sob os termos da licença MIT.
