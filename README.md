# Tech Challenge 04 — Credit Scoring Monitoring

## Observabilidade Contínua, Data Drift, Qualidade de Dados e Governança

Projeto desenvolvido para o Tech Challenge da Fase 4 da Pós-Graduação em Machine Learning Engineering — FIAP.

O objetivo é implementar uma camada de sustentação e confiabilidade para um modelo de Credit Scoring em produção, contemplando:

- validação de qualidade de dados;
- Data Contracts;
- treinamento e rastreabilidade do modelo;
- simulação de mudanças no comportamento dos dados;
- detecção estatística de Data Drift;
- geração de relatório de monitoramento;
- observabilidade do pipeline;
- alertas;
- testes automatizados;
- governança e privacidade;
- princípios da LGPD;
- análise de causalidade e riscos de viés.

---

# 1. Contexto do problema

Modelos de Credit Scoring são utilizados para estimar o risco de inadimplência de clientes.

Porém, mesmo que um modelo apresente bom desempenho no momento do treinamento, o ambiente econômico pode mudar.

Exemplos:

- redução da renda;
- aumento do endividamento;
- inflação;
- desemprego;
- mudança no perfil dos clientes;
- aumento de atrasos em pagamentos.

Essas alterações podem fazer com que os dados recebidos em produção deixem de representar a mesma distribuição utilizada no treinamento.

Esse fenômeno é conhecido como **Data Drift**.

O principal risco é a degradação silenciosa: tecnicamente o sistema continua executando, mas o comportamento dos dados e das previsões muda.

Este projeto cria uma camada de monitoramento para identificar esse cenário.

---

# 2. Arquitetura da solução

Fluxo implementado:

```text
Dataset de Crédito
        |
        v
Reference Dataset
        |
        v
Pandera Data Contract
        |
        +------ Lote inválido ------> BLOQUEADO
        |
        v
Random Forest Baseline
        |
        v
MLflow
Métricas + Parâmetros + Modelo
        |
        v
Production Dataset
        |
        v
Simulação de mudança econômica
        |
        +--> redução de renda
        +--> aumento de endividamento
        +--> aumento de pagamentos atrasados
        |
        v
Data Drift Detection
        |
        +--> Kolmogorov-Smirnov
        +--> Evidently
        |
        v
Relatório HTML + Estatísticas
        |
        v
Pipeline de Observabilidade
        |
        +--> Status
        +--> Volume
        +--> Latência
        +--> Drift
        +--> Alertas
        |
        v
Logs + Governança + LGPD
```

---

# 3. Estrutura do projeto

```text
tech-challenge-04-credit-monitoring/
|
├── data/
│   ├── reference.csv
│   ├── production.csv
│   └── invalid_batch.csv
|
├── logs/
│   └── pipeline.log
|
├── models/
│   └── credit_model.joblib
|
├── reports/
│   ├── drift_report.html
│   └── drift_statistics.csv
|
├── src/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── validate.py
│   ├── train.py
│   ├── simulate_drift.py
│   ├── detect_drift.py
│   └── pipeline.py
|
├── tests/
│   └── test_validation.py
|
├── README.md
├── requirements.txt
└── mlflow.db
```

---

# 4. Dataset

Foi utilizado um dataset sintético e reprodutível de Credit Scoring contendo:

**10.000 registros**

Variáveis:

| Variável | Descrição |
|---|---|
| customer_id | Identificador do cliente |
| age | Idade |
| income | Renda |
| debt_ratio | Grau de endividamento |
| credit_score | Score de crédito |
| employment_years | Tempo de emprego |
| late_payments | Quantidade de pagamentos atrasados |
| loan_amount | Valor solicitado |
| default | Indicador de inadimplência |

O dataset sintético foi escolhido para permitir controle sobre a geração dos dados e, principalmente, sobre os cenários de drift.

A geração utiliza seed fixa, garantindo reprodutibilidade.

A taxa de inadimplência do Reference Dataset foi:

```text
28,84%
```

---

# 5. Reference Dataset e Production Dataset

A solução separa explicitamente dois contextos.

## Reference Dataset

Representa a população conhecida no momento do desenvolvimento e treinamento do modelo.

Arquivo:

```text
data/reference.csv
```

Quantidade:

```text
10.000 registros
```

## Production Dataset

Representa um período posterior, após uma alteração do cenário econômico.

Arquivo:

```text
data/production.csv
```

Quantidade:

```text
5.000 registros
```

Essa separação é fundamental porque o monitoramento de drift compara a distribuição conhecida no desenvolvimento com a distribuição observada posteriormente em produção.

---

# 6. Data Contract com Pandera

Antes que um lote possa seguir pelo pipeline, ele passa por um contrato de dados implementado com **Pandera**.

Entre as regras implementadas estão:

- `customer_id` obrigatório, positivo e único;
- idade entre 18 e 100 anos;
- renda obrigatória e maior que zero;
- `debt_ratio` entre 0 e 1;
- `credit_score` entre 300 e 850;
- tempo de emprego não negativo;
- pagamentos atrasados não negativos;
- valor do empréstimo maior que zero;
- target `default` limitado a 0 ou 1;
- estrutura de colunas estrita.

O contrato também utiliza coerção de tipos.

---

# 7. Teste de bloqueio de dados inválidos

Foi criado propositalmente um lote contendo:

- cliente com 15 anos;
- renda nula;
- credit score igual a 950;
- debt ratio igual a 1.5;
- `customer_id` duplicado.

Resultado:

```text
BLOQUEADO PELO DATA CONTRACT.
A ingestão NÃO deve continuar.
```

O pipeline identifica individualmente as violações encontradas.

Isso evita que dados estruturalmente inválidos contaminem o processo de inferência ou monitoramento.

---

# 8. Testes automatizados

Foram implementados testes com Pytest para validar o comportamento do contrato.

Execução:

```bash
python -m pytest -v
```

Resultado:

```text
6 passed
```

Os testes verificam:

- aprovação do Reference Dataset;
- bloqueio do lote inválido;
- bloqueio de idade inferior a 18;
- bloqueio de renda nula;
- bloqueio de credit score inválido;
- bloqueio de identificadores duplicados.

---

# 9. Modelo baseline

O modelo utilizado como baseline foi:

```text
RandomForestClassifier
```

Configuração:

```text
n_estimators = 200
max_depth = 10
min_samples_leaf = 5
random_state = 42
test_size = 0.20
```

Divisão:

```text
Treino: 8.000 registros
Teste:  2.000 registros
```

---

# 10. Resultados do baseline

Métricas obtidas:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.7405 |
| Precision | 0.6115 |
| Recall | 0.2756 |
| F1-score | 0.3799 |
| ROC-AUC | 0.7121 |

O objetivo deste projeto não é otimizar exaustivamente o classificador, mas estabelecer um baseline rastreável que permita demonstrar monitoramento, drift e governança.

Em um ambiente real de concessão de crédito, métricas como recall, precisão, calibração, custos de falsos positivos/falsos negativos e métricas de fairness deveriam ser avaliadas conforme a política de risco da instituição.

---

# 11. Rastreamento com MLflow

O MLflow foi utilizado para rastrear o experimento do modelo.

São registrados:

- algoritmo;
- hiperparâmetros;
- métricas;
- modelo treinado;
- Run ID;
- histórico da execução.

Experimento:

```text
credit-scoring-baseline
```

O backend local utiliza SQLite:

```text
mlflow.db
```

Para iniciar a interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Depois acessar localmente:

```text
http://127.0.0.1:5000
```

---

# 12. Simulação de Data Drift

Para representar uma deterioração do cenário econômico, foram realizadas mudanças controladas no Production Dataset.

Três variáveis foram alteradas.

## Income

A renda foi reduzida aproximadamente em:

```text
-30,06%
```

Média:

```text
Reference:  5534,5920
Production: 3871,0311
```

## Debt Ratio

O grau de endividamento aumentou aproximadamente:

```text
+69,10%
```

Média:

```text
Reference:  0,2864
Production: 0,4842
```

## Late Payments

A quantidade média de pagamentos atrasados aumentou:

```text
+131,93%
```

Média:

```text
Reference:  1,5007
Production: 3,4806
```

Essas alterações representam um cenário hipotético de deterioração econômica.

---

# 13. Impacto observado nas previsões

Após aplicar o modelo baseline ao Production Dataset:

```text
Taxa prevista de inadimplência: 51,70%
```

Probabilidade média estimada de inadimplência:

```text
49,63%
```

Esse comportamento reforça a necessidade de monitoramento contínuo.

Uma alteração nas previsões, isoladamente, não comprova degradação do modelo. Ela funciona como um sinal operacional que deve ser analisado juntamente com drift, desempenho quando os rótulos reais estiverem disponíveis e contexto de negócio.

---

# 14. Detecção estatística de Data Drift

Foi utilizado o teste de **Kolmogorov-Smirnov (KS)** para comparar as distribuições numéricas do Reference Dataset e Production Dataset.

Hipóteses:

```text
H0: as amostras são provenientes da mesma distribuição.
H1: as distribuições são diferentes.
```

Critério utilizado:

```text
p-value < 0.05 → evidência estatística de drift
```

Resultados:

| Feature | KS | Resultado |
|---|---:|---|
| age | 0.0108 | OK |
| income | 0.3258 | DRIFT |
| debt_ratio | 0.4578 | DRIFT |
| credit_score | 0.0088 | OK |
| employment_years | 0.0049 | OK |
| late_payments | 0.4745 | DRIFT |
| loan_amount | 0.0082 | OK |

Resultado final:

```text
Features com drift: 3/7
```

Detectadas:

```text
income
debt_ratio
late_payments
```

O teste detectou exatamente as três variáveis modificadas intencionalmente na simulação.

As estatísticas completas são armazenadas em:

```text
reports/drift_statistics.csv
```

---

# 15. Evidently

Além do teste estatístico implementado diretamente no projeto, o **Evidently** foi utilizado para gerar um relatório visual de Data Drift.

Arquivo:

```text
reports/drift_report.html
```

O relatório compara o Reference Dataset com o Production Dataset e permite inspecionar alterações nas distribuições das features.

A combinação de Evidently e teste KS oferece duas formas complementares de observabilidade:

- análise visual;
- evidência estatística.

---

# 16. Observabilidade do pipeline

O pipeline centraliza eventos e métricas operacionais em:

```text
logs/pipeline.log
```

São monitorados:

- início do pipeline;
- status de cada etapa;
- sucesso ou falha;
- quantidade de registros processados;
- quantidade de features com drift;
- features afetadas;
- quantidade de alertas;
- duração total;
- taxa prevista de inadimplência;
- probabilidade média prevista.

Exemplo real da execução:

```text
STEP=LOAD_REFERENCE STATUS=SUCCESS RECORDS=10000
STEP=DATA_VALIDATION STATUS=SUCCESS RECORDS=10000
STEP=PRODUCTION_SIMULATION STATUS=SUCCESS RECORDS=5000
STEP=DRIFT_DETECTION STATUS=SUCCESS DRIFTED_FEATURES=3 TOTAL_FEATURES=7
ALERT=DATA_DRIFT DRIFTED_FEATURES=income,debt_ratio,late_payments
STEP=EVIDENTLY_REPORT STATUS=SUCCESS
METRIC=PREDICTED_DEFAULT_RATE VALUE=0.5170
METRIC=AVERAGE_DEFAULT_PROBABILITY VALUE=0.4963
METRIC=PIPELINE_DURATION_SECONDS VALUE=1.1360
METRIC=ALERT_COUNT VALUE=1
PIPELINE STATUS=SUCCESS
```

---

# 17. Métricas de saúde operacional

As principais métricas definidas para sustentação são:

| Métrica | Finalidade |
|---|---|
| Pipeline Status | Identificar sucesso ou falha |
| Task Status | Localizar a etapa responsável por falhas |
| Pipeline Duration | Monitorar latência |
| Records Processed | Monitorar volume |
| Drifted Features | Identificar mudanças de distribuição |
| Alert Count | Monitorar incidentes |
| Predicted Default Rate | Detectar mudanças no comportamento das previsões |
| Average Default Probability | Monitorar alteração do risco estimado |

Em uma implantação real, essas métricas poderiam ser exportadas para Prometheus, Grafana, Datadog ou serviço equivalente.

---

# 18. Estratégia de alertas

Neste protótipo, a presença de feature classificada com drift gera um alerta no log:

```text
ALERT=DATA_DRIFT
```

Em produção, o alerta poderia ser encaminhado para:

- plataforma de observabilidade;
- e-mail;
- Slack ou Microsoft Teams;
- sistema de incidentes;
- equipe de MLOps;
- equipe de risco/modelagem.

A ocorrência de drift não deve provocar automaticamente o retreinamento ou substituição do modelo.

O alerta inicia uma investigação.

---

# 19. Data Drift x Concept Drift

## Data Drift

Ocorre quando a distribuição das variáveis de entrada muda.

Neste projeto isso foi simulado diretamente em:

```text
income
debt_ratio
late_payments
```

## Concept Drift

Concept Drift ocorre quando a relação entre as variáveis de entrada e o resultado que se deseja prever muda.

Exemplo:

Mesmo para clientes com características semelhantes às observadas no treinamento, uma nova condição econômica pode modificar a probabilidade real de inadimplência.

A detecção robusta de Concept Drift exige observar os resultados reais posteriores, ou seja, os rótulos de inadimplência após o período necessário para maturação do crédito.

Por isso, neste protótipo o experimento implementa diretamente Data Drift e monitora mudanças nas previsões como sinal de risco. Em produção, o Concept Drift seria avaliado posteriormente por meio da comparação entre previsões e resultados reais.

---

# 20. Estratégia de resposta ao drift

Ao detectar drift:

1. registrar o evento;
2. identificar as features afetadas;
3. avaliar a magnitude da mudança;
4. verificar qualidade dos dados;
5. analisar contexto econômico e operacional;
6. aguardar/consultar labels reais quando aplicável;
7. recalcular métricas de performance;
8. analisar fairness e possíveis impactos discriminatórios;
9. avaliar necessidade de recalibração ou retreinamento;
10. validar o novo modelo antes de qualquer promoção para produção.

O retreinamento automático não é adotado como primeira resposta porque drift não significa necessariamente que o modelo perdeu capacidade preditiva.

---

# 21. Análise causal

É importante distinguir correlação de causalidade.

Neste experimento, a redução da renda e o aumento do endividamento e dos atrasos foram introduzidos deliberadamente pelo código. Portanto, conhecemos a origem técnica do Data Drift observado.

Entretanto, em um sistema real, detectar estatisticamente uma mudança em `income`, por exemplo, não demonstra que determinada condição econômica causou diretamente uma alteração na inadimplência.

O monitoramento responde:

```text
"O que mudou nos dados?"
```

Uma análise causal tenta responder:

```text
"Por que mudou e qual mecanismo provocou o efeito?"
```

Para uma investigação real seriam necessárias evidências adicionais, como:

- indicadores macroeconômicos;
- inflação;
- desemprego;
- mudanças nas políticas de concessão;
- alterações na origem dos clientes;
- campanhas comerciais;
- mudanças de produto;
- dados temporais;
- experimentos ou métodos causais adequados.

Portanto, o drift funciona como gatilho para investigação, e não como prova automática de causalidade.

---

# 22. Governança de Machine Learning

A utilização de modelos em decisões de crédito exige governança durante todo o ciclo de vida.

O processo proposto inclui:

```text
Dados
  ↓
Validação
  ↓
Treinamento
  ↓
Avaliação
  ↓
Registro
  ↓
Aprovação
  ↓
Produção
  ↓
Monitoramento
  ↓
Investigação
  ↓
Retreinamento controlado
```

Cada versão de modelo deve possuir:

- dataset ou referência da versão dos dados;
- código versionado;
- hiperparâmetros;
- métricas;
- artefato do modelo;
- responsável;
- critérios de aprovação;
- histórico de alterações;
- evidências de testes;
- análise de risco;
- avaliação de viés.

---

# 23. LGPD e Privacy by Design

Este projeto utiliza exclusivamente **dados sintéticos**, portanto não contém dados pessoais reais.

Entretanto, um sistema real de Credit Scoring trataria dados pessoais e deveria ser desenvolvido considerando a Lei Geral de Proteção de Dados Pessoais — LGPD (Lei nº 13.709/2018).

A proteção da privacidade deve ser incorporada desde o desenho da solução, seguindo princípios de **Privacy by Design** e **Privacy by Default**.

Isso significa que a proteção de dados não deve ser adicionada somente após a implantação do sistema.

---

# 24. Tratamento de PII

Em ambiente real, identificadores pessoais devem ser separados das features utilizadas pelo modelo sempre que possível.

Exemplos de PII:

- CPF;
- nome;
- endereço;
- telefone;
- e-mail;
- número de conta.

Medidas recomendadas:

- pseudonimização;
- criptografia em trânsito e em repouso;
- segregação de dados identificáveis;
- controle de acesso baseado em função;
- princípio do menor privilégio;
- trilhas de auditoria;
- monitoramento de acessos;
- mascaramento de dados em ambientes não produtivos.

O modelo deve receber apenas os dados necessários para sua finalidade.

---

# 25. Minimização de dados

O princípio de minimização deve orientar a seleção das features.

A pergunta central é:

```text
"Esta informação é realmente necessária para avaliar o risco?"
```

Dados que não agregam valor legítimo ao processo não devem ser coletados apenas porque estão disponíveis.

Além de reduzir riscos de privacidade, a minimização diminui superfície de ataque, complexidade operacional e possibilidade de uso inadequado.

---

# 26. Base legal e proteção do crédito

Em um cenário real, a organização deve mapear formalmente a hipótese legal aplicável a cada atividade de tratamento.

A LGPD prevê, entre as hipóteses do art. 7º, o tratamento de dados pessoais para **proteção do crédito**.

Isso não elimina as demais obrigações de proteção de dados.

Continuam relevantes:

- finalidade;
- adequação;
- necessidade;
- transparência;
- segurança;
- prevenção;
- não discriminação;
- responsabilização e prestação de contas.

A definição final da base legal e do desenho do tratamento deve ser validada pelas áreas jurídica, de privacidade e compliance da instituição.

---

# 27. Decisões automatizadas e transparência

Credit Scoring pode afetar diretamente os interesses de uma pessoa.

Quando houver decisão tomada unicamente com base em tratamento automatizado de dados pessoais que afete seus interesses, a governança deve contemplar os direitos previstos pela LGPD, inclusive mecanismos relacionados à revisão da decisão e ao fornecimento de informações adequadas sobre critérios e procedimentos utilizados, observados os limites legais aplicáveis.

O processo deve possuir:

- canal de atendimento ao titular;
- rastreabilidade da decisão;
- registro da versão do modelo;
- registro das principais entradas utilizadas;
- possibilidade de investigação;
- procedimento de revisão;
- documentação dos critérios do sistema.

---

# 28. Retenção e descarte

Dados não devem ser armazenados indefinidamente apenas porque podem ser úteis futuramente.

Uma política de retenção deve considerar:

- finalidade original;
- base legal;
- obrigações legais e regulatórias;
- necessidade operacional;
- prazos de defesa de direitos quando aplicáveis;
- requisitos de auditoria.

Ao final do período definido, os dados devem seguir procedimento controlado de:

- eliminação; ou
- anonimização, quando juridicamente e tecnicamente aplicável.

Os prazos específicos devem ser definidos pela instituição conforme sua realidade regulatória e jurídica.

---

# 29. Segurança

Controles recomendados:

- criptografia;
- gestão segura de credenciais;
- IAM e menor privilégio;
- segregação entre desenvolvimento e produção;
- gestão de segredos;
- logs de auditoria;
- backups;
- monitoramento;
- gestão de vulnerabilidades;
- revisão periódica de acessos;
- plano de resposta a incidentes.

Dados pessoais e credenciais nunca devem ser armazenados diretamente no repositório Git.

---

# 30. Viés e não discriminação

Um modelo pode apresentar boa performance global e ainda produzir impactos desiguais entre grupos.

Antes da implantação real, devem ser avaliados:

- representatividade dos dados;
- features potencialmente discriminatórias;
- proxies de atributos sensíveis;
- métricas por subgrupo;
- falsos positivos e falsos negativos por grupo;
- estabilidade das métricas ao longo do tempo.

A presença de drift também pode alterar fairness.

Por isso, avaliações de viés devem fazer parte do monitoramento contínuo e não apenas do treinamento inicial.

---

# 31. Privacy by Default

A configuração padrão deve ser a mais protetiva possível.

Exemplos:

- coletar somente dados necessários;
- acesso negado por padrão;
- retenção limitada;
- ausência de PII em logs;
- ambientes de desenvolvimento com dados sintéticos ou anonimizados;
- compartilhamento restrito;
- monitoramento e auditoria habilitados desde o início.

---

# 32. Arquitetura de produção — Batch x Real-time

Este protótipo executa o monitoramento em **batch**.

## Batch

Adequado para:

- análise diária ou periódica;
- cálculo de drift em janelas temporais;
- consolidação de métricas;
- geração de relatórios;
- retreinamentos programados.

Exemplo:

```text
Dados do período
      ↓
Validação
      ↓
Inferência
      ↓
Drift
      ↓
Relatório
      ↓
Alerta
```

## Real-time

Em um cenário de decisão de crédito online, a inferência pode ocorrer em tempo real.

Fluxo possível:

```text
API
 ↓
Validação
 ↓
Modelo
 ↓
Score
 ↓
Logs/Métricas
 ↓
Monitoramento
```

A detecção estatística de drift normalmente continua utilizando janelas de observações acumuladas, evitando decisões baseadas em uma única requisição.

Uma arquitetura real pode combinar:

```text
Inferência Real-time + Observabilidade contínua + Drift em Batch
```

---

# 33. Execução do projeto

## Criar ambiente virtual

Windows:

```bash
python -m venv .venv
```

Ativar:

```bash
.venv\Scripts\activate
```

Instalar dependências:

```bash
pip install -r requirements.txt
```

---

# 34. Gerar Reference Dataset

```bash
python -m src.prepare_data
```

---

# 35. Executar Data Contract

```bash
python -m src.validate
```

Resultado esperado:

```text
Dataset válido → APROVADO
Lote inválido → BLOQUEADO
```

---

# 36. Treinar baseline

```bash
python -m src.train
```

O modelo será salvo em:

```text
models/credit_model.joblib
```

---

# 37. Simular produção com drift

```bash
python -m src.simulate_drift
```

Arquivo gerado:

```text
data/production.csv
```

---

# 38. Detectar Data Drift

```bash
python -m src.detect_drift
```

Artefatos:

```text
reports/drift_statistics.csv
reports/drift_report.html
```

---

# 39. Executar pipeline monitorado

```bash
python -m src.pipeline
```

Log:

```text
logs/pipeline.log
```

---

# 40. Executar testes

```bash
python -m pytest -v
```

Resultado atual:

```text
6 passed
```

---

# 41. Abrir MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Acessar:

```text
http://127.0.0.1:5000
```

Experimento:

```text
credit-scoring-baseline
```

---

# 42. Principais resultados

O projeto demonstrou:

```text
10.000 registros no Reference Dataset
5.000 registros no Production Dataset
6 testes automatizados aprovados
3 de 7 features com Data Drift
1 alerta operacional de Data Drift
Pipeline executado com SUCCESS
Relatório Evidently gerado
Modelo rastreado pelo MLflow
```

Features com drift:

```text
income
debt_ratio
late_payments
```

Alterações simuladas:

```text
income:        -30,06%
debt_ratio:    +69,10%
late_payments: +131,93%
```

---

# 43. Conclusão

O projeto demonstra que colocar um modelo em produção não encerra o ciclo de Machine Learning.

É necessário monitorar continuamente:

- dados;
- qualidade;
- distribuições;
- previsões;
- infraestrutura;
- logs;
- métricas;
- alertas;
- privacidade;
- riscos;
- governança.

O Data Contract impede a entrada de dados inválidos.

O MLflow mantém a rastreabilidade do baseline.

O teste KS e o Evidently identificam alterações na distribuição dos dados.

O pipeline registra métricas operacionais e gera alertas.

A camada de governança define princípios para tratamento responsável de dados e decisões de crédito.

Assim, a solução deixa de tratar o modelo como um artefato isolado e passa a considerar todo o ciclo de sustentação de um sistema de Machine Learning em produção.