"""Single source of truth for canonical NINO-BRASIL notebooks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nino_brasil.artifact_codes import parse_notebook_code


@dataclass(frozen=True)
class NotebookSpec:
    code: str
    relative_path: str
    title: str
    question: str
    method: str
    hypothesis: str
    context: str = ""
    hypothesis_statement: str = ""
    method_rationale: str = ""
    expected_outputs: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    compact_source: bool = False

    def __post_init__(self) -> None:
        parse_notebook_code(self.code)
        base_text = {
            "title": self.title,
            "question": self.question,
            "method": self.method,
        }
        for field, value in base_text.items():
            if len(value.strip()) < 10:
                raise ValueError(f"{self.code}: catalog field {field} is too short")
        required_text: dict[str, str] = {}
        if self.code.startswith("F3"):
            required_text.update(
                {
                    "context": self.context,
                    "hypothesis_statement": self.hypothesis_statement,
                    "method_rationale": self.method_rationale,
                }
            )
        for field, value in required_text.items():
            if len(value.strip()) < 40:
                raise ValueError(f"{self.code}: scientific field {field} is too short")
        if self.code.startswith("F3") and len(self.expected_outputs) < 2:
            raise ValueError(f"{self.code}: at least two expected outputs are required")
        if self.code.startswith("F3") and len(self.references) < 2:
            raise ValueError(f"{self.code}: at least two scientific references are required")

    @property
    def path(self) -> Path:
        return Path(self.relative_path)


F3_BLOCKS: dict[str, tuple[str, str, str]] = {
    "A": (
        "Índices físicos semanais",
        "Como evoluem historicamente os 31 índices semanais oceânicos e atmosféricos do El Niño?",
        "Séries semanais completas, separadas por família física, com tendência OLS descritiva por variável; comparações inferenciais entre fases permanecem nos notebooks próprios.",
    ),
    "B": (
        "Eventos e ciclo de vida",
        "Quando começam e terminam gênese, crescimento, faixa de pico e decaimento?",
        "Eventos ONI locais simétricos (|ONI|≥0,5 °C por cinco estações) e faixa de pico relativa ao extremo de cada evento.",
    ),
    "C": (
        "Precursores e lags",
        "Quais variáveis antecedem cada período e em quantas semanas?",
        "Correlação defasada condicionada pela fase na semana emissora t−lag, N efetivo AR(1), BH-FDR e unidade independente evento.",
    ),
    "D": (
        "Rigor estatístico",
        "Quais diferenças entre períodos sobrevivem ao controle de multiplicidade?",
        "Friedman pareado por evento, Kendall W e BH-FDR separado para o sinal analisado.",
    ),
    "E": (
        "Sensibilidade temporal",
        "Os lags selecionados permanecem estáveis quando eventos inteiros são retirados?",
        "Bootstrap por evento e leave-one-event-out; amostras semanais não aumentam o número de eventos independentes.",
    ),
    "F": (
        "Kelvin, SLA e acoplamento",
        "Como SSH/SLA, termoclina e vento evoluem nas quatro fases?",
        "Compostos evento-fase orientados pelo sinal; La Niña usa anomalias frias/upwelling, não janelas quentes de El Niño.",
    ),
    "G": (
        "Compostos de SSTA",
        "Como intensidade e duração diferem entre classes do mesmo sinal?",
        "Compostos por evento e classe baseada na magnitude absoluta do pico, mantendo o sinal físico original.",
    ),
    "H": (
        "Gênese e precursores por classe",
        "Quais variáveis na gênese separam eventos fracos, moderados, fortes e muito fortes?",
        "Médias por evento na gênese e contraste entre classes; classificação usa |ONI| para Niño e Niña separadamente.",
    ),
    "I": (
        "Interpretação integrada",
        "Quais variáveis reúnem efeito entre fases, antecipação e estabilidade?",
        "Integração auditável dos rankings FDR, lags e estabilidade sem criar nova família confirmatória pós-hoc.",
    ),
    "K": (
        "PCA por período",
        "Quais modos conjuntos dominam cada período do evento?",
        "PCA descritiva sobre médias evento-fase padronizadas; evento, não semana, é a unidade.",
    ),
    "L": (
        "Caracterização das quatro fases",
        "Qual é a assinatura final de gênese, crescimento, pico e decaimento?",
        "Síntese de duração, nível físico, variabilidade e incerteza entre eventos do sinal isolado.",
    ),
}


F3_SCIENTIFIC_DETAILS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "A": (
        "Este notebook apresenta a série histórica semanal completa das 31 variáveis do Pacífico tropical que sustentam a análise de El Niño. Os índices são separados em painéis oceânicos e atmosféricos; para cada variável, o leitor vê o valor semanal original do master, unidade, fonte e tendência linear de longo prazo. O objetivo é sanidade e contexto físico da série, não testar ainda diferenças entre fases.",
        "Os índices oceânicos e atmosféricos exibem trajetórias históricas observáveis e tendências descritivas próprias; essas tendências não devem ser confundidas com efeito de fase, causalidade ou poder preditivo do El Niño.",
        "Cada série é mantida na unidade do master semanal e recebe uma tendência OLS calculada sobre todo o período disponível, apenas como referência visual. A separação oceano/atmosfera evita escalas incompatíveis no mesmo eixo. Como semanas são autocorrelacionadas, a tendência não recebe p-valor nem é usada como teste; evento-fase, lags e robustez são avaliados nos blocos F3NinoC–E.",
        (
            "painel completo dos índices oceânicos semanais, com série original e tendência OLS por variável",
            "painel completo dos índices atmosféricos semanais, com série original e tendência OLS por variável",
            "tabelas longas auditáveis com cada semana, unidade, fonte, representação e tendência correspondente",
        ),
    ),
    "B": (
        "Este notebook constrói o catálogo local de eventos e a segmentação do ciclo de vida que serão usados por todas as análises posteriores. Gênese, crescimento, faixa de pico e decaimento precisam ter definições reproduzíveis e específicas do sinal, sem emprestar semanas ou limiares do sinal oposto.",
        "Os eventos de {signal} identificados pela regra ONI local formam trajetórias coerentes em quatro fases, e o pico é melhor representado por uma faixa relativa ao extremo de cada evento do que por uma única semana.",
        "A regra |ONI| ≥ 0,5 °C por pelo menos cinco estações móveis delimita eventos; a polaridade seleciona apenas {signal}. A faixa de pico é derivada do extremo interno do evento e submetida a sensibilidade, separando caracterização retrospectiva de qualquer alvo preditivo.",
        (
            "catálogo de eventos com início, fim, intensidade e duração",
            "rótulo semanal de gênese, crescimento, faixa de pico e decaimento",
            "tabela de sensibilidade da largura da faixa de pico",
        ),
    ),
    "C": (
        "Este notebook testa antecedência física: em vez de correlacionar toda a série sem considerar o ciclo, relaciona cada variável na semana emissora t−lag ao estado do {signal} na semana receptora t, separadamente em cada fase.",
        "Variáveis associadas à recarga oceânica e ao acoplamento vento-oceano antecedem mudanças do {signal} por lags positivos e com direção física coerente, mas o conjunto e o lag dominante podem variar entre gênese, crescimento, pico e decaimento.",
        "A varredura de lags usa pares condicionados pela fase emissora, número efetivo de graus de liberdade sob AR(1) e BH-FDR dentro da família declarada. O evento permanece a unidade de suporte; o maior |r| só é selecionado depois do controle de multiplicidade.",
        (
            "varredura completa variável × fase × lag com r, N efetivo, p e q",
            "melhor lag significativo de cada precursor e fase",
            "tabela que conserva também os resultados nulos após FDR",
        ),
    ),
    "D": (
        "Este notebook pergunta quais variáveis realmente mudam entre as quatro fases dentro dos mesmos eventos. Ele funciona como filtro inferencial do comportamento multivariado, não como repetição do ranking de correlações do notebook anterior.",
        "Para {signal}, ao menos parte das 31 variáveis apresenta distribuições diferentes entre as quatro fases com tamanho de efeito consistente; outras variáveis permanecerão indistinguíveis e deverão ser registradas como resultado nulo.",
        "Friedman é usado porque as quatro fases são medidas repetidas dentro do evento e não se presume normalidade. Kendall W quantifica o tamanho do efeito e BH-FDR controla a família de 31 testes; semanas não entram como réplicas independentes.",
        (
            "estatística de Friedman, p, q BH-FDR e Kendall W para cada variável",
            "ranking apenas dos discriminantes confirmados no sinal isolado",
            "registro explícito das variáveis sem evidência de diferença entre fases",
        ),
    ),
    "E": (
        "Este notebook avalia se os lags e efeitos selecionados são propriedades recorrentes do sinal ou consequências de poucos eventos extremos. A estabilidade entre eventos é indispensável porque a série semanal contém muitas observações, mas poucos eventos independentes.",
        "Os precursores fisicamente robustos de {signal} mantêm direção e uma faixa de lag semelhante quando eventos completos são reamostrados ou retirados; relações dominadas por um único evento devem perder estabilidade.",
        "O bootstrap reamostra eventos inteiros e repete a seleção de lag dentro de cada réplica; o leave-one-event-out mede influência individual. Essa estratégia respeita dependência intrassistêmica e não converte semanas correlacionadas em tamanho amostral artificial.",
        (
            "frequência de seleção e distribuição do lag em bootstrap por evento",
            "diagnóstico leave-one-event-out de direção e influência",
            "identificação auditável de relações estáveis e instáveis",
        ),
    ),
    "F": (
        "Este notebook examina a ponte dinâmica entre vento zonal, SSH/SLA, profundidade da termoclina e evolução térmica durante o {signal}. O objetivo é verificar coerência com propagação equatorial e com o mecanismo de recarga/descarga, sem chamar automaticamente qualquer anomalia de onda de Kelvin.",
        "A sequência de vento, SLA/SSH e termoclina durante {signal} apresenta polaridade e ordenamento temporal coerentes com {kelvin_process}, distinguindo gênese, crescimento, pico e decaimento.",
        "São construídos compostos por evento-fase e diagnósticos longitudinais/temporais. A interpretação de propagação exige coerência de sinal, direção e antecedência entre variáveis; permanece diagnóstica quando não existe detector formal de onda.",
        (
            "trajetórias evento-fase de vento, SLA/SSH e termoclina",
            "diagnóstico temporal/longitudinal da propagação equatorial",
            "tabela de coerência física e limitações da interpretação de Kelvin",
        ),
    ),
    "G": (
        "Este notebook descreve a diversidade interna dos eventos de {signal}, separando magnitude, duração e forma da trajetória. Classes de intensidade são comparadas somente dentro do mesmo sinal para não transformar assimetria El Niño–La Niña em diferença de classe.",
        "Eventos mais intensos de {signal} apresentam trajetórias de SSTA/ONI, duração e amplitude sistematicamente diferentes, embora localização e evolução espacial possam variar entre eventos da mesma classe.",
        "Os compostos são calculados por evento e por classe definida pela magnitude absoluta do pico, preservando a polaridade original. Resumos entre eventos, e não todas as semanas agrupadas, sustentam a comparação.",
        (
            "compostos de SSTA/ONI por classe de intensidade",
            "distribuições de duração, taxa de crescimento e magnitude",
            "registro da variabilidade entre eventos dentro de cada classe",
        ),
    ),
    "H": (
        "Este notebook concentra-se na gênese e pergunta se o estado precursor já contém informação sobre a intensidade futura do {signal}. A análise deve evitar vazamento: a caracterização da gênese usa apenas a janela definida para essa fase, ainda que a classe seja um rótulo retrospectivo do evento.",
        "Diferenças no pré-condicionamento oceânico e atmosférico durante a gênese separam parcialmente as classes futuras de {signal}, sobretudo em variáveis ligadas à recarga, termoclina e acoplamento do vento.",
        "Calculam-se médias por evento na gênese e contrastes entre classes, com classificação pela magnitude absoluta do pico. O resultado é diagnóstico de separabilidade e não uma estimativa operacional de previsão.",
        (
            "estado das 31 variáveis na gênese para cada evento",
            "contrastes e tamanhos de efeito entre classes de intensidade",
            "lista de candidatos precursores sem alegação automática de causalidade ou skill",
        ),
    ),
    "I": (
        "Este notebook integra três dimensões que não podem ser confundidas: diferença entre fases, antecedência temporal e estabilidade entre eventos. Uma variável só ganha interpretação forte quando as evidências convergem e a direção permanece fisicamente plausível.",
        "Um subconjunto das 31 variáveis reúne, para {signal}, efeito entre fases, lag antecedente e estabilidade entre eventos; variáveis apoiadas por apenas uma dimensão permanecem evidência parcial.",
        "A síntese cruza tabelas confirmatórias já produzidas, sem reabrir testes nem criar um novo limiar pós-hoc. Cada conclusão aponta para efeito, lag e estabilidade que a sustentam, preservando resultados nulos e divergências.",
        (
            "matriz integrada variável × fase com efeito, lag e estabilidade",
            "classificação transparente entre evidência convergente, parcial e ausente",
            "tabela de rastreabilidade de cada interpretação às análises de origem",
        ),
    ),
    "K": (
        "Este notebook investiga a covariação entre as 31 variáveis sem contar variáveis colineares como evidências independentes. A PCA é aplicada às médias evento-fase e descreve modos conjuntos específicos de cada período do {signal}.",
        "Gênese, crescimento, pico e decaimento de {signal} são dominados por combinações distintas de recarga oceânica, estado térmico e acoplamento atmosférico, expressas por cargas diferentes nos componentes principais.",
        "As variáveis são padronizadas com parâmetros do conjunto analisado e a unidade é o evento-fase. A PCA é descritiva, não EOF espacial nem teste de hipótese; variância explicada e loadings devem ser lidos junto com o pequeno número de eventos.",
        (
            "variância explicada e critério de retenção por fase",
            "loadings das 31 variáveis em cada componente e fase",
            "interpretação física dos modos com ressalva explícita de incerteza amostral",
        ),
    ),
    "L": (
        "Este notebook fecha a caracterização do {signal} reunindo duração, nível físico, variabilidade e incerteza das quatro fases. Ele não mistura o sinal oposto nem converte a descrição retrospectiva em previsão.",
        "O {signal} possui assinatura multivariada e duração características em gênese, crescimento, faixa de pico e decaimento, mas com dispersão relevante entre eventos que deve acompanhar qualquer valor médio.",
        "A síntese utiliza somente tabelas do escopo isolado, resume eventos com peso igual e apresenta intervalos e dispersão entre eventos. Resultados de Friedman, lags e PCA são contextualizados sem serem fundidos em um único escore arbitrário.",
        (
            "painel final das quatro fases do sinal isolado",
            "duração e incerteza entre eventos por fase",
            "quadro das variáveis características, discriminantes e limitações",
        ),
    ),
}


F3_REFERENCES: dict[str, tuple[str, ...]] = {
    "A": (
        "Bjerknes, J. (1969). Atmospheric Teleconnections from the Equatorial Pacific. Monthly Weather Review, 97, 163–172.",
        "Trenberth, K. E. (1997). The Definition of El Niño. Bulletin of the American Meteorological Society, 78, 2771–2777.",
    ),
    "B": (
        "Trenberth, K. E. (1997). The Definition of El Niño. Bulletin of the American Meteorological Society, 78, 2771–2777.",
        "Timmermann, A. et al. (2018). El Niño–Southern Oscillation complexity. Nature, 559, 535–545.",
    ),
    "C": (
        "Jin, F.-F. (1997). An Equatorial Ocean Recharge Paradigm for ENSO. Journal of the Atmospheric Sciences, 54, 811–829.",
        "Meinen, C. S.; McPhaden, M. J. (2000). Observations of Warm Water Volume Changes in the Equatorial Pacific. Journal of Climate, 13, 3551–3559.",
    ),
    "D": (
        "Benjamini, Y.; Hochberg, Y. (1995). Controlling the False Discovery Rate. Journal of the Royal Statistical Society B, 57, 289–300.",
        "Bretherton, C. S. et al. (1999). The Effective Number of Spatial Degrees of Freedom. Journal of Climate, 12, 1990–2009.",
        "Wilks, D. S. (2016). The Stippling Shows Statistically Significant Grid Points. Bulletin of the American Meteorological Society, 97, 2263–2273.",
    ),
    "E": (
        "Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure. Ecography, 40, 913–929.",
        "Wilks, D. S. (2016). The Stippling Shows Statistically Significant Grid Points. Bulletin of the American Meteorological Society, 97, 2263–2273.",
    ),
    "F": (
        "Bjerknes, J. (1969). Atmospheric Teleconnections from the Equatorial Pacific. Monthly Weather Review, 97, 163–172.",
        "Cui, J. et al. (2025). Mixed Layer and Oceanic Kelvin Wave Response to Equatorial Pacific Westerly Wind Events. Journal of Geophysical Research: Oceans.",
    ),
    "G": (
        "Timmermann, A. et al. (2018). El Niño–Southern Oscillation complexity. Nature, 559, 535–545.",
        "Trenberth, K. E. (1997). The Definition of El Niño. Bulletin of the American Meteorological Society, 78, 2771–2777.",
    ),
    "H": (
        "Jin, F.-F. (1997). An Equatorial Ocean Recharge Paradigm for ENSO. Journal of the Atmospheric Sciences, 54, 811–829.",
        "Meinen, C. S.; McPhaden, M. J. (2000). Observations of Warm Water Volume Changes in the Equatorial Pacific. Journal of Climate, 13, 3551–3559.",
    ),
    "I": (
        "Timmermann, A. et al. (2018). El Niño–Southern Oscillation complexity. Nature, 559, 535–545.",
        "Cawley, G. C.; Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias. Journal of Machine Learning Research, 11, 2079–2107.",
    ),
    "K": (
        "Jolliffe, I. T.; Cadima, J. (2016). Principal component analysis: a review and recent developments. Philosophical Transactions A, 374, 20150202.",
        "Bretherton, C. S. et al. (1999). The Effective Number of Spatial Degrees of Freedom. Journal of Climate, 12, 1990–2009.",
    ),
    "L": (
        "Bjerknes, J. (1969). Atmospheric Teleconnections from the Equatorial Pacific. Monthly Weather Review, 97, 163–172.",
        "Jin, F.-F. (1997). An Equatorial Ocean Recharge Paradigm for ENSO. Journal of the Atmospheric Sciences, 54, 811–829.",
    ),
}


# Diretrizes F3 El Niño (2026-07): a análise toma como base os El Niños com
# anomalia de pico acima de 1 °C (classes moderado, forte e muito forte),
# segmenta o ciclo em gênese, crescimento, faixa de pico e decaimento, e
# quantifica sensibilidade, conjuntos descritores, influência percentual e
# variáveis-guia de transição por fase.  La Niña permanece no protocolo
# simétrico geral (todos os eventos |ONI| ≥ 0,5 °C).
F3_NINO_OVERRIDES: dict[str, dict[str, object]] = {
    "A": {
        "context": (
            "Este notebook apresenta a série histórica semanal completa das 31 variáveis do "
            "Pacífico tropical que sustentam a análise do El Niño. Os índices são separados em "
            "painéis oceânicos e atmosféricos; para cada variável, o leitor vê o valor semanal "
            "original do master, unidade, fonte e tendência linear de longo prazo. Esta é a base "
            "de contexto físico sobre a qual os blocos seguintes isolam os El Niños com pico "
            "acima de 1 °C."
        ),
    },
    "B": {
        "title": "Ciclo de vida dos El Niños fortes (pico ≥ 1 °C)",
        "question": (
            "Quando começam e terminam gênese, crescimento, faixa de pico e decaimento nos "
            "El Niños com anomalia de pico acima de 1 °C?"
        ),
        "method": (
            "Eventos ONI locais (|ONI| ≥ 0,5 °C por cinco estações) com elegibilidade adicional "
            "de pico ≥ 1,0 °C; faixa de pico relativa ao extremo de cada evento; catálogo completo "
            "preservado com a flag de elegibilidade."
        ),
        "context": (
            "Este notebook constrói o catálogo local de eventos e a segmentação do ciclo de vida "
            "usados por todas as análises posteriores. A diretriz do escopo El Niño toma como base "
            "os eventos considerados fortes — anomalia de pico acima de 1 °C (classes moderado, "
            "forte e muito forte) — porque são eles que carregam sinal físico suficiente para a "
            "caracterização robusta das quatro fases. Gênese, crescimento, faixa de pico e "
            "decaimento recebem definições reproduzíveis e específicas do sinal quente."
        ),
        "hypothesis_statement": (
            "Os El Niños com pico acima de 1 °C formam trajetórias coerentes em quatro fases, e o "
            "pico é melhor representado por uma faixa relativa ao extremo de cada evento do que "
            "por uma única semana."
        ),
        "method_rationale": (
            "A regra |ONI| ≥ 0,5 °C por cinco estações delimita os eventos; a elegibilidade "
            "pico ≥ 1,0 °C seleciona a base de análise sem apagar os eventos fracos do catálogo, "
            "que permanecem com flag para auditoria. A gênese é a janela pré-onset (bibliografia "
            "de recarga), e a faixa de pico é derivada do extremo interno com sensibilidade "
            "80/90/95%."
        ),
        "expected_outputs": (
            "catálogo completo de eventos com intensidade, classe e flag de elegibilidade (pico ≥ 1 °C)",
            "rótulo semanal de gênese, crescimento, faixa de pico e decaimento dos eventos elegíveis",
            "linha do tempo de cada evento e sensibilidade conjunta das quatro fases para gênese 13/26/39 semanas e pico 80/90/95%",
        ),
        "references": (
            "NOAA/CPC. Oceanic Niño Index (ONI v5): Niño 3.4, limiar ±0,5 °C por cinco estações móveis sobrepostas.",
            "Trenberth, K. E. (1997). The Definition of El Niño. Bulletin of the American Meteorological Society, 78, 2771–2777.",
            "Timmermann, A. et al. (2018). El Niño–Southern Oscillation complexity. Nature, 559, 535–545.",
            "Jin, F.-F. (1997). An Equatorial Ocean Recharge Paradigm for ENSO. Journal of the Atmospheric Sciences, 54, 811–829.",
        ),
    },
    "C": {
        "context": (
            "Este notebook testa antecedência física nos El Niños com pico acima de 1 °C: em vez "
            "de correlacionar toda a série sem considerar o ciclo, relaciona cada variável na "
            "semana emissora t−lag ao estado do El Niño na semana receptora t, separadamente em "
            "cada uma das quatro fases. O resultado descreve associação antecedente com o nível "
            "térmico dentro da fase; não é apresentado como detector da fronteira entre fases."
        ),
    },
    "D": {
        "title": "Sensibilidade das variáveis por fase (rigor estatístico)",
        "question": (
            "Quais variáveis mudam de forma estatisticamente defensável entre as quatro fases dos "
            "El Niños fortes, e com que grau de sensibilidade?"
        ),
        "context": (
            "Este notebook mede o grau de sensibilidade de cada uma das 31 variáveis às quatro "
            "fases dos El Niños com pico acima de 1 °C, dentro dos mesmos eventos. Friedman "
            "pareado por evento com Kendall W fornece o tamanho de efeito que alimenta o peso "
            "discriminante usado na influência percentual do bloco I."
        ),
    },
    "F": {
        "title": "Kelvin, Hovmöller e picos de aquecimento e vento",
        "question": (
            "A sequência vento → SLA/SSH → termoclina → SSTA dos El Niños fortes é compatível com "
            "propagação equatorial de Kelvin, e onde ficam os picos de aquecimento e de vento de "
            "cada evento?"
        ),
        "method": (
            "Hovmöller da SSTA equatorial 2S–2N por evento elegível com epicentro e pico de vento "
            "marcados; pulsos de SLA no Pacífico oeste com chegada estimada a 2,4 m/s nas janelas "
            "com SSH; compostos evento-fase de vento, SSH e termoclina."
        ),
        "context": (
            "Este notebook examina a ponte dinâmica entre rajadas de vento de oeste, SLA/SSH, "
            "profundidade da termoclina e a evolução térmica dos El Niños com pico acima de 1 °C. "
            "Cada evento elegível ganha um painel Hovmöller com o epicentro do aquecimento "
            "(extremo espaço-temporal da SSTA 2S–2N) e o pico da anomalia de vento zonal "
            "marcados, permitindo ler diretamente o ordenamento vento → oceano que caracteriza o "
            "acoplamento de Bjerknes e a propagação de Kelvin de downwelling."
        ),
        "hypothesis_statement": (
            "Nos El Niños com pico acima de 1 °C, o pico da anomalia de vento de oeste antecede o "
            "epicentro do aquecimento, e as janelas com SSH mostram pulsos de SLA propagando de "
            "oeste para leste em velocidade compatível com ondas de Kelvin de downwelling."
        ),
        "method_rationale": (
            "A leitura de propagação exige coerência de sinal, direção e antecedência entre vento, "
            "SLA e termoclina; sem detector formal de onda, o diagnóstico permanece qualitativo e "
            "cada seta é ancorada em um pulso auditável de SLA no Pacífico oeste. O SSH diário "
            "cobre apenas as janelas 1997/98, 2015/16 e 2023+, e essa limitação é registrada."
        ),
        "expected_outputs": (
            "painéis Hovmöller por evento elegível com epicentro do aquecimento e pico de vento marcados",
            "tabela de pulsos de SLA e chegada estimada no leste (diagnóstico de Kelvin)",
            "trajetórias evento-fase de vento, SLA/SSH e termoclina",
        ),
        "references": (
            "Bjerknes, J. (1969). Atmospheric Teleconnections from the Equatorial Pacific. Monthly Weather Review, 97, 163–172.",
            "Kessler, W. S.; McPhaden, M. J.; Weickmann, H. K. (1995). Forcing of intraseasonal Kelvin waves in the equatorial Pacific. Journal of Geophysical Research: Oceans, 100, 10613–10631.",
            "Jin, F.-F. (1997). An Equatorial Ocean Recharge Paradigm for ENSO. Journal of the Atmospheric Sciences, 54, 811–829.",
        ),
    },
    "G": {
        "title": "Compostos de SSTA por classe de intensidade",
        "question": (
            "Como a trajetória de SSTA difere entre El Niños moderados, fortes e muito fortes, do "
            "início ao fim do ciclo?"
        ),
        "method": (
            "Compostos por evento alinhados ao pico (−52 a +52 semanas), resumidos por classe de "
            "intensidade dos eventos elegíveis (pico ≥ 1 °C), com dispersão entre eventos."
        ),
        "context": (
            "Este notebook descreve a diversidade interna dos El Niños fortes, separando "
            "magnitude, duração e forma da trajetória térmica. As classes moderado (1,0–1,5 °C), "
            "forte (1,5–2,0 °C) e muito forte (≥ 2,0 °C) são comparadas somente dentro do sinal "
            "quente, alinhadas ao pico de cada evento."
        ),
        "hypothesis_statement": (
            "El Niños mais intensos apresentam trajetórias de SSTA com crescimento mais rápido, "
            "pico maior e decaimento mais prolongado, embora a dispersão entre eventos da mesma "
            "classe permaneça relevante."
        ),
        "method_rationale": (
            "Os compostos são calculados por evento e por classe definida pela magnitude do pico, "
            "preservando a polaridade original; resumos entre eventos — e não semanas agrupadas — "
            "sustentam a comparação, e a dispersão acompanha cada média."
        ),
        "expected_outputs": (
            "composto de SSTA Niño 3.4 por classe de intensidade com dispersão",
            "curva média de todas as classes analisadas como referência",
            "número de eventos independentes por classe acompanhando cada média",
        ),
        "references": (
            "Timmermann, A. et al. (2018). El Niño–Southern Oscillation complexity. Nature, 559, 535–545.",
            "Trenberth, K. E. (1997). The Definition of El Niño. Bulletin of the American Meteorological Society, 78, 2771–2777.",
        ),
    },
    "H": {
        "context": (
            "Este notebook concentra-se na gênese dos El Niños com pico acima de 1 °C e pergunta "
            "se o estado precursor já contém informação sobre a intensidade futura. A análise "
            "evita vazamento: a caracterização usa apenas a janela de gênese, ainda que a classe "
            "seja um rótulo retrospectivo do evento."
        ),
    },
    "I": {
        "title": "Influência percentual e variáveis-guia de transição",
        "question": (
            "Qual o percentual de influência de cada variável na condição do El Niño e quais "
            "variáveis servem de guia para cada mudança de fase?"
        ),
        "method": (
            "Síntese auditável dos rankings FDR, lags e estabilidade; influência percentual como "
            "participação relativa das variáveis físicas, com o alvo SSTA excluído (peso descritivo "
            "por fase e peso discriminante Kendall W); "
            "marcadores de transição = mudanças pareadas entre fases adjacentes, Wilcoxon com "
            "BH-FDR e top-3 diversificado por família física."
        ),
        "context": (
            "Este notebook integra três dimensões que não podem ser confundidas: diferença entre "
            "fases, antecedência temporal e estabilidade entre eventos. Sobre essa base, quantifica "
            "o percentual de influência de cada variável na condição do El Niño — como participação "
            "relativa transparente, por fase e entre fases — e destaca marcadores físicos que "
            "mudam consistentemente entre fases adjacentes dos eventos elegíveis."
        ),
        "hypothesis_statement": (
            "Um subconjunto das 31 variáveis concentra a maior parte do peso descritivo e "
            "discriminante do ciclo dos El Niños fortes, e variáveis ligadas à recarga oceânica, "
            "ao acoplamento vento-oceano e à atmosfera mudam de forma consistente nas transições "
            "adjacentes."
        ),
        "method_rationale": (
            "A influência percentual é derivada de quantidades já auditadas (|média z| por fase e "
            "Kendall W do Friedman), após excluir a própria SSTA alvo, e normalizadas para somar "
            "100% — participação relativa, não "
            "variância explicada causal. Os marcadores de transição usam diferenças pareadas de "
            "médias evento-fase, Wilcoxon e BH-FDR por transição; o alvo térmico é excluído e a "
            "seleção limita redundância por família."
        ),
        "expected_outputs": (
            "matriz integrada variável × fase com efeito, lag e estabilidade",
            "tabela de influência percentual por variável física, fase e métrica, com alvo excluído",
            "top-3 marcadores por transição adjacente com efeito pareado, consistência, q e família física",
        ),
        "references": (
            "Timmermann, A. et al. (2018). El Niño–Southern Oscillation complexity. Nature, 559, 535–545.",
            "Meinen, C. S.; McPhaden, M. J. (2000). Observations of Warm Water Volume Changes in the Equatorial Pacific. Journal of Climate, 13, 3551–3559.",
            "Cawley, G. C.; Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias. Journal of Machine Learning Research, 11, 2079–2107.",
        ),
    },
    "K": {
        "context": (
            "Este notebook investiga a covariação entre as 31 variáveis nos El Niños com pico "
            "acima de 1 °C, sem contar variáveis colineares como evidências independentes. A PCA "
            "é aplicada às médias evento-fase e descreve os modos conjuntos específicos de cada "
            "período do ciclo."
        ),
    },
    "L": {
        "title": "Caracterização final e conjuntos de variáveis por fase",
        "question": (
            "Qual é a assinatura final de cada fase do El Niño forte e qual conjunto de variáveis "
            "a descreve com mais precisão?"
        ),
        "method": (
            "Síntese de duração, nível físico e incerteza entre eventos elegíveis; conjunto "
            "descritor por fase = até cinco variáveis físicas pelo score sensibilidade × √Kendall W, "
            "excluindo a SSTA-alvo e limitando representantes redundantes da mesma família."
        ),
        "context": (
            "Este notebook fecha a caracterização do El Niño forte reunindo duração, nível físico, "
            "variabilidade e incerteza das quatro fases, e responde qual conjunto de variáveis "
            "descreve cada fase com mais precisão — com a justificativa explícita de sensibilidade "
            "entre eventos e capacidade discriminante, separada por família física."
        ),
        "hypothesis_statement": (
            "Cada fase do El Niño forte possui um conjunto descritor distinto: a gênese é dominada "
            "por variáveis de recarga subsuperficial, o crescimento e o pico pelo conteúdo térmico "
            "superior e pelo acoplamento, e o decaimento pela descarga e pela resposta atmosférica."
        ),
        "method_rationale": (
            "A sensibilidade |média z|/desvio z privilegia variáveis com sinal forte e consistente "
            "entre eventos independentes; o Kendall W confirma capacidade discriminante entre "
            "fases. O alvo térmico permanece sinalizado como referência, mas não integra o conjunto "
            "físico descritor; limites por família reduzem a multiplicação de proxies colineares."
        ),
        "expected_outputs": (
            "painel final das quatro fases com duração e incerteza entre eventos",
            "conjunto descritor de cada fase com sensibilidade, família e justificativa",
            "sinalização explícita do alvo térmico como referência excluída da seleção",
        ),
        "references": (
            "Jin, F.-F. (1997). An Equatorial Ocean Recharge Paradigm for ENSO. Journal of the Atmospheric Sciences, 54, 811–829.",
            "Bjerknes, J. (1969). Atmospheric Teleconnections from the Equatorial Pacific. Monthly Weather Review, 97, 163–172.",
        ),
    },
}


def _f3_specs() -> list[NotebookSpec]:
    specs: list[NotebookSpec] = []
    for signal, folder, label in (
        ("Nino", "fase3_nino", "El Niño"),
        ("Nina", "fase3_nina", "La Niña"),
    ):
        for block, (title, question, method) in F3_BLOCKS.items():
            context, hypothesis_statement, method_rationale, expected_outputs = (
                F3_SCIENTIFIC_DETAILS[block]
            )
            substitutions = {
                "signal": label,
                "kelvin_process": (
                    "pulsos de downwelling e aprofundamento da termoclina"
                    if signal == "Nino"
                    else "descarga/upwelling e soerguimento da termoclina"
                ),
            }
            overrides = F3_NINO_OVERRIDES.get(block, {}) if signal == "Nino" else {}
            title = str(overrides.get("title", title))
            question = str(overrides.get("question", question))
            method = str(overrides.get("method", method))
            context = str(overrides.get("context", context))
            hypothesis_statement = str(
                overrides.get("hypothesis_statement", hypothesis_statement)
            )
            method_rationale = str(
                overrides.get("method_rationale", method_rationale)
            )
            expected_outputs = tuple(
                overrides.get("expected_outputs", expected_outputs)
            )
            references = tuple(overrides.get("references", F3_REFERENCES[block]))
            code = f"F3{signal}{block}"
            slug = {
                "A": "indices_fisicos_semanais",
                "B": "alvo_eventos_ciclo_vida",
                "C": "precursores_lags",
                "D": "rigor_estatistico",
                "E": "sensibilidade_temporal",
                "F": "kelvin_sla",
                "G": "compostos_ssta",
                "H": "genese_precursores_classe",
                "I": "interpretacao_integrada",
                "K": "pca_crescimento",
                "L": "caracterizacao_fases",
            }[block]
            specs.append(
                NotebookSpec(
                    code=code,
                    relative_path=f"notebooks/{folder}/{code}_{slug}.ipynb",
                    title=f"{code} — {title}: {label}",
                    question=question,
                    method=method,
                    hypothesis="HIP0",
                    context=context.format(**substitutions),
                    hypothesis_statement=hypothesis_statement.format(**substitutions),
                    method_rationale=method_rationale.format(**substitutions),
                    expected_outputs=expected_outputs,
                    references=references,
                    # Os notebooks F3Nino são os artefatos científicos de leitura
                    # direta: o runner oficial persiste neles tabelas e figuras inline.
                    compact_source=(signal == "Nino"),
                )
            )
    return specs


CANONICAL_NOTEBOOKS: tuple[NotebookSpec, ...] = tuple(
    [
        NotebookSpec(
            "F2Z",
            "notebooks/fase2/F2Z_sanidade_variaveis.ipynb",
            "F2Z — Sanidade das variáveis semanais contratadas",
            "Todas as variáveis físicas contratadas possuem cobertura, unidade e continuidade adequadas?",
            "Auditoria de cobertura, distribuição, continuidade temporal e contratos de unidade antes de qualquer inferência.",
            "HIP0",
            context=(
                "A Fase 2 recebe fontes atualizadas pela Fase 1, trata e organiza uma matriz semanal comum e disponibiliza os produtos. O notebook documenta cobertura real, data máxima válida, defasagem por fonte, unidades, calendários, lacunas, transformações, gráficos de sanidade e validações independentes."
            ),
            hypothesis_statement=(
                "A matriz semanal contém cobertura e continuidade suficientes para comparar eventos ENSO sem que lacunas, unidades incompatíveis ou mudanças de fonte expliquem artificialmente os padrões atribuídos ao ciclo."
            ),
            method_rationale=(
                "A auditoria é descritiva e contratual porque qualidade de entrada não deve ser inferida pelo desempenho posterior. Cobertura por variável, intervalos temporais, unidades, valores ausentes e descontinuidades são avaliados explicitamente e registrados antes da Fase 3."
            ),
            expected_outputs=(
                "inventário completo das variáveis contratadas com fonte, unidade e cobertura temporal",
                "diagnóstico de lacunas, duplicidades, descontinuidades e faixas implausíveis",
                "frescor real por variável, sem confundir extensão do eixo com dado válido",
                "validação CTD/WOD de UFS+GLORYS e auditoria das malhas IBGE quando disponível",
            ),
        ),
        NotebookSpec(
            "F2V",
            "notebooks/fase2/F2V_validacao_insitu.ipynb",
            "F2V — Validação independente in situ",
            "Qual é a cobertura observacional disponível em CTD/WOD, TAO/TRITON e Argo para validar UFS+GLORYS?",
            "Inventário dos Zarrs observacionais, cobertura temporal e vertical e agregação comparativa semanal W-SUN sem preenchimento artificial.",
            "HIP0",
        ),
        *_f3_specs(),
        NotebookSpec(
            "F4NinoC",
            "notebooks/fase4_nino/F4NinoC_sinal_pixel_lags.ipynb",
            "F4NinoC — Resposta CHIRPS pixel-a-pixel",
            "Qual lag semanal liga cada fase emissora do El Niño à resposta de cada pixel brasileiro?",
            "Correlação por pixel na grade CHIRPS nativa; fase em t−lag, resposta em t; N efetivo, BH-FDR e significância de campo.",
            "HIP1",
        ),
        NotebookSpec(
            "F4NinoD",
            "notebooks/fase4_nino/F4NinoD_clusters_alvo.ipynb",
            "F4NinoD — Padrões espaciais e extremos",
            "Quais pixels compartilham respostas e como chuva, seca e extremos variam por fase?",
            "Clusters descritivos após a estimativa pixel-a-pixel e estabilidade leave-one-event-out; nenhum cluster substitui a inferência por pixel.",
            "HIP1",
        ),
        NotebookSpec(
            "F4NinaC",
            "notebooks/fase4_nina/F4NinaC_sinal_pixel_lags.ipynb",
            "F4NinaC — Resposta CHIRPS pixel-a-pixel",
            "Qual lag semanal liga cada fase emissora da La Niña à resposta de cada pixel brasileiro?",
            "Correlação por pixel na grade CHIRPS nativa; fase em t−lag, resposta em t; N efetivo, BH-FDR e significância de campo.",
            "HIP1",
        ),
        NotebookSpec(
            "F4NinaD",
            "notebooks/fase4_nina/F4NinaD_clusters_alvo.ipynb",
            "F4NinaD — Padrões espaciais e extremos",
            "Quais pixels compartilham respostas e como chuva, seca e extremos variam por fase?",
            "Clusters descritivos após a estimativa pixel-a-pixel e estabilidade leave-one-event-out; nenhum cluster substitui a inferência por pixel.",
            "HIP1",
        ),
        NotebookSpec(
            "F5A",
            "notebooks/fase5/F5A_ciclo_ml.ipynb",
            "F5A — RF/XGBoost do ciclo ENSO",
            "Quais variáveis antecipam tipo, fase, intensidade e duração do evento?",
            "Rolling-origin por evento, embargo, baselines, RF/XGBoost e augmentation somente no treino com ablação ON/OFF.",
            "HIP2",
        ),
        NotebookSpec(
            "F6A",
            "notebooks/fase6/F6A_brasil_ml.ipynb",
            "F6A — RF/XGBoost por pixel CHIRPS",
            "Qual skill espacial RF/XGBoost acrescenta sobre a referência estatística?",
            "Validação temporal por evento e avaliação na grade CHIRPS nativa, sem tratar pixels-semanas como eventos independentes.",
            "HIP3",
        ),
        NotebookSpec(
            "F7A",
            "notebooks/fase7/F7A_ciclo_convlstm.ipynb",
            "F7A — ConvLSTM do ciclo ENSO",
            "A estrutura espaço-temporal do Pacífico melhora a previsão do ciclo?",
            "ConvLSTM com folds por evento, embargo, comparação pareada com F5 e augmentation somente no treino.",
            "HIP4",
        ),
        NotebookSpec(
            "F8A",
            "notebooks/fase8/F8A_brasil_convlstm.ipynb",
            "F8A — ConvLSTM para chuva e seca no Brasil",
            "A rede melhora previsão e calibração por pixel sobre F4/F6?",
            "Saída probabilística na grade CHIRPS nativa, validação por evento, skill espacial e calibração de intervalos.",
            "HIP5",
        ),
    ]
)

NOTEBOOK_BY_CODE = {spec.code: spec for spec in CANONICAL_NOTEBOOKS}


def specs_for_phase(phase: int, enso_type: str | None = None) -> list[NotebookSpec]:
    selected: list[NotebookSpec] = []
    for spec in CANONICAL_NOTEBOOKS:
        parsed = parse_notebook_code(spec.code)
        if parsed.phase != int(phase):
            continue
        if enso_type is not None and parsed.enso_type != enso_type:
            continue
        selected.append(spec)
    return selected
