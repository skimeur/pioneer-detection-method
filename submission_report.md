# Submission: Análisis de Detección de Pioneros en Inflación Europea

## Parte A: Pesos Pioneros PDM en el Panel Europeo

En esta parte, se calculan los pesos pioneros utilizando el método de ángulos (PDM) para el panel completo de inflación mensual de 11 países europeos (DE, FR, IT, ES, NL, BE, AT, PT, IE, FI, GR) desde enero de 2000 hasta diciembre de 2025. Los pesos representan la contribución relativa de cada país como "pionero" en la evolución de la inflación del panel. Un peso positivo indica que el país lidera en aumentos de inflación, mientras que un peso negativo sugiere liderazgo en disminuciones.

### Evolución Temporal de los Pesos
- El heatmap (`pioneer_weights_heatmap.png`) muestra la evolución mensual de los pesos, donde colores rojos indican pesos positivos altos y azules pesos negativos altos. Se observan periodos con pesos claros (e.g., alrededor de 2008-2012) y otros con pesos débiles o nulos, indicando ausencia de un pionero dominante.
- El gráfico de líneas para los top 5 países (`pioneer_weights_lines_top5.png`) destaca a NL, IT, FR, BE y AT como los países con mayor promedio absoluto de pesos, mostrando fluctuaciones a lo largo del tiempo.

### Promedios por Subperiodo
Los pesos promedio por subperiodo se resumen en la siguiente tabla (valores extraídos de `pioneer_weights_subperiod_table.csv`):

| País | I_2002_2007 | II_2008_2012 | III_2013_2019 | IV_2020_2021 | V_2022_2023 | VI_2024_2025 |
|------|-------------|--------------|----------------|--------------|-------------|--------------|
| AT   | 0.093      | 0.107        | 0.113          | 0.131        | 0.000       | 0.028        |
| BE   | 0.099      | 0.038        | 0.080          | 0.035        | 0.208       | 0.120        |
| DE   | 0.087      | 0.043        | 0.047          | 0.093        | 0.019       | 0.067        |
| ES   | 0.061      | 0.088        | 0.052          | 0.063        | 0.126       | 0.051        |
| FI   | 0.040      | 0.078        | 0.111          | 0.156        | 0.118       | 0.101        |
| FR   | 0.057      | 0.100        | 0.132          | 0.082        | 0.075       | 0.083        |
| GR   | 0.086      | 0.047        | 0.055          | 0.030        | 0.125       | 0.048        |
| IE   | 0.032      | 0.084        | 0.085          | 0.051        | 0.056       | 0.000        |
| IT   | 0.130      | 0.064        | 0.078          | 0.052        | 0.000       | 0.063        |
| NL   | 0.152      | 0.105        | 0.069          | 0.107        | 0.127       | 0.067        |
| PT   | 0.079      | 0.080        | 0.059          | 0.031        | 0.020       | 0.082        |

**Interpretación clave:**
- En el periodo I (2002-2007), NL tiene el mayor peso promedio (0.152), seguido de IT (0.130) y BE (0.099), indicando que estos países lideraban los movimientos de inflación en Europa.
- En II (2008-2012), FR emerge con mayor peso (0.100), posiblemente reflejando impactos de la crisis financiera.
- En III (2013-2019), FR mantiene liderazgo (0.132), con FI y AT también relevantes.
- En IV (2020-2021), FI destaca (0.156), coincidiendo con la pandemia.
- En V (2022-2023), BE tiene el mayor peso (0.208), seguido de NL (0.127) y ES (0.126), reflejando shocks energéticos.
- En VI (2024-2025), BE y FR dominan, con pesos alrededor de 0.12 y 0.08.
- Se observa una dispersión de liderazgo: NL e IT dominan temprano, FR y FI en medio, y BE/ES en periodos recientes, sugiriendo cambios en dinámicas económicas europeas.

## Parte B: Forecasting de Francia usando Pooling de Expertos

En esta parte, se trata a Francia (FR) como la variable objetivo, utilizando los demás países europeos como "expertos" para predecir su inflación mensual. Se comparan varios métodos de pooling: PDM (ángulos y distancia), Granger, correlación retardada, regresión lineal multivariada, entropía de transferencia, pooling lineal, y mediana.

### Pionero Dominante en Ventanas Móviles
- El gráfico `dominant_pioneer_fr_rolling.png` muestra el país experto dominante en ventanas móviles de 24 meses. Alemania (DE) aparece frecuentemente como dominante, especialmente en periodos recientes, sugiriendo que los movimientos de inflación en Alemania influyen fuertemente en Francia. Otros países como NL, IT y BE también aparecen en rachas, indicando cambios en la influencia.

### Comparación de RMSE
Los RMSE (raíz del error cuadrático medio) se calculan para la predicción de Francia, tanto para la muestra completa como por subperiodo (valores de `fr_forecast_rmse.csv`):

| Método       | RMSE_all | RMSE_I | RMSE_II | RMSE_III | RMSE_IV | RMSE_V | RMSE_VI |
|--------------|----------|--------|---------|----------|---------|--------|---------|
| PDM_angles   | 1.327    | 0.854  | 0.752   | 0.665    | 0.791   | 3.429  | 1.495   |
| PDM_distance | 1.327    | 0.854  | 0.752   | 0.665    | 0.791   | 3.428  | 1.495   |
| Granger      | 0.922    | 0.527  | 0.523   | 0.384    | 0.527   | 2.435  | 1.244   |
| LagCorr      | 0.915    | 0.584  | 0.443   | 0.372    | 0.576   | 2.369  | 1.189   |
| LinReg       | 0.888    | 0.545  | 0.461   | 0.369    | 0.547   | 2.302  | 1.172   |
| TransEnt     | 0.919    | 0.666  | 0.452   | 0.404    | 0.604   | 2.271  | 1.213   |
| LinPool      | 0.915    | 0.587  | 0.443   | 0.372    | 0.576   | 2.362  | 1.191   |
| Median       | 0.867    | 0.614  | 0.489   | 0.405    | 0.600   | 2.185  | 1.133   |

**Interpretación clave:**
- El método con menor RMSE global es **Median** (0.867), seguido de **LinReg** (0.888) y **LagCorr** (0.915). Los métodos PDM tienen RMSE más altos (~1.327), indicando menor precisión en este contexto.
- En subperiodos estables (I-III), métodos como Granger y LinReg funcionan mejor (RMSE ~0.4-0.5), mientras que en periodos volátiles (V: 2022-2023), todos los métodos tienen RMSE altos (~2-3), reflejando la dificultad de predecir shocks.
- Los métodos basados en correlaciones (LagCorr, LinReg) superan a PDM en la mayoría de subperiodos, sugiriendo que relaciones lineales simples capturan mejor la influencia de expertos en Francia que el enfoque pionero.
- Alemania como pionero dominante (de la Parte B1) se alinea con RMSE bajos en métodos que ponderan expertos, indicando su relevancia para predecir Francia.

Este análisis destaca la utilidad del método PDM para identificar líderes en paneles, pero resalta limitaciones en forecasting cuando se usan como pesos de pooling, donde métodos más simples como la mediana pueden ser superiores.</content>
<parameter name="filePath">c:\Users\mgmorenogonz\OneDrive - Université Paris 1 Panthéon-Sorbonne\FTD\Quantitative methods\pioneer-detection-method\submission_report.md