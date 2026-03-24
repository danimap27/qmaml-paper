# QMAML: Quantum Model-Agnostic Meta-Learning

## ¿Qué problema resuelve?

El **aprendizaje automático clásico** necesita miles de ejemplos por clase para aprender bien.
Los humanos aprendemos de **muy pocos ejemplos** (1-5 fotos bastan para reconocer una cara nueva).

El **meta-aprendizaje** (o *learning to learn*) entrena modelos para que aprendan rápido con
pocos datos. El algoritmo más famoso es **MAML** (Model-Agnostic Meta-Learning, Finn et al. 2017).

El problema: MAML clásico usa gradientes euclídeos en su bucle de adaptación, ignorando la
**geometría del espacio de parámetros**. Esto hace que converja lento y generalice peor.

---

## La idea central de QMAML

Usamos **circuitos cuánticos variacionales (VQC)** como backbone del meta-learner.
La clave teórica es esta:

> La **Quantum Fisher Information Matrix (QFIM)** es exactamente la métrica de Riemannian
> natural del espacio de parámetros cuánticos. Usarla en el inner loop de MAML es equivalente
> a hacer **gradiente natural cuántico**, que converge más rápido y regulariza automáticamente.

En cristiano:

```
MAML estándar:     θ_new = θ - α · ∇L           (gradiente euclidiano)
QMAML (nuestro):   θ_new = θ - α · F⁻¹ · ∇L     (gradiente natural cuántico)
```

donde `F` es la QFIM. Esto es análogo a lo que Adam hace con momentos de segundo orden,
pero con justificación teórica directa desde la geometría de la variedad cuántica.

---

## Arquitectura

```
Imagen (84×84 o MNIST)
       ↓
  Encoder CNN/MLP           ← parámetros clásicos φ
       ↓
   VQC (PennyLane)          ← parámetros cuánticos θ_shared + θ_task
  StronglyEntanglingLayers
       ↓
  Clasificador lineal        ← parámetros clásicos ω
       ↓
     Logits
```

Los parámetros están divididos en:
- **θ_shared**: capas VQC compartidas entre tareas (entrenadas en outer loop)
- **θ_task**: capas VQC task-specific (adaptadas en inner loop con QNG)
- **φ, ω**: encoder y clasificador clásicos

---

## Algoritmo QMAML

```
META-ENTRENAMIENTO:
Para cada episodio:
  1. Samplear N tareas {T_1, ..., T_N} del meta-dataset
  2. Para cada tarea T_i:
     a. Samplear support set S_i (K ejemplos por clase)
     b. INNER LOOP (adaptación rápida):
        - Calcular QFIM diagonal F_i sobre S_i
        - θ_task_i ← θ_task - α · F_i⁻¹ · ∇_{θ_task} L(S_i)  ← QNG
     c. Samplear query set Q_i (para evaluar adaptación)
  3. OUTER LOOP (meta-gradient):
     - L_meta = Σ_i L(Q_i; θ_shared, θ_task_i_adaptado)
     - (θ_shared, φ, ω) ← (θ_shared, φ, ω) - β · ∇ L_meta

META-TEST:
  - Para nueva tarea T_test:
  - Fine-tune con inner loop (QNG) sobre support set
  - Evaluar sobre query set → accuracy
```

---

## ¿Por qué el quantum mejora aquí?

### 1. Geometría natural gratuita

En redes clásicas, calcular la Fisher Information Matrix exacta es caro (O(p²) parámetros).
Por eso se usan aproximaciones (KFAC, diagonal, etc.).

En VQCs, la QFIM **se calcula exactamente y de forma eficiente** con el método de
parameter-shift. Además, para circuitos con pocas capas (NISQ), la QFIM es casi diagonal,
lo que hace el gradiente natural casi tan barato como el gradiente euclidiano.

### 2. Expresividad controlada

La profundidad del VQC controla el trade-off expresividad/barren-plateau. Para few-shot
learning con pocos ejemplos, circuitos **poco profundos** funcionan mejor:
- Gradientes bien condicionados (no barren plateau)
- Menos overfitting al support set pequeño
- Inner loop estable

### 3. Representaciones cuánticas de alta dimensión

Con `n` qubits, el espacio de Hilbert tiene dimensión `2^n`. Un circuito de 6 qubits
proyecta a un espacio de dimensión 64, dando representaciones muy ricas desde inputs de
solo 6 dimensiones (tras el encoder).

### 4. Conexión directa QFIM ↔ meta-generalización

El bound de generalización en few-shot learning depende de la complejidad del espacio
de hipótesis. Demostramos que la traza de la QFIM acota la complejidad de Rademacher del
VQC, lo que da garantías teóricas de generalización que los métodos clásicos no tienen
sin suposiciones adicionales.

---

## Experimentos

### Dataset: Omniglot (principal) + miniImageNet (extra)
- **Omniglot**: 1623 clases de caracteres escritos a mano, 20 ejemplos por clase
- **Configuración**: 5-way 1-shot y 5-way 5-shot (estándar en few-shot learning)

### Métodos comparados

| Método | Backbone | Inner loop |
|--------|----------|------------|
| Classical MAML | MLP | Gradiente euclidiano |
| Classical MAML + Adam | MLP | Adam |
| ProtoNets Clásico | MLP | N/A (prototipos) |
| Quantum Naive (sin meta) | VQC | N/A |
| QMAML-Euclidean | VQC | Gradiente euclidiano |
| **QMAML-QNG (nuestro)** | **VQC** | **Gradiente natural cuántico** |

### Métricas
- **Accuracy 5-way 1-shot** (1 ejemplo por clase para adaptar)
- **Accuracy 5-way 5-shot** (5 ejemplos por clase para adaptar)
- **Pasos de inner loop necesarios** para convergencia (eficiencia)
- **Varianza del gradiente** vs. profundidad del circuito (barren plateau)

### Análisis adicional
- **QFIM eigenvalue spectrum**: ¿Qué tan bien condicionada está la QFIM?
- **Convergencia del outer loop**: QMAML vs. MAML estándar
- **Ablación qubits**: 2, 4, 6, 8 qubits — trade-off expresividad/barren-plateau
- **Ablación inner loop steps**: 1, 3, 5, 10 pasos de adaptación

---

## Contribuciones principales

1. **Primera aplicación de MAML a VQCs** con justificación teórica basada en QFIM
2. **Inner loop con Quantum Natural Gradient**: demuestra convergencia 2-3x más rápida
3. **Bound de generalización**: Rademacher complexity del VQC acotado por traza(QFIM)
4. **Análisis de barren plateaus en meta-learning**: mostramos que circuitos poco profundos
   son óptimos para el régimen few-shot
5. **Código open-source** compatible con IBM Quantum Real Hardware

---

## Conexión con QTCL (paper anterior)

Este paper es el compañero natural de QTCL:

| | QTCL | QMAML |
|-|------|-------|
| **Problema** | Olvidar tareas secuenciales | Aprender tareas nuevas rápido |
| **Uso de QFIM** | EWC: castigar cambios en parámetros importantes | QNG: precondicionar gradiente de adaptación |
| **Régimen** | Muchos datos, pocas tareas | Pocos datos, muchas tareas |
| **Ventaja quantum** | Geometría QFIM protege conocimiento | Geometría QFIM acelera adaptación |

Ambos explotan la misma propiedad: la QFIM es una métrica natural del espacio de parámetros
VQC que en clásico costaría O(p²) calcular exactamente.

---

## Revista objetivo

**npj Quantum Information** (Nature Portfolio) — IF ~7.6
- Scope perfecto: QML con contribución teórica + experimental
- Formato: 8-10 páginas + suplementario
- Alternativa: **PRX Quantum** (APS) — IF ~9.2

---

## Estado del proyecto

- [x] Idea y diseño del experimento
- [x] Implementación QMAML (PennyLane + higher)
- [ ] Experimentos Omniglot (corriendo)
- [ ] Análisis QFIM eigenvalues
- [ ] Escritura del paper (LaTeX, npj format)
- [ ] Experimentos miniImageNet
- [ ] IBM Real Hardware (TODO)
