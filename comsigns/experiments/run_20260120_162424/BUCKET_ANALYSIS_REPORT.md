# ComSigns Bucket Analysis Report

## Resumen Ejecutivo

Este análisis clasifica las 505 glosas del dataset AEC según su número de muestras en el set de validación y evalúa el rendimiento del modelo en cada bucket.

---

## Definición de Buckets

| Bucket | Definición | Interpretación |
|--------|-----------|----------------|
| **HEAD** | ≥ 10 muestras | Clases bien representadas |
| **MID** | 3-9 muestras | Clases moderadamente representadas |
| **TAIL** | 1-2 muestras | Clases infrarepresentadas (long-tail) |

---

## Resultados por Bucket (Validation Support)

| Bucket | # Clases | # Muestras | Acc@1 | Acc@5 | Cov@5 | Cov@10 |
|--------|---------|------------|-------|-------|-------|--------|
| HEAD | 3 | 39 | 61.54% | 100.00% | 100.00% | 100.00% |
| MID | 47 | 222 | 10.36% | 25.90% | 25.90% | 36.26% |
| TAIL | 455 | 260 | 0.00% | 0.00% | 0.00% | 0.00% |
| **GLOBAL** | 505 | 521 | ~9% | - | 9.02% | 9.02% |

---

## Análisis Long-Tail

### Distribución del Vocabulario
- **TAIL representa el 90.1% del vocabulario** (455 de 505 clases)
- **TAIL contiene el 49.9% de las muestras** (260 de 521)

### Rendimiento del Modelo en TAIL
- **Accuracy@1 en TAIL: 0.00%** - El modelo nunca predice correctamente una clase TAIL
- **Accuracy@5 en TAIL: 0.00%** - El modelo nunca incluye la clase correcta en top-5
- **Coverage@5 en TAIL: 0.00%** - Ninguna muestra TAIL es "cubierta" por el modelo

### Comparación HEAD vs TAIL
- **Ratio Acc@1:** 615x (HEAD es 615 veces mejor)
- **Ratio Acc@5:** 1000x (HEAD es 1000 veces mejor, TAIL es efectivamente 0)

---

## Diagnóstico

### ⚠️ TAIL es RUIDO
El modelo **NO está aprendiendo nada útil del TAIL**:
- 0% de precisión en todas las métricas
- Las clases TAIL nunca aparecen en las predicciones
- El modelo efectivamente "ignora" el 90% del vocabulario

### Causas Probables
1. **Insuficientes muestras de entrenamiento** (1-2 por clase)
2. **El clasificador no puede generalizar** con tan pocos ejemplos
3. **Gradientes dominados por HEAD/MID** durante el entrenamiento

---

## Estrategias Recomendadas

### 🏆 RECOMENDACIÓN: TAIL → OTHER

**Descripción:** Fusionar todas las clases TAIL en una única clase "OTHER"

| Aspecto | Detalle |
|---------|---------|
| **Implementación** | Remapear labels: TAIL → 0 (OTHER), HEAD/MID → 1..N |
| **Clases resultantes** | 51 (1 OTHER + 50 HEAD/MID) |
| **Beneficio esperado** | Mayor accuracy en HEAD/MID, modelo más estable |

**Pros:**
- Simplifica el problema (505 → 51 clases)
- Elimina el ruido del TAIL
- Mejora el gradient flow hacia clases útiles

**Contras:**
- Pierde granularidad en 455 glosas
- La clase OTHER será muy heterogénea

---

### Alternativa: TAIL EXCLUSION

**Descripción:** Entrenar solo con HEAD + MID, ignorar TAIL completamente

| Aspecto | Detalle |
|---------|---------|
| **Clases** | 50 (3 HEAD + 47 MID) |
| **Muestras** | ~1500 train (estimado), ~260 val |
| **Beneficio** | Señal de entrenamiento más limpia |

**Cuando usar:** Si la clase OTHER se vuelve demasiado grande o heterogénea

---

### Futuro: TAIL FEW-SHOT

**Descripción:** Arquitectura de dos etapas con retrieval para clases raras

| Aspecto | Detalle |
|---------|---------|
| **Etapa 1** | Clasificador HEAD/MID/OTHER |
| **Etapa 2** | Si OTHER → buscar en embeddings TAIL |
| **Requisito** | Embeddings de calidad (verificar con t-SNE/UMAP) |

**Cuando usar:** Si se necesita el vocabulario completo y los embeddings muestran estructura

---

## Próximos Pasos

1. **Implementar TAIL → OTHER**
   - Crear función de remapeo de labels
   - Modificar dataset para agrupar TAIL
   - Reentrenar y comparar métricas

2. **Evaluar calidad de embeddings**
   - Extraer embeddings del encoder
   - Visualizar con t-SNE/UMAP
   - Evaluar si hay clusters por glosa

3. **Considerar data augmentation** para MID
   - Rotation, flip, noise en keypoints
   - Aumentar muestras de 3-9 a ~10+

---

## Archivos Generados

```
experiments/run_20260120_162424/
├── bucket_analysis.json      # Análisis completo en JSON
├── confusion_matrix.png      # Matriz de confusión visual
├── confusion_matrix.csv      # Matriz de confusión en CSV
├── metrics_by_class.json     # Métricas detalladas por clase
└── evaluation_summary.json   # Resumen de evaluación
```

---

## Conclusión

El análisis confirma un **problema severo de long-tail**: el 90% del vocabulario (TAIL) tiene rendimiento de 0%, mientras que solo el 0.6% (HEAD) tiene rendimiento aceptable.

**La recomendación es clara: implementar TAIL → OTHER como primer paso, reduciendo el problema de 505 a ~51 clases efectivas.**

Esto permitirá:
1. Validar que el modelo puede aprender bien las clases HEAD/MID
2. Establecer una baseline realista
3. Explorar estrategias few-shot en el futuro para recuperar granularidad TAIL
