# Control posterior acotado de sesgo y varianza

Este paquete contiene un diagnóstico nuevo de los tres grafos C, no una repetición de C/D ni una validación de la cuadrícula oficial 120×120. El protocolo y su aclaración se escribieron antes de la corrida; la decisión de realizar este control es posterior al conocimiento de la campaña original.

## Archivos y lectura

- `protocol.md`: protocolo original preservado; contiene una secuencia literal `` `n `` antes de la aclaración, sin efecto sobre el cálculo.
- `pilot.json`: tiempos piloto y proyección de 209,3 s.
- `control.py`: cálculo exacto de momentos, muestreo multinomial y métricas.
- `results.json`: metadatos originales, hashes de entradas principales, versiones, tiempo de corrida y contraste Aer.
- `summary.csv`: 24 configuraciones; `pointwise.csv`: 3.456 filas; `replicates.csv`: 12.000 filas.
- `report.py`: regenera figuras separadas de mecanismo y decisión, texto numérico y `note.tex` integrable.
- `bias_variance.pdf/png` y `decision_regret.pdf/png`: figuras verificadas visualmente. La segunda usa una escala vertical común para no magnificar redondeo en G1.
- `audit_control.py` y `audit.json`: revisión independiente de agregados, asignaciones, selección, identidad MSE y referencia ideal statevector en los 432 puntos.
- `review_scientific.md`: interpretación y limitaciones en español.
- `manifest_final.json`: hashes complementarios de archivos de este paquete y fuentes actuales; no sustituye los hashes originales de `results.json`.

## Reproducibilidad y portabilidad

Los scripts resuelven el proyecto a partir de su ubicación y se pueden ejecutar desde cualquier directorio. Mantener esta carpeta dentro de `revision_20261015/` del proyecto, con `src/`, la configuración y las dependencias accesibles. Usan NumPy, Qiskit, Aer, PyYAML y Matplotlib; prueban además el directorio relativo `.venv/Lib/site-packages`, por compatibilidad con la copia recibida. No contienen rutas absolutas de esta máquina. No invocar el ejecutable de la `.venv` copiada si apunta al Python de otra computadora.

Con un Python funcional y las dependencias disponibles:

```text
python -B revision_20261015/bias_variance/control.py --pilot
python -B revision_20261015/bias_variance/control.py
python -B revision_20261015/bias_variance/report.py
python -B revision_20261015/bias_variance/audit_control.py
```

Los dos primeros comandos solo son necesarios para una reproducción deliberada: la corrida completa ya terminó en 69,3 s. El segundo reemplaza los CSV y `results.json` de este diagnóstico. Los dos últimos regeneran exclusivamente reportes/auditoría, sin muestreo nuevo. Todos los resultados se guardan junto a los scripts. El presupuesto B=1024 es por punto, por paisaje; cada paisaje usa 147.456 shots simulados. Las 500 réplicas sirven para caracterizar la distribución de selección, no para aumentar B de una decisión individual.

Para compilar `note.tex`, ejecutar un motor LaTeX con soporte para `geometry`, `amsmath`, `graphicx` y `booktabs` desde esta carpeta, donde están los dos PDF incluidos. Se entrega el fuente integrable; este documento independiente no se compiló. La integración y compilación de main/SI están a cargo del coordinador, que dispone de MiKTeX. Las figuras PDF/PNG sí están generadas y verificadas: cada figura mide 5,2 × 6 pulgadas y usa dos columnas de ruido por tres filas de grafos, para su inclusión a aproximadamente 4,7 pulgadas de ancho.

Las etiquetas de réplica son comunes por índice, pero las semillas y observaciones de métodos distintos son independientes. No son números aleatorios comunes ni observaciones emparejadas con correlación positiva. No se cambió el protocolo, los CSV originales de esta corrida, ni archivos oficiales del proyecto.
