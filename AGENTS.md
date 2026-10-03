# AGENTS.md — aforo-vision

Este archivo le dice a los asistentes de IA (Claude Code, Cursor, GitHub Copilot, etc.) cómo comportarse dentro de este repositorio.

## Qué es este repositorio

`aforo-vision` es el pipeline local de visión por computador del piloto de aforo. Corre en el laptop de Josuram, lee 2 cámaras (una USB, una WiFi), y produce eventos JSON (`ENTRY`/`EXIT` + identidad opcional) que envía por HTTPS a `aforo-backend`. **No** tiene base de datos ni interfaz visual — es un productor de eventos.

Antes de proponer cambios, lee `PRD.md` (qué debe hacer el sistema) y `ARCHITECTURE.md` (cómo está construido, en inglés).

## Reglas de git (obligatorias, sin excepciones)

1. **Ninguna IA puede hacer commit ni push.** Ningún asistente (Claude Code, Cursor, Copilot, Codex, Gemini, etc.) ejecuta `git commit`, `git push`, `git merge`, `git rebase`, `git tag` ni `git reset`, ni crea o fusiona PRs, ni por terminal ni por herramientas MCP/API de GitHub. El agente deja los cambios en el árbol de trabajo y, si ayuda, **propone** el mensaje de commit (formato de `TASKS.md`); el commit y el push los hace siempre una persona.
2. **Nadie puede tener `Co-Authored-By` de una IA.** Ningún commit ni PR puede incluir líneas `Co-Authored-By: Claude ...` (ni de ninguna otra IA), `noreply@anthropic.com` ni "Generated with Claude Code". Esto aplica también a las personas: si el mensaje propuesto trae esa línea, se borra antes de commitear.
3. **Cómo se hace cumplir:**
   - `.githooks/commit-msg` rechaza localmente esos mensajes. Actívalo una vez por clon: `git config core.hooksPath .githooks`.
   - `.github/workflows/no-ai-coauthor.yml` falla en GitHub si algún commit del historial los tiene.
   - `.claude/settings.json` desactiva la coautoría automática de Claude Code y le bloquea `git commit`/`git push`.
   No desactives ni modifiques estos tres archivos sin que el usuario lo pida explícitamente.

## Reglas para el agente

1. **El video nunca sale de este repositorio ni del laptop.** Ningún cambio debe agregar código que suba frames, clips o imágenes crudas a un servicio externo (nube, Amplify, `aforo-backend`, servicios de streaming). Solo el JSON del evento resuelto viaja fuera (ver `ARCHITECTURE.md` sección 4.8-4.9). La única excepción es el servidor de video de la sección 4.10: transmite **solo en la red local**, **solo a usuarios con login** (Cognito, grupos `viewer`/`dev`) y nunca graba.
2. **No inventes un "modelo periocular" propietario.** La identificación periocular se hace recortando la región de los ojos y pasándola por un modelo de embeddings faciales estándar (ArcFace/InsightFace) — no existe un modelo comercial dedicado "periocular", y el código/documentación debe seguir siendo honesto sobre esto.
3. **Mantén ambas cámaras simétricas.** `camera-outside` y `camera-inside` deben correr la misma lógica de detección/tracking/identidad/anti-spoofing. No agregues lógica que sea exclusiva de una sola cámara sin actualizar también la otra y documentar por qué.
4. **La dirección (ENTRY/EXIT) se decide por orden de checkpoint**, no por orientación corporal. Si tocas `src/direction/resolver.py`, no reintroduzcas lógica de "mirando hacia adelante/atrás" como señal primaria — eso ya se descartó por ser vulnerable a caminar de espaldas (ver ADR-003 en `ARCHITECTURE.md`).
5. **Sigue el contrato de evento JSON compartido** definido en `ARCHITECTURE.md` (este repo) y `aforo-backend/ARCHITECTURE.md`. Si cambias un campo del evento, avisa explícitamente que también hay que actualizar `aforo-backend` y `aforo-frontend` — no lo hagas de forma aislada.
6. **No agregues Kafka, MQTT, Docker o colas distribuidas.** Ya se decidió (ADR en `aforo-backend`) que esta pieza no las necesita — es una sola fuente de eventos, de baja frecuencia, para un piloto de un día. Si el agente cree que hacen falta, debe preguntar antes de agregarlas, no asumir.
7. **Corre en CPU, sin GPU dedicada.** Cualquier dependencia o configuración debe funcionar en el laptop de Josuram tal cual, sin asumir CUDA disponible.
8. **Verifica versiones antes de fijar una dependencia nueva.** Antes de agregar una librería a `requirements.txt`, confirma que soporta Python 3.14 y que es la última versión estable — no asumas que una versión recordada de entrenamiento sigue siendo la vigente.
9. **Presupuesto de hardware**: cualquier sugerencia de comprar hardware (cámaras, adaptadores, etc.) debe considerarse dentro de un presupuesto TOTAL de ~500.000 COP para todo el proyecto (compartido con los otros repos), no por componente. Si una sugerencia se acerca o supera ese total, dilo de inmediato, no después de construir toda la propuesta.
10. **Textos para humanos van en español**; nombres de variables, funciones, clases, commits y comentarios de código van en inglés, siguiendo el resto del proyecto.

## Herramientas que el agente puede usar libremente

- Lectura/escritura de archivos dentro de este repositorio.
- Ejecución de tests y scripts locales (`pytest`, scripts en `enrollment/`).
- Búsqueda web para verificar versiones de librerías o APIs de modelos (YOLO26n-pose, ArcFace/InsightFace, OSNet, SORT/DeepSORT).

## Herramientas que requieren confirmación explícita del usuario

- Cualquier llamada de red que no sea a `aforo-backend` (por ejemplo, subir datos a un servicio externo nuevo).
- Instalar dependencias del sistema operativo (fuera de `pip`).
- Modificar `config/pilot.yaml` con credenciales reales de la cámara WiFi o URLs de producción del backend.
- Exponer el servidor de video (sección 4.10) fuera de la red local (port forwarding, túneles tipo ngrok/Cloudflare, `0.0.0.0` en una red pública) o permitir acceso sin token.
